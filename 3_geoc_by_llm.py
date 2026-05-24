#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：cnipa_address 
@File    ：3_geoc_by_llm.py
@IDE     ：PyCharm 
@Author  ：Fan Zhang
@Note    : geocoding by LLM and GeoNames-based lexicon.
'''


import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from openai import OpenAI


# ============================================================
# 0. File paths and settings
# ============================================================

ADDRESS_CSV = Path("path/to/address_non_mainland_cn.csv")
LEXICON_CSV = Path("path/to/geonames_lexicon_cities500_normalized.csv")
FINAL_OUTPUT_CSV = Path("path/to/address_llm_geonames_geocoded.csv")

ENCODING = "utf-8"

ADDRESS_COLUMN = "address"

MODEL = "gpt-5.4-mini"
MAX_RETRIES = 6
BACKOFF_BASE = 1.5
BACKOFF_JITTER = 0.5
SLEEP_BETWEEN_CALLS = 0.0

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # input OpenAI API key here


# ============================================================
# 1. LLM prompt
# ============================================================

SYSTEM_PROMPT = """你是一个全球地址结构化解析器。你将收到一条地址字符串，可能包含中英混杂、音译误差、拼写错误、别名、缩写等。

任务：将地址解析到统一层级并输出尽可能中文、规范化的结果。

规则：
1. 按该国家或地区真实行政区划体系判断 admin1-admin5，不要只按原始写法机械拆分。
2. admin1为一级行政区划，admin2为二级行政区划（优先表示普遍意义上的城市），例如香港、首尔、伦敦、东京等。把更细层级下沉到 admin3 及以下。
3. 兼容跳级地址：若原地址省略中间层级，可根据通常的行政区划归属合理补全缺失上级。
4. 允许将音译误差、拼写错误、别名、缩写识别为对应的标准地名；不要编造缺乏依据的低层级信息。
5. 如有歧义，优先赋值给更大、更稳妥的行政区划级别。
6. 输出中文常用标准写法，即只保留 country_norm_zh、admin1_norm_zh、admin2_norm_zh、admin3_norm_zh、admin4_norm_zh、admin5_norm_zh。无法确定时输出 null。
7. 台湾按省级处理，对应 admin1；香港、澳门按城市级处理，对应 admin2。
8. confidence 范围为 0~1。识别越明确、层级越稳定，confidence 越高；存在音译误差、跳级补全、歧义或较强推断时应降低。
9. matched_place_type 表示当前结果中最具体且相对可信的匹配层级，只能取：
   country, admin1, admin2, admin3, admin4, admin5, unknown
10. 只输出 JSON，不要输出其他文字；JSON 必须可解析，使用双引号；无法确定的字段输出 null；不要遗漏字段。

输出 JSON schema：
{
  "raw_address": "原始地址字符串",
  "country_norm_zh": null,
  "admin1_norm_zh": null,
  "admin2_norm_zh": null,
  "admin3_norm_zh": null,
  "admin4_norm_zh": null,
  "admin5_norm_zh": null,
  "confidence": 0.0,
  "notes": "",
  "matched_place_type": "unknown"
}
""".strip()


USER_PROMPT_TEMPLATE = """请解析下面的地址字符串并输出 JSON。

要求：
- 按真实行政区划体系判断 admin 层级，不要只按原始写法机械拆分。
- admin1 对应一级行政区；admin2 优先保留城市层级，不要把区、县、郡、洞、街道等更细层级提前放入 admin2。
- 如果原地址省略了中间层级，可根据通常的行政区划归属合理补全缺失上级。
- 将音译误差、拼写错误、别名、缩写匹配到对应的标准地名，输出中文常用标准写法。
- 如有歧义，优先保守地赋值给更大层级。
- 台湾按 admin1处理；香港、澳门按 admin2处理。
- 除 JSON 外不要输出任何内容。

地址：{address}
""".strip()


# ============================================================
# 2. LLM helpers
# ============================================================

def safe_load_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from model output."""
    text = str(text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    first = text.find("{")
    last = text.rfind("}")

    if first != -1 and last != -1 and last > first:
        candidate = text[first:last + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


def normalize_matched_place_type(value: Any) -> str:
    """Normalize matched place type."""
    allowed = {
        "country",
        "admin1",
        "admin2",
        "admin3",
        "admin4",
        "admin5",
        "unknown",
    }

    if value is None:
        return "unknown"

    value = str(value).strip().lower()
    return value if value in allowed else "unknown"


def empty_llm_result(address: str, error: str = "") -> Dict[str, Any]:
    """Return an empty normalized address result."""
    return {
        "raw_address": address,
        "country_norm_zh": None,
        "admin1_norm_zh": None,
        "admin2_norm_zh": None,
        "admin3_norm_zh": None,
        "admin4_norm_zh": None,
        "admin5_norm_zh": None,
        "confidence": 0.0,
        "notes": "",
        "matched_place_type": "unknown",
        "llm_error": error,
    }


def call_llm_parse_address(address: str) -> Dict[str, Any]:
    """Normalize one address using the OpenAI API."""
    required_keys = [
        "raw_address",
        "country_norm_zh",
        "admin1_norm_zh",
        "admin2_norm_zh",
        "admin3_norm_zh",
        "admin4_norm_zh",
        "admin5_norm_zh",
        "confidence",
        "notes",
        "matched_place_type",
    ]

    if not str(address).strip():
        return empty_llm_result(address="", error="empty address")

    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                input=USER_PROMPT_TEMPLATE.format(address=address),
                text={"format": {"type": "json_object"}},
                reasoning={"effort": "low"},
                max_output_tokens=2000,
            )

            obj = safe_load_json(response.output_text)

            if obj is None:
                return empty_llm_result(
                    address=address,
                    error="json parse failed",
                )

            for key in required_keys:
                if key not in obj:
                    obj[key] = None

            if not obj.get("raw_address"):
                obj["raw_address"] = address

            try:
                obj["confidence"] = float(obj.get("confidence") or 0.0)
            except Exception:
                obj["confidence"] = 0.0

            obj["matched_place_type"] = normalize_matched_place_type(
                obj.get("matched_place_type")
            )
            obj["llm_error"] = ""

            return obj

        except Exception as exc:
            last_error = repr(exc)

        sleep_seconds = (BACKOFF_BASE ** (attempt - 1)) + random.random() * BACKOFF_JITTER
        time.sleep(sleep_seconds)

    return empty_llm_result(
        address=address,
        error=f"failed after retries: {last_error}",
    )


# ============================================================
# 3. GeoNames lexicon helpers
# ============================================================

ADMIN_SUFFIXES = [
    "特别行政区", "自治州", "自治区", "自治县", "自治旗", "地区", "盟",
    "省", "市", "州", "县", "区", "旗", "郡", "道", "府", "厅",
    "町", "村", "乡", "镇", "大区", "专区", "岛", "城",
]


def strip_admin_suffix(value) -> str:
    """Remove common administrative suffixes."""
    text = "" if value is None else str(value).strip()

    if not text:
        return ""

    for suffix in sorted(ADMIN_SUFFIXES, key=len, reverse=True):
        if text.lower().endswith(suffix.lower()) and len(text) > len(suffix):
            return text[:-len(suffix)]

    return text


def split_aliases(value) -> list:
    """Split alias string and remove duplicates while preserving order."""
    text = "" if value is None else str(value).strip()

    if not text:
        return []

    aliases = [x.strip() for x in text.split("|") if x.strip()]

    seen = set()
    out = []

    for alias in aliases:
        if alias not in seen:
            seen.add(alias)
            out.append(alias)

    return out


def match_alias_position(alias_key_list, input_key) -> Optional[int]:
    """Return alias position if input_key is matched."""
    if not input_key:
        return None

    for i, alias in enumerate(alias_key_list):
        if alias == input_key:
            return i

    return None


def empty_coord_result() -> Dict[str, str]:
    """Return empty coordinate matching result."""
    return {
        "matched_country_code": "",
        "matched_country_name": "",
        "matched_admin1_code": "",
        "matched_admin1_name": "",
        "matched_city_geonameid": "",
        "matched_city_name": "",
        "coord_level": "",
        "matched_latitude": "",
        "matched_longitude": "",
        "match_method": "",
    }


# ============================================================
# 4. Build GeoNames matching indexes
# ============================================================

def load_lexicon() -> Dict[str, Any]:
    """Load GeoNames lexicon and build matching indexes."""
    lex = pd.read_csv(
        LEXICON_CSV,
        dtype=str,
        keep_default_na=False,
        encoding=ENCODING,
    )

    required_columns = [
        "level",
        "country_code",
        "country_zh_name",
        "admin1_code",
        "admin1_zh_name",
        "city_geonameid",
        "city_zh_name",
        "latitude",
        "longitude",
    ]

    for col in required_columns:
        if col not in lex.columns:
            raise ValueError(f"Required column not found in lexicon: {col}")

    lex["country_code"] = lex["country_code"].str.replace(
        r"^(TW|HK|MO)$",
        "CN",
        regex=True,
    )
    lex["row_order"] = range(len(lex))

    country_rows = lex[lex["level"] == "country"].copy().reset_index(drop=True)
    admin1_rows = lex[lex["level"] == "admin1"].copy().reset_index(drop=True)
    city_rows = lex[lex["level"] == "city"].copy().reset_index(drop=True)

    country_rows["country_alias_list"] = country_rows["country_zh_name"].apply(split_aliases)

    admin1_rows["country_alias_list"] = admin1_rows["country_zh_name"].apply(split_aliases)
    admin1_rows["admin1_alias_list"] = admin1_rows["admin1_zh_name"].apply(split_aliases)
    admin1_rows["admin1_alias_key_list"] = admin1_rows["admin1_alias_list"].apply(
        lambda values: [strip_admin_suffix(x) for x in values]
    )

    city_rows["country_alias_list"] = city_rows["country_zh_name"].apply(split_aliases)
    city_rows["admin1_alias_list"] = city_rows["admin1_zh_name"].apply(split_aliases)
    city_rows["admin1_alias_key_list"] = city_rows["admin1_alias_list"].apply(
        lambda values: [strip_admin_suffix(x) for x in values]
    )
    city_rows["city_alias_list"] = city_rows["city_zh_name"].apply(split_aliases)
    city_rows["city_alias_key_list"] = city_rows["city_alias_list"].apply(
        lambda values: [strip_admin_suffix(x) for x in values]
    )

    country_code_index = {}

    for row in country_rows.itertuples(index=False):
        for alias in row.country_alias_list:
            if alias:
                country_code_index.setdefault(alias, []).append(row.country_code)

    for key, values in country_code_index.items():
        country_code_index[key] = list(dict.fromkeys(values))

    country_by_code = {
        country_code: list(sub.itertuples(index=False))
        for country_code, sub in country_rows.groupby("country_code", sort=False)
    }

    admin1_by_code = {
        country_code: list(sub.itertuples(index=False))
        for country_code, sub in admin1_rows.groupby("country_code", sort=False)
    }

    city_by_code = {
        country_code: list(sub.itertuples(index=False))
        for country_code, sub in city_rows.groupby("country_code", sort=False)
    }

    return {
        "country_code_index": country_code_index,
        "country_by_code": country_by_code,
        "admin1_by_code": admin1_by_code,
        "city_by_code": city_by_code,
    }


# ============================================================
# 5. Coordinate matching
# ============================================================

def fill_city_result(result: Dict[str, str], row, coord_level: str, method: str) -> Dict[str, str]:
    """Fill result with city-level match."""
    result["matched_country_code"] = row.country_code
    result["matched_country_name"] = row.country_zh_name
    result["matched_admin1_code"] = row.admin1_code
    result["matched_admin1_name"] = row.admin1_zh_name
    result["matched_city_geonameid"] = row.city_geonameid
    result["matched_city_name"] = row.city_zh_name
    result["coord_level"] = coord_level
    result["matched_latitude"] = row.latitude
    result["matched_longitude"] = row.longitude
    result["match_method"] = method

    return result


def fill_admin1_result(result: Dict[str, str], row, method: str) -> Dict[str, str]:
    """Fill result with admin1-level match."""
    result["matched_country_code"] = row.country_code
    result["matched_country_name"] = row.country_zh_name
    result["matched_admin1_code"] = row.admin1_code
    result["matched_admin1_name"] = row.admin1_zh_name
    result["coord_level"] = "country-admin1"
    result["matched_latitude"] = row.latitude
    result["matched_longitude"] = row.longitude
    result["match_method"] = method

    return result


def fill_country_result(result: Dict[str, str], row) -> Dict[str, str]:
    """Fill result with country-level match."""
    result["matched_country_code"] = row.country_code
    result["matched_country_name"] = row.country_zh_name
    result["coord_level"] = "country"
    result["matched_latitude"] = row.latitude
    result["matched_longitude"] = row.longitude
    result["match_method"] = "country"

    return result


def match_one_normalized_row(row, indexes: Dict[str, Any]) -> Dict[str, str]:
    """Match one normalized address row to GeoNames coordinates."""
    result = empty_coord_result()

    country_value = str(row.country_key).strip()
    admin1_key = str(row.admin1_key).strip()
    admin2_key = str(row.admin2_key).strip()
    admin3_key = str(row.admin3_key).strip()

    country_code_index = indexes["country_code_index"]
    country_by_code = indexes["country_by_code"]
    admin1_by_code = indexes["admin1_by_code"]
    city_by_code = indexes["city_by_code"]

    candidate_country_codes = country_code_index.get(country_value, [])

    if not candidate_country_codes:
        return result

    candidate_country = []
    candidate_admin1 = []
    candidate_city = []

    for country_code in candidate_country_codes:
        candidate_country.extend(country_by_code.get(country_code, []))
        candidate_admin1.extend(admin1_by_code.get(country_code, []))
        candidate_city.extend(city_by_code.get(country_code, []))

    city_match_specs = [
        (admin1_key, admin2_key, "admin1-admin1_city-admin2"),
        (admin1_key, admin3_key, "admin1-admin1_city-admin3"),
        (admin2_key, admin3_key, "admin1-admin2_city-admin3"),
    ]

    for admin_key, city_key, method in city_match_specs:
        if not admin_key or not city_key:
            continue

        candidates = []

        for candidate in candidate_city:
            if country_value not in candidate.country_alias_list:
                continue

            admin_pos = match_alias_position(candidate.admin1_alias_key_list, admin_key)
            city_pos = match_alias_position(candidate.city_alias_key_list, city_key)

            if admin_pos is None or city_pos is None:
                continue

            candidates.append((admin_pos, city_pos, candidate.row_order, candidate))

        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            return fill_city_result(
                result,
                candidates[0][3],
                "country-admin1-admin2",
                method,
            )

    admin1_specs = [
        (admin1_key, "admin1-admin1"),
        (admin2_key, "admin1-admin2"),
        (admin3_key, "admin1-admin3"),
    ]

    for admin_key, method in admin1_specs:
        if not admin_key:
            continue

        candidates = []

        for candidate in candidate_admin1:
            if country_value not in candidate.country_alias_list:
                continue

            pos = match_alias_position(candidate.admin1_alias_key_list, admin_key)

            if pos is None:
                continue

            candidates.append((pos, candidate.row_order, candidate))

        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            return fill_admin1_result(result, candidates[0][2], method)

    constrained_admin1_codes = set()

    for probe in [admin1_key, admin2_key, admin3_key]:
        if not probe:
            continue

        admin1_candidates = []

        for candidate in candidate_admin1:
            if country_value not in candidate.country_alias_list:
                continue

            pos = match_alias_position(candidate.admin1_alias_key_list, probe)

            if pos is None:
                continue

            admin1_candidates.append((pos, candidate.row_order, candidate))

        if admin1_candidates:
            admin1_candidates.sort(key=lambda x: (x[0], x[1]))
            constrained_admin1_codes.add(admin1_candidates[0][2].admin1_code)

    def find_city(city_key: str, method: str):
        if not city_key:
            return None

        candidates = []

        for candidate in candidate_city:
            if country_value not in candidate.country_alias_list:
                continue

            if constrained_admin1_codes and candidate.admin1_code not in constrained_admin1_codes:
                continue

            pos = match_alias_position(candidate.city_alias_key_list, city_key)

            if pos is None:
                continue

            candidates.append((pos, candidate.row_order, candidate))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2], method

    for city_key, method in [
        (admin1_key, "city-admin1"),
        (admin2_key, "city-admin2"),
        (admin3_key, "city-admin3"),
    ]:
        hit = find_city(city_key, method)

        if hit is not None:
            city_row, match_method = hit
            return fill_city_result(
                result,
                city_row,
                "country-admin2",
                match_method,
            )

    candidates = []

    for candidate in candidate_country:
        if country_value in candidate.country_alias_list:
            pos = candidate.country_alias_list.index(country_value)
            candidates.append((pos, candidate.row_order, candidate))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        return fill_country_result(result, candidates[0][2])

    return result


def attach_coordinates(normalized_df: pd.DataFrame, indexes: Dict[str, Any]) -> pd.DataFrame:
    """Attach GeoNames coordinates to normalized address records."""
    df = normalized_df.copy()

    df["country_key"] = df["country_norm_zh"].astype(str).str.strip()
    df["admin1_key"] = df["admin1_norm_zh"].apply(strip_admin_suffix)
    df["admin2_key"] = df["admin2_norm_zh"].apply(strip_admin_suffix)
    df["admin3_key"] = df["admin3_norm_zh"].apply(strip_admin_suffix)

    results = [
        match_one_normalized_row(row, indexes)
        for row in df.itertuples(index=False)
    ]

    match_df = pd.DataFrame(results)

    out = pd.concat(
        [
            df.reset_index(drop=True),
            match_df.reset_index(drop=True),
        ],
        axis=1,
    )

    out["latitude"] = out["matched_latitude"]
    out["longitude"] = out["matched_longitude"]

    helper_cols = [
        "country_key",
        "admin1_key",
        "admin2_key",
        "admin3_key",
    ]

    out = out.drop(columns=[col for col in helper_cols if col in out.columns])

    return out


# ============================================================
# 6. Main pipeline
# ============================================================

def main() -> None:
    """Run LLM normalization and GeoNames coordinate matching."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    address_df = pd.read_csv(
        ADDRESS_CSV,
        dtype=str,
        keep_default_na=False,
        encoding=ENCODING,
    )

    if ADDRESS_COLUMN not in address_df.columns:
        if address_df.shape[1] == 1:
            address_df.columns = [ADDRESS_COLUMN]
        else:
            raise ValueError(f"Input CSV must contain column: {ADDRESS_COLUMN}")

    address_df = address_df[address_df[ADDRESS_COLUMN] != ""].copy()
    address_df = address_df.drop_duplicates(subset=[ADDRESS_COLUMN], keep="first")

    llm_results = []

    for address in address_df[ADDRESS_COLUMN]:
        parsed = call_llm_parse_address(address)
        llm_results.append(parsed)

        if SLEEP_BETWEEN_CALLS > 0:
            time.sleep(SLEEP_BETWEEN_CALLS)

    normalized_df = pd.concat(
        [
            address_df.reset_index(drop=True),
            pd.DataFrame(llm_results).reset_index(drop=True),
        ],
        axis=1,
    )

    indexes = load_lexicon()
    final_df = attach_coordinates(normalized_df, indexes)

    FINAL_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(
        FINAL_OUTPUT_CSV,
        index=False,
        encoding=ENCODING,
    )


if __name__ == "__main__":
    main()



