#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：cnipa_address 
@File    ：1_countrycode_imputation.py
@IDE     ：PyCharm 
@Author  : Fan Zhang
@Note    : impute and correct country/region codes
'''


import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ============================================================
# 0. File paths
# ============================================================

ADDRESS_CSV = Path("path/to/address_cleaned.csv")
BULK_COUNTRY_TXT = Path("path/to/bulk_country_code.txt")
BULK_PUBLICATION_TXT = Path("path/to/bulk_publication_date.txt")

FINAL_COUNTRYCODE_CSV = Path("path/to/countrycode_corrected.csv")


# ============================================================
# 1. Text normalization
# ============================================================

def build_t2s_converter():
    """Use OpenCC for Traditional-to-Simplified Chinese conversion if available."""
    try:
        from opencc import OpenCC

        cc = OpenCC("t2s")
        return lambda x: cc.convert(x)
    except Exception:
        return lambda x: x


t2s = build_t2s_converter()

_WS_RE = re.compile(r"\s+")
_HEAD_NON_ZH_RE = re.compile(r"^[^\u4e00-\u9fff]+")
_DOT_SUFFIX_RE = re.compile(r"\..*$")


def norm_text(value) -> str:
    """Normalize text before country/region matching."""
    if value is None:
        return ""

    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = t2s(text)
    text = text.strip()
    text = _WS_RE.sub(" ", text)
    text = _HEAD_NON_ZH_RE.sub("", text)

    return text


# ============================================================
# 2. Country/region alias dictionary
# ============================================================

COMMON_ZH_VARIANTS: Dict[str, List[str]] = {
    "AD": ["安道尔"],
    "AE": ["阿拉伯联合酋长国", "阿联酋"],
    "AF": ["阿富汗"],
    "AG": ["安提瓜和巴布达"],
    "AI": ["安圭拉"],
    "AL": ["阿尔巴尼亚"],
    "AM": ["亚美尼亚"],
    "AN": ["荷属安的列斯"],
    "AO": ["安哥拉"],
    "AQ": ["南极洲"],
    "AR": ["阿根廷"],
    "AS": ["美属萨摩亚"],
    "AT": ["奥地利"],
    "AU": ["澳大利亚", "澳洲"],
    "AW": ["阿鲁巴"],
    "AX": ["奥兰群岛"],
    "AZ": ["阿塞拜疆"],

    "BA": ["波斯尼亚和黑塞哥维那", "波黑"],
    "BB": ["巴巴多斯"],
    "BD": ["孟加拉国"],
    "BE": ["比利时"],
    "BF": ["布基纳法索"],
    "BG": ["保加利亚"],
    "BH": ["巴林"],
    "BI": ["布隆迪"],
    "BJ": ["贝宁"],
    "BL": ["圣巴泰勒米"],
    "BM": ["百慕大", "百幕大", "英属百慕大", "百慕达"],
    "BN": ["文莱"],
    "BO": ["玻利维亚"],
    "BQ": ["博内尔、圣尤斯特歇斯和萨巴", "荷兰加勒比"],
    "BR": ["巴西"],
    "BS": ["巴哈马"],
    "BT": ["不丹"],
    "BV": ["布韦岛"],
    "BW": ["博茨瓦纳"],
    "BY": ["白俄罗斯"],
    "BZ": ["伯利兹"],

    "CA": ["加拿大"],
    "CC": ["科科斯（基林）群岛", "科科斯群岛"],
    "CD": ["刚果民主共和国", "刚果"],
    "CF": ["中非共和国", "中非"],
    "CG": ["刚果共和国", "刚果（布）"],
    "CH": ["瑞士"],
    "CI": ["科特迪瓦", "象牙海岸"],
    "CK": ["库克群岛"],
    "CL": ["智利"],
    "CM": ["喀麦隆"],
    "CO": ["哥伦比亚"],
    "CR": ["哥斯达黎加"],
    "CU": ["古巴"],
    "CV": ["佛得角"],
    "CW": ["库拉索"],
    "CX": ["圣诞岛"],
    "CY": ["塞浦路斯"],
    "CZ": ["捷克"],

    "DE": ["德国", "联邦德国", "民主德国"],
    "DJ": ["吉布提"],
    "DK": ["丹麦"],
    "DM": ["多米尼克"],
    "DO": ["多米尼加共和国", "多米尼加"],
    "DZ": ["阿尔及利亚"],

    "EC": ["厄瓜多尔"],
    "EE": ["爱沙尼亚"],
    "EG": ["埃及"],
    "EH": ["西撒哈拉"],
    "ER": ["厄立特里亚"],
    "ES": ["西班牙"],
    "ET": ["埃塞俄比亚"],

    "FI": ["芬兰"],
    "FJ": ["斐济"],
    "FK": ["福克兰群岛", "马尔维纳斯群岛"],
    "FM": ["密克罗尼西亚联邦", "密克罗尼西亚"],
    "FO": ["法罗群岛"],
    "FR": ["法国"],

    "GA": ["加蓬"],
    "GB": ["英国"],
    "GD": ["格林纳达"],
    "GE": ["格鲁吉亚"],
    "GF": ["法属圭亚那"],
    "GG": ["根西"],
    "GH": ["加纳"],
    "GI": ["直布罗陀"],
    "GL": ["格陵兰"],
    "GM": ["冈比亚"],
    "GN": ["几内亚"],
    "GP": ["瓜德罗普"],
    "GQ": ["赤道几内亚"],
    "GR": ["希腊"],
    "GS": ["南乔治亚和南桑威奇群岛"],
    "GT": ["危地马拉"],
    "GU": ["关岛"],
    "GW": ["几内亚比绍"],
    "GY": ["圭亚那"],

    "HK": ["香港", "中国香港", "香港特别行政区"],
    "HM": ["赫德岛和麦克唐纳群岛"],
    "HN": ["洪都拉斯"],
    "HR": ["克罗地亚"],
    "HT": ["海地"],
    "HU": ["匈牙利", "匈亚利"],

    "ID": ["印度尼西亚", "印尼"],
    "IE": ["爱尔兰"],
    "IL": ["以色列"],
    "IM": ["马恩岛"],
    "IN": ["印度"],
    "IO": ["英属印度洋领地"],
    "IQ": ["伊拉克"],
    "IR": ["伊朗"],
    "IS": ["冰岛"],
    "IT": ["意大利"],

    "JE": ["泽西"],
    "JM": ["牙买加"],
    "JO": ["约旦"],
    "JP": ["日本"],

    "KE": ["肯尼亚"],
    "KG": ["吉尔吉斯斯坦"],
    "KH": ["柬埔寨"],
    "KI": ["基里巴斯"],
    "KM": ["科摩罗"],
    "KN": ["圣基茨和尼维斯"],
    "KP": ["朝鲜", "朝鲜民主主义人民共和国"],
    "KR": ["韩国", "大韩民国", "南韩", "南朝鲜"],
    "KW": ["科威特"],
    "KY": ["开曼群岛", "英属西印度群岛大开曼岛", "英属开曼群岛", "英属西印度群岛开曼群岛"],
    "KZ": ["哈萨克斯坦"],

    "LA": ["老挝"],
    "LB": ["黎巴嫩"],
    "LC": ["圣卢西亚"],
    "LI": ["列支敦士登"],
    "LK": ["斯里兰卡"],
    "LR": ["利比里亚"],
    "LS": ["莱索托"],
    "LT": ["立陶宛"],
    "LU": ["卢森堡"],
    "LV": ["拉脱维亚"],
    "LY": ["利比亚"],

    "MA": ["摩洛哥"],
    "MC": ["摩纳哥"],
    "MD": ["摩尔多瓦"],
    "ME": ["黑山"],
    "MF": ["法属圣马丁"],
    "MG": ["马达加斯加"],
    "MH": ["马绍尔群岛"],
    "MK": ["北马其顿"],
    "ML": ["马里"],
    "MM": ["缅甸"],
    "MN": ["蒙古"],
    "MO": ["澳门", "中国澳门"],
    "MP": ["北马里亚纳群岛"],
    "MQ": ["马提尼克"],
    "MR": ["毛里塔尼亚"],
    "MS": ["蒙特塞拉特"],
    "MT": ["马耳他"],
    "MU": ["毛里求斯"],
    "MV": ["马尔代夫"],
    "MW": ["马拉维"],
    "MX": ["墨西哥"],
    "MY": ["马来西亚"],
    "MZ": ["莫桑比克"],

    "NA": ["纳米比亚"],
    "NC": ["新喀里多尼亚"],
    "NE": ["尼日尔"],
    "NF": ["诺福克岛"],
    "NG": ["尼日利亚"],
    "NI": ["尼加拉瓜"],
    "NL": ["荷兰", "尼德兰"],
    "NO": ["挪威"],
    "NP": ["尼泊尔"],
    "NR": ["瑙鲁"],
    "NU": ["纽埃"],
    "NZ": ["新西兰"],

    "OM": ["阿曼"],

    "PA": ["巴拿马"],
    "PE": ["秘鲁"],
    "PF": ["法属波利尼西亚"],
    "PG": ["巴布亚新几内亚"],
    "PH": ["菲律宾"],
    "PK": ["巴基斯坦"],
    "PL": ["波兰"],
    "PM": ["圣皮埃尔和密克隆"],
    "PN": ["皮特凯恩群岛"],
    "PR": ["波多黎各", "美属波多黎各"],
    "PS": ["巴勒斯坦"],
    "PT": ["葡萄牙"],
    "PW": ["帕劳", "帛琉"],
    "PY": ["巴拉圭"],

    "QA": ["卡塔尔"],

    "RE": ["留尼汪"],
    "RO": ["罗马尼亚"],
    "RS": ["塞尔维亚"],
    "RU": ["俄罗斯", "俄国", "苏联"],
    "RW": ["卢旺达"],

    "SA": ["沙特阿拉伯", "沙特"],
    "SB": ["所罗门群岛"],
    "SC": ["塞舌尔"],
    "SD": ["苏丹"],
    "SE": ["瑞典"],
    "SG": ["新加坡", "狮城"],
    "SH": ["圣赫勒拿、阿森松和特里斯坦-达库尼亚", "圣赫勒拿"],
    "SI": ["斯洛文尼亚"],
    "SJ": ["斯瓦尔巴和扬马延"],
    "SK": ["斯洛伐克"],
    "SL": ["塞拉利昂", "狮子山"],
    "SM": ["圣马力诺"],
    "SN": ["塞内加尔"],
    "SO": ["索马里"],
    "SR": ["苏里南"],
    "SS": ["南苏丹"],
    "ST": ["圣多美和普林西比"],
    "SV": ["萨尔瓦多"],
    "SX": ["荷属圣马丁"],
    "SY": ["叙利亚"],
    "SZ": ["斯威士兰", "埃斯瓦蒂尼"],

    "TC": ["特克斯和凯科斯群岛"],
    "TD": ["乍得"],
    "TF": ["法属南部和南极领地", "法属南部领地"],
    "TG": ["多哥"],
    "TH": ["泰国"],
    "TJ": ["塔吉克斯坦"],
    "TK": ["托克劳"],
    "TL": ["东帝汶", "帝汶-莱斯特"],
    "TM": ["土库曼斯坦"],
    "TN": ["突尼斯"],
    "TO": ["汤加"],
    "TR": ["土耳其", "土耳其共和国"],
    "TT": ["特立尼达和多巴哥"],
    "TV": ["图瓦卢"],
    "TW": ["台湾", "中国台湾", "中华民国", "台北", "中国台北", "中华台北"],
    "TZ": ["坦桑尼亚"],

    "UA": ["乌克兰"],
    "UG": ["乌干达"],
    "UM": ["美国本土外小岛屿"],
    "US": ["美国", "美利坚合众国", "美利坚"],
    "UY": ["乌拉圭"],
    "UZ": ["乌兹别克斯坦"],

    "VA": ["梵蒂冈"],
    "VC": ["圣文森特和格林纳丁斯"],
    "VE": ["委内瑞拉"],
    "VG": ["英属维尔京群岛", "英属维京群岛", "维尔京群岛"],
    "VI": ["美属维尔京群岛", "美属维京群岛"],
    "VN": ["越南"],
    "VU": ["瓦努阿图"],

    "WF": ["瓦利斯和富图纳"],
    "WS": ["萨摩亚"],

    "YE": ["也门"],
    "YT": ["马约特"],

    "ZA": ["南非"],
    "ZM": ["赞比亚"],
    "ZW": ["津巴布韦"],

    "X": ["不公告专利权人地址", "申请人要求不公布地址", "不公告申请人地址"],
}


# ============================================================
# 3. Longest-prefix country/region matching
# ============================================================

@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    code: Optional[str] = None


class CountryPrefixTrie:
    """Trie for longest-prefix country/region matching."""

    def __init__(self):
        self.root = TrieNode()

    def add(self, alias: str, code: str) -> None:
        node = self.root

        for ch in alias:
            node = node.children.setdefault(ch, TrieNode())

        if node.code is None or code < node.code:
            node.code = code

    def match(self, text: str) -> Optional[str]:
        node = self.root
        best_code = None

        for ch in text:
            nxt = node.children.get(ch)

            if nxt is None:
                break

            node = nxt

            if node.code is not None:
                best_code = node.code

        return best_code


def build_country_trie(alias_dict: Dict[str, List[str]]) -> CountryPrefixTrie:
    """Build a trie from country/region aliases."""
    trie = CountryPrefixTrie()

    for code, aliases in alias_dict.items():
        for alias in aliases:
            alias = norm_text(alias)

            if alias:
                trie.add(alias, code)

    return trie


country_trie = build_country_trie(COMMON_ZH_VARIANTS)


@lru_cache(maxsize=1_000_000)
def detect_countrycode(address: str) -> str:
    """
    Detect country/region code from the address prefix.

    If no foreign or special-region prefix is matched, the address is treated as CN.
    """
    text = norm_text(address)
    code = country_trie.match(text)

    if code is None:
        return "CN"

    return code


# ============================================================
# 4. Helpers for CNIPA bulk-downloaded files
# ============================================================

def first_cn_and_drop_dot(value) -> str:
    """Extract the first CN application number and remove suffix after dot."""
    text = "" if value is None else str(value).strip()

    if not text:
        return ""

    for item in text.split(";"):
        item = item.strip()

        if item.startswith("CN"):
            return _DOT_SUFFIX_RE.sub("", item).strip()

    return ""


def count_cn_publication(value) -> int:
    """Count CN publication numbers in a semicolon-separated field."""
    text = "" if value is None else str(value).strip()

    if not text:
        return 0

    return sum(
        1
        for item in text.split(";")
        if item.strip().startswith("CN")
    )


# ============================================================
# 5. Detect country/region codes from address strings
# ============================================================

address_df = pd.read_csv(
    ADDRESS_CSV,
    dtype=str,
    encoding="utf-8",
    keep_default_na=False,
    na_filter=False,
)

address_df = address_df[["ida", "address"]]
address_df = address_df.drop_duplicates(subset=["ida"], keep="first")

detected_df = address_df[["ida"]].copy()
detected_df["countrycode_detected"] = address_df["address"].map(detect_countrycode)

# Preserve HK/MO/TW detected from address strings.
hkmotw_df = detected_df[
    detected_df["countrycode_detected"].isin(["HK", "MO", "TW"])
].copy()


# ============================================================
# 6. Extract reliable country/province records from bulk data
# ============================================================

bulk_country_df = pd.read_csv(
    BULK_COUNTRY_TXT,
    sep="|",
    dtype=str,
    encoding="utf-8",
    keep_default_na=False,
    na_filter=False,
)

bulk_pub_df = pd.read_csv(
    BULK_PUBLICATION_TXT,
    sep="|",
    dtype=str,
    encoding="utf-8",
    keep_default_na=False,
    na_filter=False,
)

bulk_country_df["ida"] = bulk_country_df["申请号"].map(first_cn_and_drop_dot)
bulk_pub_df["ida"] = bulk_pub_df["申请号"].map(first_cn_and_drop_dot)

bulk_country_df = bulk_country_df.drop_duplicates(subset=["ida"], keep="first")
bulk_pub_df = bulk_pub_df.drop_duplicates(subset=["ida"], keep="first")

bulk_country_df = bulk_country_df[
    (bulk_country_df["ida"] != "")
    & (bulk_country_df["申请人所在国（省）"] != "")
].copy()

bulk_pub_df["cn_pub_cnt"] = bulk_pub_df["公开（公告）号"].map(count_cn_publication)

bulk_pub_df = bulk_pub_df[
    (bulk_pub_df["ida"] != "")
    & (bulk_pub_df["cn_pub_cnt"] == 1)
].copy()

bulk_code_df = bulk_pub_df[["ida"]].merge(
    bulk_country_df[["ida", "申请人所在国（省）"]],
    on="ida",
    how="inner",
)

bulk_code_df = bulk_code_df.drop_duplicates(subset=["ida"], keep="first")
bulk_code_df = bulk_code_df.rename(
    columns={"申请人所在国（省）": "countrycode_detected"}
)

# Keep only IDs that appear in the address dataset.
bulk_code_df = bulk_code_df[bulk_code_df["ida"].isin(detected_df["ida"])]


# ============================================================
# 7. Correct detected country/region codes
# ============================================================

# Replace detected codes with reliable bulk-downloaded records.
final_df = detected_df[~detected_df["ida"].isin(bulk_code_df["ida"])]
final_df = pd.concat([final_df, bulk_code_df], ignore_index=True)

# Give priority to HK/MO/TW detected from address strings.
final_df = pd.concat([hkmotw_df, final_df], ignore_index=True)
final_df = final_df.drop_duplicates(subset=["ida"], keep="first")

FINAL_COUNTRYCODE_CSV.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(FINAL_COUNTRYCODE_CSV, index=False, encoding="utf-8")

