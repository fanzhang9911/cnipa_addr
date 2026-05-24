#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：cnipa_address 
@File    ：4_assembly.py
@IDE     ：PyCharm 
@Author  ：Fan Zhang
@Note    : assembly the final geocoded CNIPA applicant-address dataset.
'''


from pathlib import Path
import pandas as pd


# ============================================================
# 0. File paths
# ============================================================

IMPUTED_CSV = Path("path/to/addr_imputed.csv")

PATENT_ADDRESS_AMAP_CSV = Path("path/to/patent_address_amap_geocoded.csv")
REGISTERED_ADDRESS_AMAP_CSV = Path("path/to/registration_address_amap_geocoded.csv")
RETRIEVAL_AMAP_CSV = Path("path/to/retrieval_amap_geocoded.csv")
NONCN_LLM_GEONAMES_CSV = Path("path/to/noncn_llm_geonames_geocoded.csv")

REFINE_RESULT_CSV = Path("path/to/refine_result.csv")

FINAL_GEOCODED_CSV = Path("path/to/cnipa_geoc_2024jun.csv")

ENCODING = "utf-8"


# ============================================================
# 1. Common settings
# ============================================================

FINAL_KEY_COLS = ["ida", "applicant_seq"]

FINAL_GEOC_COLUMNS = [
    "address",
    "country/region_code",
    "source",
    "method",
    "admin_area_1",
    "admin_area_2",
    "city_code",
    "admin_area_3",
    "admin_area_4",
    "admin_area_5",
    "adcode",
    "longitude",
    "latitude",
    "match_level",
]

AMAP_RENAME_MAP = {
    "countrycode_detected": "country/region_code",
    "省份": "admin_area_1",
    "城市": "admin_area_2",
    "地址所在的区": "admin_area_3",
    "街道": "admin_area_4",
    "门牌": "admin_area_5",
    "城市编码": "city_code",
    "区域编码": "adcode",
    "经度": "longitude",
    "纬度": "latitude",
    "匹配级别": "match_level",
    "geoc_source": "method",
}

MATCH_LEVEL_MAP = {
    "nan": "admin0",
    "country": "admin0",
    "国家": "admin0",
    "admin0": "admin0",

    "admin1": "admin1",
    "省": "admin1",

    "admin2": "admin2",
    "市": "admin2",

    "开发区": "admin3",
    "区县": "admin3",
    "乡镇": "admin3",
    "村庄": "admin3",

    "道路": "admin4",
    "道路交叉路口": "admin4",
    "小巷": "admin4",

    "热点商圈": "admin5",
    "住宅区": "admin5",
    "兴趣点": "admin5",
    "公交地铁站点": "admin5",
    "门牌号": "admin5",
    "门址": "admin5",
}

METHOD_RENAME_MAP = {
    "amap_high_level": "amap",
    "regaddr_geoc": "amap",
    "reetrival_geoc": "amap",
    "failed": "amap",
    "re_result": "re_refine",
    "missing_re": "re_refine",
    "poi": "poi_refine",
    "regadd": "regaddr_refine",
}


# ============================================================
# 2. Generic helpers
# ============================================================

def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file as strings."""
    return pd.read_csv(
        path,
        encoding=ENCODING,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )


def require_columns(df: pd.DataFrame, columns: list[str], file_label: str) -> None:
    """Check required columns."""
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(f"{file_label} is missing required columns: {missing}")


def clean_address_for_amap_match(value) -> str:
    """Clean address text before matching with Amap results."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = text.replace("|", "")
    text = text.replace("#", "号")
    text = text.split("、")[0].strip()

    return text


def drop_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop columns if they exist."""
    return df.drop(columns=[col for col in columns if col in df.columns])


def standardize_match_level(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize geocoding match levels to admin0-admin5."""
    df = df.copy()

    if "match_level" not in df.columns:
        df["match_level"] = pd.NA

    df["match_level_std"] = df["match_level"].astype(str).map(MATCH_LEVEL_MAP)

    unknown_mask = df["match_level"] == "未知"

    df.loc[unknown_mask, "match_level_std"] = "admin0"
    df.loc[unknown_mask & df["admin_area_1"].notna(), "match_level_std"] = "admin1"
    df.loc[unknown_mask & df["admin_area_2"].notna(), "match_level_std"] = "admin2"
    df.loc[unknown_mask & df["admin_area_3"].notna(), "match_level_std"] = "admin3"
    df.loc[unknown_mask & df["admin_area_4"].notna(), "match_level_std"] = "admin4"
    df.loc[unknown_mask & df["admin_area_5"].notna(), "match_level_std"] = "admin5"

    df["match_level"] = df["match_level_std"]
    df = df.drop(columns=["match_level_std"])

    return df


def normalize_method(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize method names."""
    df = df.copy()

    if "method" in df.columns:
        df["method"] = df["method"].replace(METHOD_RENAME_MAP)

    return df


def remove_invalid_individual_addresses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove addresses for non-first individual applicants.

    Non-first individual applicants are not assigned applicant-level addresses.
    """
    df = df.copy()

    if "applicant_seq" not in df.columns or "is_individual" not in df.columns:
        return df

    cols_to_clear = [
        "address",
        "country/region_code",
        "source",
        "method",
        "admin_area_1",
        "admin_area_2",
        "city_code",
        "admin_area_3",
        "admin_area_4",
        "admin_area_5",
        "adcode",
        "longitude",
        "latitude",
        "match_level",
    ]

    cols_to_clear = [col for col in cols_to_clear if col in df.columns]

    mask = (df["applicant_seq"] != "1") & (df["is_individual"] == "1")
    df.loc[mask, cols_to_clear] = pd.NA

    return df


def clear_geocoding_if_missing_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Clear method and match level when longitude is missing."""
    df = df.copy()

    if "longitude" in df.columns:
        mask = df["longitude"].isna()

        if "match_level" in df.columns:
            df.loc[mask, "match_level"] = pd.NA

        if "method" in df.columns:
            df.loc[mask, "method"] = pd.NA

    return df


def clean_special_admin_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove placeholder administrative values."""
    df = df.copy()

    if "admin_area_2" in df.columns:
        df.loc[df["admin_area_2"].isin(["zhixiaxian", "bushequ"]), "admin_area_2"] = pd.NA

    if "admin_area_3" in df.columns:
        df.loc[df["admin_area_3"] == "bushequ", "admin_area_3"] = pd.NA

    return df


def raise_admin_level_if_district_exists(df: pd.DataFrame) -> pd.DataFrame:
    """Upgrade match level from admin2 to admin3 when admin_area_3 is available."""
    df = df.copy()

    required = ["admin_area_3", "match_level"]

    if all(col in df.columns for col in required):
        mask = df["admin_area_3"].notna() & (df["match_level"] == "admin2")
        df.loc[mask, "match_level"] = "admin3"

    return df


# ============================================================
# 3. Prepare base imputed records
# ============================================================

def load_imputed_records() -> pd.DataFrame:
    """Load and prepare imputed applicant-address records."""
    df = read_csv(IMPUTED_CSV)

    require_columns(
        df,
        ["ida", "applicant_seq", "source", "countrycode_detected"],
        str(IMPUTED_CSV),
    )

    df = df.dropna(subset=["source"])
    df = df[df["source"] != "undisclosed"].copy()

    return df


def split_imputed_records(imputed_df: pd.DataFrame):
    """Split imputed records by source and country/region code."""
    pataddr_cn = imputed_df[
        (imputed_df["countrycode_detected"] == "CN")
        & (imputed_df["source"] == "pataddr")
    ].copy()

    regaddr_cn = imputed_df[
        (imputed_df["countrycode_detected"] == "CN")
        & (imputed_df["source"] == "regaddr")
    ].copy()

    agapi_cn = imputed_df[
        (imputed_df["countrycode_detected"] == "CN")
        & (imputed_df["source"] == "agapi")
    ].copy()

    pataddr_noncn = imputed_df[
        (imputed_df["countrycode_detected"] != "CN")
        & (imputed_df["source"] == "pataddr")
    ].copy()

    return pataddr_cn, regaddr_cn, agapi_cn, pataddr_noncn


# ============================================================
# 4. Merge mainland-China patent-address Amap results
# ============================================================

def merge_pataddr_cn_with_amap(pataddr_cn_df: pd.DataFrame) -> pd.DataFrame:
    """Merge mainland-China patent addresses with Amap results."""
    amap_df = read_csv(PATENT_ADDRESS_AMAP_CSV)

    require_columns(amap_df, ["address"], str(PATENT_ADDRESS_AMAP_CSV))
    require_columns(pataddr_cn_df, ["address"], "pataddr_cn_df")

    amap_df = drop_existing_columns(
        amap_df,
        ["country", "applicant", "address_x", "address_y", "详细地址"],
    )

    amap_df = amap_df.drop_duplicates(subset=["address"], keep="first")
    amap_df = amap_df.rename(
        columns={
            "address": "address_cleaned",
            "source": "geoc_source",
        }
    )

    pataddr_cn_df = pataddr_cn_df.copy()
    pataddr_cn_df["address_cleaned"] = pataddr_cn_df["address"].apply(
        clean_address_for_amap_match
    )

    out = pataddr_cn_df.merge(
        amap_df,
        on="address_cleaned",
        how="left",
    )

    out = out.drop(columns=["address_cleaned"])
    out = out.rename(columns=AMAP_RENAME_MAP)

    return out


# ============================================================
# 5. Merge registered-address Amap results
# ============================================================

def merge_regaddr_cn_with_amap(regaddr_cn_df: pd.DataFrame) -> pd.DataFrame:
    """Merge registered-address records with Amap results by applicant."""
    regaddr_geoc_df = read_csv(REGISTERED_ADDRESS_AMAP_CSV)

    require_columns(regaddr_geoc_df, ["applicant"], str(REGISTERED_ADDRESS_AMAP_CSV))
    require_columns(regaddr_cn_df, ["applicant"], "regaddr_cn_df")

    regaddr_geoc_df = drop_existing_columns(
        regaddr_geoc_df,
        ["reg_addr", "国家", "详细地址"],
    )

    regaddr_geoc_df = regaddr_geoc_df.drop_duplicates(
        subset=["applicant"],
        keep="first",
    )

    out = regaddr_cn_df.merge(
        regaddr_geoc_df,
        on="applicant",
        how="left",
    )

    out["method"] = "regaddr_geoc"
    out = out.rename(columns=AMAP_RENAME_MAP)

    return out


# ============================================================
# 6. Merge applicant-name retrieval Amap results
# ============================================================

def merge_agapi_cn_with_amap(agapi_cn_df: pd.DataFrame) -> pd.DataFrame:
    """Merge applicant-name API retrieval records with Amap results."""
    retrieval_df = read_csv(RETRIEVAL_AMAP_CSV)

    require_columns(retrieval_df, ["applicant", "经度"], str(RETRIEVAL_AMAP_CSV))
    require_columns(agapi_cn_df, ["applicant"], "agapi_cn_df")

    retrieval_df = retrieval_df.dropna(subset=["经度"])
    retrieval_df = retrieval_df.drop_duplicates(subset=["applicant"], keep="first")
    retrieval_df = drop_existing_columns(retrieval_df, ["国家", "详细地址"])

    out = agapi_cn_df.merge(
        retrieval_df,
        on="applicant",
        how="left",
    )

    out["method"] = "retrieval_geoc"
    out = out.rename(columns=AMAP_RENAME_MAP)

    return out


# ============================================================
# 7. Merge non-mainland-China LLM + GeoNames results
# ============================================================

def merge_pataddr_noncn_with_llm(pataddr_noncn_df: pd.DataFrame) -> pd.DataFrame:
    """Merge non-mainland-China patent addresses with LLM + GeoNames results."""
    llm_df = read_csv(NONCN_LLM_GEONAMES_CSV)

    require_columns(llm_df, ["address"], str(NONCN_LLM_GEONAMES_CSV))
    require_columns(pataddr_noncn_df, ["address"], "pataddr_noncn_df")

    keep_cols = [
        "address",
        "admin1_norm_zh",
        "admin2_norm_zh",
        "admin3_norm_zh",
        "admin4_norm_zh",
        "admin5_norm_zh",
        "coord_level",
        "latitude",
        "longitude",
    ]

    keep_cols = [col for col in keep_cols if col in llm_df.columns]
    llm_df = llm_df[keep_cols].copy()

    llm_df = llm_df.rename(
        columns={
            "admin1_norm_zh": "admin_area_1",
            "admin2_norm_zh": "admin_area_2",
            "admin3_norm_zh": "admin_area_3",
            "admin4_norm_zh": "admin_area_4",
            "admin5_norm_zh": "admin_area_5",
        }
    )

    if "coord_level" in llm_df.columns:
        llm_df["match_level"] = (
            llm_df["coord_level"]
            .astype(str)
            .str.rsplit("-", n=1)
            .str[-1]
        )
        llm_df = llm_df.drop(columns=["coord_level"])
    else:
        llm_df["match_level"] = pd.NA

    out = pataddr_noncn_df.merge(
        llm_df,
        on="address",
        how="left",
    )

    out["method"] = "llm_geoc"
    out = out.rename(columns={"countrycode_detected": "country/region_code"})

    return out


# ============================================================
# 8. Build first-pass geocoded result
# ============================================================

def build_first_pass_geocoded() -> pd.DataFrame:
    """Build first-pass geocoded records from all geocoding sources."""
    imputed_df = load_imputed_records()

    (
        pataddr_cn_df,
        regaddr_cn_df,
        agapi_cn_df,
        pataddr_noncn_df,
    ) = split_imputed_records(imputed_df)

    pataddr_cn_df = merge_pataddr_cn_with_amap(pataddr_cn_df)
    regaddr_cn_df = merge_regaddr_cn_with_amap(regaddr_cn_df)
    agapi_cn_df = merge_agapi_cn_with_amap(agapi_cn_df)
    pataddr_noncn_df = merge_pataddr_noncn_with_llm(pataddr_noncn_df)

    geoc_df = pd.concat(
        [
            pataddr_cn_df,
            regaddr_cn_df,
            agapi_cn_df,
            pataddr_noncn_df,
        ],
        ignore_index=True,
    )

    geoc_df = standardize_match_level(geoc_df)
    geoc_df = geoc_df.drop_duplicates(subset=FINAL_KEY_COLS, keep="first")
    geoc_df = geoc_df.sort_values(FINAL_KEY_COLS)
    geoc_df = raise_admin_level_if_district_exists(geoc_df)

    return geoc_df


# ============================================================
# 9. Add refined and fallback records
# ============================================================

def build_final_result(first_pass_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build final result.

    Priority order:
    1. refined records
    2. first-pass geocoded records
    3. original imputed records
    """
    refine_df = read_csv(REFINE_RESULT_CSV)

    imputed_df = read_csv(IMPUTED_CSV)
    imputed_df = imputed_df.rename(
        columns={"countrycode_detected": "country/region_code"}
    )

    result = pd.concat(
        [
            refine_df,
            first_pass_df,
            imputed_df,
        ],
        ignore_index=True,
    )

    result = result.drop_duplicates(subset=FINAL_KEY_COLS, keep="first")

    result = clean_special_admin_values(result)
    result = standardize_match_level(result)
    result = normalize_method(result)
    result = remove_invalid_individual_addresses(result)
    result = clear_geocoding_if_missing_coordinates(result)
    result = raise_admin_level_if_district_exists(result)

    result = result.sort_values(FINAL_KEY_COLS)

    return result


# ============================================================
# 10. Main pipeline
# ============================================================

def main() -> None:
    """Run the full geocoding integration pipeline."""
    first_pass_df = build_first_pass_geocoded()
    final_df = build_final_result(first_pass_df)

    FINAL_GEOCODED_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(
        FINAL_GEOCODED_CSV,
        index=False,
        encoding=ENCODING,
    )


if __name__ == "__main__":
    main()