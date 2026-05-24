#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：cnipa_address 
@File    ：2_imputation_by_registration_addr.py
@IDE     ：PyCharm 
@Author  ：Fan Zhang
@Note    : imputation using registration address.
'''


import re
from pathlib import Path

import pandas as pd


# ============================================================
# 0. File paths
# ============================================================

APPLICANT_CSV = Path("path/to/applicants_need_registeration_address.csv")
REG_ADDRESS_CSV = Path("path/to/necips.csv")

FINAL_MATCHED_CSV = Path("path/to/app_matched_reg_addr_filtered.csv")

ENCODING = "utf-8"


# ============================================================
# 1. Name cleaning patterns
# ============================================================

COUNTRY_PATTERN = re.compile(
    r"菲律宾|西班牙|维尔京群岛|汶莱|法国|澳洲|意大利|塞舌尔|安圭拉|埃及|"
    r"马来西亚|加拿大|印尼|南非|南韩|巴拿马|文莱|毛里求斯|泰国|萨摩亚|"
    r"澳大利亚|澳门|韩国|香港|荷兰|美国|英国|英属|日本|德国|新加坡|台湾|印度"
)

COMPANY_TYPE_PATTERN = re.compile(
    r"集团股份有限公司|有限责任公司|股份有限公司|集团有限公司|有限公司|"
    r"总公司|分公司|总厂|工厂|分厂|公司|集团|厂|"
    r"研究所|研究院|株式会社|省|市|区|地区"
)

LEADING_NUMBER_PATTERN = re.compile(r"^\s*[0-9]+\s*")
SYMBOL_PATTERN = re.compile(r"[\(\)\?\-~_]|（|）|\.|、")


# ============================================================
# 2. Utility functions
# ============================================================

def clean_company_name(value) -> str:
    """Clean applicant or company name for rule-based matching."""
    if not isinstance(value, str):
        return ""

    name = COUNTRY_PATTERN.sub("", value)
    name = COMPANY_TYPE_PATTERN.sub("", name)
    name = LEADING_NUMBER_PATTERN.sub("", name)
    name = SYMBOL_PATTERN.sub("", name)

    return name.strip()


def similarity_ratio(left, right) -> float:
    """
    Compute string similarity.

    Uses python-Levenshtein if available; otherwise falls back to difflib.
    """
    left = "" if left is None else str(left)
    right = "" if right is None else str(right)

    try:
        from Levenshtein import ratio

        return ratio(left, right)
    except ImportError:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, left, right).ratio()


def get_best_match(group: pd.DataFrame, src_col: str, tgt_col: str) -> pd.Series:
    """Keep the row with the highest name similarity within a group."""
    scores = group.apply(
        lambda row: similarity_ratio(row[src_col], row[tgt_col]),
        axis=1,
    )

    return group.loc[scores.idxmax()]


def anti_join_by_pair(
    df: pd.DataFrame,
    used_pairs: pd.DataFrame,
    left_col: str = "applicant",
    right_col: str = "company_name",
) -> pd.DataFrame:
    """Remove rows whose applicant-company pair appears in used_pairs."""
    if used_pairs.empty:
        return df.copy()

    used_key = (
        used_pairs[left_col].astype(str)
        + "|||"
        + used_pairs[right_col].astype(str)
    )

    df_key = (
        df[left_col].astype(str)
        + "|||"
        + df[right_col].astype(str)
    )

    return df.loc[~df_key.isin(set(used_key))].copy()


# ============================================================
# 3. Load and clean data
# ============================================================

def load_applicants() -> pd.DataFrame:
    """Load applicants and add cleaned applicant names."""
    df = pd.read_csv(
        APPLICANT_CSV,
        encoding=ENCODING,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )

    if "applicant" not in df.columns:
        raise ValueError("Required column not found in applicant file: applicant")

    df["applicant_cleaned"] = df["applicant"].apply(clean_company_name)

    return df


def load_registered_companies() -> pd.DataFrame:
    """Load registered companies and add cleaned company names."""
    df = pd.read_csv(
        REG_ADDRESS_CSV,
        encoding=ENCODING,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )

    required_columns = ["ent_name", "reg_addr"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column not found in registered company file: {col}")

    df = df.dropna(subset=["ent_name", "reg_addr"])
    df = df.drop_duplicates(subset=["ent_name"], keep="first").copy()

    df["ent_name_cleaned"] = df["ent_name"].apply(clean_company_name)

    return df


# ============================================================
# 4. Match applicants with registered companies
# ============================================================

def build_raw_matches(
    applicants_df: pd.DataFrame,
    reg_df: pd.DataFrame,
) -> pd.DataFrame:
    """Match applicants and registered companies by cleaned names."""
    matched = applicants_df.merge(
        reg_df,
        left_on="applicant_cleaned",
        right_on="ent_name_cleaned",
        how="inner",
    )

    matched = matched.rename(
        columns={
            "ent_name": "company_name",
            "ent_name_cleaned": "company_name_cleaned",
        }
    )

    return matched


# ============================================================
# 5. Resolve duplicated matches
# ============================================================

def filter_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter raw applicant-company matches.

    Matching types:
    1. One-to-one cleaned-name matches are kept directly.
    2. One applicant to multiple companies:
       keep the company with the highest raw-name similarity.
    3. Multiple applicants to one company:
       keep the applicant with the highest raw-name similarity.
    4. Many-to-many:
       keep the best match within each cleaned applicant name.
    """
    applicant_counts = df["applicant_cleaned"].value_counts()
    company_counts = df["company_name_cleaned"].value_counts()

    unique_applicants = applicant_counts[applicant_counts == 1].index
    unique_companies = company_counts[company_counts == 1].index

    one_to_one_df = df[
        df["applicant_cleaned"].isin(unique_applicants)
        & df["company_name_cleaned"].isin(unique_companies)
    ].copy()

    one_to_many_df = df.groupby("applicant_cleaned", group_keys=False).filter(
        lambda g: g["applicant"].nunique() == 1
        and g["company_name"].nunique() > 1
    )

    many_to_one_df = df.groupby("company_name_cleaned", group_keys=False).filter(
        lambda g: g["company_name"].nunique() == 1
        and g["applicant"].nunique() > 1
    )

    best_one_to_many_df = (
        one_to_many_df
        .groupby("applicant_cleaned", group_keys=False)
        .apply(lambda g: get_best_match(g, "applicant", "company_name"))
        .reset_index(drop=True)
        if not one_to_many_df.empty
        else one_to_many_df.copy()
    )

    best_many_to_one_df = (
        many_to_one_df
        .groupby("company_name_cleaned", group_keys=False)
        .apply(lambda g: get_best_match(g, "applicant", "company_name"))
        .reset_index(drop=True)
        if not many_to_one_df.empty
        else many_to_one_df.copy()
    )

    used_pairs = pd.concat(
        [
            one_to_one_df[["applicant", "company_name"]],
            one_to_many_df[["applicant", "company_name"]],
            many_to_one_df[["applicant", "company_name"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    many_to_many_df = anti_join_by_pair(
        df=df,
        used_pairs=used_pairs,
        left_col="applicant",
        right_col="company_name",
    )

    best_many_to_many_df = (
        many_to_many_df
        .groupby("applicant_cleaned", group_keys=False)
        .apply(lambda g: get_best_match(g, "applicant", "company_name"))
        .reset_index(drop=True)
        if not many_to_many_df.empty
        else many_to_many_df.copy()
    )

    final_df = pd.concat(
        [
            one_to_one_df,
            best_one_to_many_df,
            best_many_to_one_df,
            best_many_to_many_df,
        ],
        ignore_index=True,
    )

    final_df = final_df.drop_duplicates(
        subset=["applicant", "company_name"],
        keep="first",
    )

    return final_df


# ============================================================
# 6. Main pipeline
# ============================================================

def main() -> None:
    applicants_df = load_applicants()
    reg_df = load_registered_companies()

    matched_df = build_raw_matches(
        applicants_df=applicants_df,
        reg_df=reg_df,
    )

    final_df = filter_matches(matched_df)

    FINAL_MATCHED_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(
        FINAL_MATCHED_CSV,
        index=False,
        encoding=ENCODING,
    )


if __name__ == "__main__":
    main()