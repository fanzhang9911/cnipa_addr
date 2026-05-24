#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：cnipa_address 
@File    ：2_imputation_by_agapi.py
@IDE     ：PyCharm 
@Author  ：Fan Zhang
@Note    : imputation using the Amap Geocoding API.
'''


import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests


# ============================================================
# 0. File paths and settings
# ============================================================

INPUT_CSV = Path("path/to/unmatched_applicants.csv")
PREVIOUS_RESULT_CSV = Path("path/to/previous_geocoding_result.csv")
OUTPUT_CSV = Path("path/to/imputation_by_agapi.csv")

ENCODING = "utf-8"

ADDRESS_COLUMN = "applicant"
MAX_WORKERS = 1
REQUEST_TIMEOUT = 3

AMAP_API_KEY = os.getenv("AMAP_API_KEY")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/54.0.2840.99 Safari/537.36"
    )
}


# ============================================================
# 1. Applicant name cleaning
# ============================================================

def clean_query_text(value) -> str:
    """Clean applicant name before geocoding."""
    if value is None:
        return ""

    text = str(value).strip()
    text = text.replace("|", "")
    text = text.replace("#", "号")
    text = text.split("、")[0].strip()

    return text


# ============================================================
# 2. XML parsing helpers
# ============================================================

def get_xml_text(root: ET.Element, tag: str) -> Optional[str]:
    """Return the first text value of an XML tag."""
    elem = root.find(f".//{tag}")

    if elem is None or elem.text is None:
        return None

    text = elem.text.strip()
    return text if text else None


def empty_geocode_result() -> Dict[str, Optional[str]]:
    """Return an empty geocoding result."""
    return {
        "country": None,
        "province": None,
        "city": None,
        "citycode": None,
        "district": None,
        "street": None,
        "number": None,
        "adcode": None,
        "longitude": None,
        "latitude": None,
        "level": None,
        "formatted_address": None,
    }


# ============================================================
# 3. Amap geocoding
# ============================================================

def geocode_amap(address: str) -> Dict[str, Optional[str]]:
    """Geocode one address using the Amap Geocoding API."""
    if not address:
        return empty_geocode_result()

    params = {
        "address": address,
        "output": "XML",
        "key": AMAP_API_KEY,
    }

    try:
        response = requests.get(
            "https://restapi.amap.com/v3/geocode/geo",
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        root = ET.fromstring(response.text)

        location = get_xml_text(root, "location")
        longitude = None
        latitude = None

        if location and "," in location:
            longitude, latitude = location.split(",", 1)

        return {
            "country": get_xml_text(root, "country"),
            "province": get_xml_text(root, "province"),
            "city": get_xml_text(root, "city"),
            "citycode": get_xml_text(root, "citycode"),
            "district": get_xml_text(root, "district"),
            "street": get_xml_text(root, "street"),
            "number": get_xml_text(root, "number"),
            "adcode": get_xml_text(root, "adcode"),
            "longitude": longitude,
            "latitude": latitude,
            "level": get_xml_text(root, "level"),
            "formatted_address": get_xml_text(root, "formatted_address"),
        }

    except Exception:
        return empty_geocode_result()


# ============================================================
# 4. Load and filter data
# ============================================================

def load_applicants() -> pd.DataFrame:
    """Load unmatched applicants and remove already processed applicants if available."""
    df = pd.read_csv(
        INPUT_CSV,
        encoding=ENCODING,
        dtype=str,
        keep_default_na=False,
    )

    if ADDRESS_COLUMN not in df.columns:
        raise ValueError(f"Required column not found: {ADDRESS_COLUMN}")

    if PREVIOUS_RESULT_CSV.exists():
        previous_df = pd.read_csv(
            PREVIOUS_RESULT_CSV,
            encoding=ENCODING,
            dtype=str,
            keep_default_na=False,
        )

        if ADDRESS_COLUMN in previous_df.columns:
            df = df[~df[ADDRESS_COLUMN].isin(previous_df[ADDRESS_COLUMN])]

    df[ADDRESS_COLUMN] = df[ADDRESS_COLUMN].apply(clean_query_text)
    df = df[df[ADDRESS_COLUMN] != ""]
    df = df.drop_duplicates(subset=[ADDRESS_COLUMN], keep="first")

    return df


# ============================================================
# 5. Main pipeline
# ============================================================

def main() -> None:
    """Run geocoding and save the result."""
    if not AMAP_API_KEY:
        raise ValueError(
            "Amap API key not found. Please set the AMAP_API_KEY environment variable."
        )

    start_time = time.time()

    df = load_applicants()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(geocode_amap, df[ADDRESS_COLUMN]))

    result_df = pd.DataFrame(results)

    output_df = pd.concat(
        [
            df.reset_index(drop=True),
            result_df.reset_index(drop=True),
        ],
        axis=1,
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding=ENCODING,
    )

    elapsed = time.time() - start_time
    print(f"Saved geocoding results to {OUTPUT_CSV}")
    print(f"Total time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()

