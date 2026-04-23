#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：cnipa_address
@File    ：amap_geocoding.py
@IDE     ：PyCharm
@Machine : 12700KF + 4060Ti16G
@Author  ：Fan ZHANG
@Date    ：2026/04/20
@Note    : Geocoding with Amap Geocoding API (https://lbs.amap.com/api/webservice/guide/api/georegeo)
'''

import argparse
import csv
import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List
from urllib.parse import quote

import requests


OUTPUT_FIELDS = [
    "country",
    "province",
    "city",
    "citycode",
    "district",
    "street",
    "number",
    "adcode",
    "longitude",
    "latitude",
    "match_level",
    "formatted_address",
]


@dataclass
class GeocodeResult:
    country: str = ""
    province: str = ""
    city: str = ""
    citycode: str = ""
    district: str = ""
    street: str = ""
    number: str = ""
    adcode: str = ""
    longitude: str = ""
    latitude: str = ""
    match_level: str = ""
    formatted_address: str = ""


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch geocode addresses in a CSV file using the Amap Geocoding API."
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to the output CSV file. Default: input filename + _result.csv",
    )
    parser.add_argument(
        "--address-column",
        default="address",
        help="Name of the address column in the input CSV. Default: address",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("AMAP_API_KEY"),
        help="Amap API key. If omitted, reads from AMAP_API_KEY environment variable.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Request timeout in seconds. Default: 3.0",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Geocode each unique cleaned address only once and reuse cached results.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV encoding. Default: utf-8",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def clean_address(address: object) -> str:
    """
    Normalize an address string before geocoding.
    """
    if address is None:
        return ""

    cleaned = str(address).strip()
    cleaned = cleaned.replace("#", "号")
    cleaned = cleaned.split("、")[0].strip()
    return cleaned


def safe_find_text(root: ET.Element, tag: str) -> str:
    """Return the stripped text of the first matching XML tag, or an empty string."""
    element = root.find(f".//{tag}")
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def parse_geocode_xml(xml_text: str) -> GeocodeResult:
    """Parse XML returned by the Amap geocoding API."""
    root = ET.fromstring(xml_text)

    location = safe_find_text(root, "location")
    longitude = ""
    latitude = ""
    if location and "," in location:
        longitude, latitude = location.split(",", 1)

    return GeocodeResult(
        country=safe_find_text(root, "country"),
        province=safe_find_text(root, "province"),
        city=safe_find_text(root, "city"),
        citycode=safe_find_text(root, "citycode"),
        district=safe_find_text(root, "district"),
        street=safe_find_text(root, "street"),
        number=safe_find_text(root, "number"),
        adcode=safe_find_text(root, "adcode"),
        longitude=longitude,
        latitude=latitude,
        match_level=safe_find_text(root, "level"),
        formatted_address=safe_find_text(root, "formatted_address"),
    )


def geocode_address(
    address: str,
    api_key: str,
    timeout: float,
    session: requests.Session,
) -> GeocodeResult:
    """Call the Amap geocoding API for a single address."""
    if not address:
        return GeocodeResult()

    encoded_address = quote(address)
    url = (
        "https://restapi.amap.com/v3/geocode/geo"
        f"?address={encoded_address}&output=XML&key={api_key}"
    )

    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return parse_geocode_xml(response.text)
    except requests.RequestException as exc:
        logging.warning("Request failed for address %r: %s", address, exc)
        return GeocodeResult()
    except ET.ParseError as exc:
        logging.warning("XML parsing failed for address %r: %s", address, exc)
        return GeocodeResult()
    except Exception as exc:
        logging.exception("Unexpected error for address %r: %s", address, exc)
        return GeocodeResult()


def derive_output_path(input_csv: str) -> str:
    """Derive a default output path from the input CSV path."""
    if input_csv.lower().endswith(".csv"):
        return input_csv[:-4] + "_result.csv"
    return input_csv + "_result.csv"


def read_csv_rows(input_csv: str, encoding: str) -> List[Dict[str, str]]:
    """Read all rows from a CSV file into a list of dictionaries."""
    with open(input_csv, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv_rows(
    output_csv: str,
    rows: Iterable[Dict[str, str]],
    fieldnames: List[str],
    encoding: str,
) -> None:
    """Write rows to a CSV file using the provided fieldnames."""
    with open(output_csv, "w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_rows(
    rows: List[Dict[str, str]],
    address_column: str,
    api_key: str,
    timeout: float,
    deduplicate: bool,
) -> List[Dict[str, str]]:
    """
    Process CSV rows and append geocoding results.

    If deduplicate=True, identical cleaned addresses are geocoded once and cached.
    """
    if not rows:
        return rows

    for row in rows:
        raw_address = row.get(address_column, "")
        row[address_column] = clean_address(raw_address)

    session = requests.Session()
    cache: Dict[str, Dict[str, str]] = {}

    processed_rows: List[Dict[str, str]] = []

    for idx, row in enumerate(rows, start=1):
        address = row.get(address_column, "")

        if deduplicate and address in cache:
            result_dict = cache[address]
        else:
            result = geocode_address(
                address=address,
                api_key=api_key,
                timeout=timeout,
                session=session,
            )
            result_dict = asdict(result)

            if deduplicate:
                cache[address] = result_dict

        merged = dict(row)
        merged.update(result_dict)
        processed_rows.append(merged)

        if idx % 100 == 0:
            logging.info("Processed %d rows", idx)

    return processed_rows


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not args.api_key:
        raise ValueError(
            "Amap API key is required. Use --api-key or set AMAP_API_KEY."
        )

    start_time = time.time()

    logging.info("Reading input CSV: %s", args.input_csv)
    rows = read_csv_rows(args.input_csv, args.encoding)

    if not rows:
        raise ValueError(f"No data rows found in input file: {args.input_csv}")

    first_row = rows[0]
    if args.address_column not in first_row:
        raise ValueError(
            f"Address column {args.address_column!r} not found in input CSV."
        )

    processed_rows = process_rows(
        rows=rows,
        address_column=args.address_column,
        api_key=args.api_key,
        timeout=args.timeout,
        deduplicate=args.deduplicate,
    )

    output_csv = args.output_csv or derive_output_path(args.input_csv)

    original_fields = list(rows[0].keys())
    output_fields = original_fields + [f for f in OUTPUT_FIELDS if f not in original_fields]

    logging.info("Writing output CSV: %s", output_csv)
    write_csv_rows(
        output_csv=output_csv,
        rows=processed_rows,
        fieldnames=output_fields,
        encoding=args.encoding,
    )

    elapsed = time.time() - start_time
    logging.info("Finished in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()

