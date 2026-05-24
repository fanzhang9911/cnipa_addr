#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：cnipa_address 
@File    ：2_imputation_within_cnipa.py
@IDE     ：PyCharm 
@Author  ：Fan Zhang
@Note    : imputation within cnipa
'''


import csv
import os
from collections import defaultdict
from datetime import datetime


# ============================================================
# 0. File paths
# ============================================================

APPLICANT_CSV = "path/to/applicant_classified.csv"
ADDRESS_CSV = "path/to/address_cleaned.csv"
COUNTRYCODE_CSV = "path/to/countrycode_corrected.csv"
ADATE_CSV = "path/to/application_date.csv"

FINAL_OUT = "path/to/applicant_address_supplemented.csv"

ENCODING = "utf-8"


# ============================================================
# 1. Generic helpers
# ============================================================

def ensure_parent_dir(path: str) -> None:
    """Create parent directory if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def safe_str(value) -> str:
    """Convert missing values to an empty string."""
    if value is None:
        return ""
    return str(value)


def is_nonempty(value) -> bool:
    """Return True if a value is not an empty string."""
    return safe_str(value) != ""


def read_csv_rows(path: str, delimiter: str = ","):
    """Read a delimited file as dictionaries."""
    with open(path, "r", encoding=ENCODING, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        rows = [dict(row) for row in reader]

    return fieldnames, rows


def write_csv_rows(path: str, fieldnames, rows) -> None:
    """Write dictionaries to a CSV file."""
    ensure_parent_dir(path)

    with open(path, "w", encoding=ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def require_columns(fieldnames, columns, source_name: str) -> None:
    """Check required columns."""
    for col in columns:
        if col not in fieldnames:
            raise ValueError(f"Required column not found in {source_name}: {col}")


def drop_duplicates_keep_first(rows, key_fields):
    """Drop duplicates by key fields, keeping the first row."""
    seen = set()
    out = []

    for row in rows:
        key = tuple(safe_str(row.get(k, "")) for k in key_fields)

        if key in seen:
            continue

        seen.add(key)
        out.append(row)

    return out


def parse_int_maybe(value):
    """Parse an integer; return None if parsing fails."""
    text = safe_str(value).strip()

    if text == "":
        return None

    try:
        return int(text)
    except ValueError:
        return None


def parse_date_maybe(value):
    """Parse common date formats; return None if parsing fails."""
    text = safe_str(value).strip()

    if text == "":
        return None

    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y%m%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def split_into_batches(seq, n_batches: int):
    """Split a sequence into nearly equal batches."""
    seq = list(seq)
    n = len(seq)

    if n_batches <= 0:
        return [seq]

    q, r = divmod(n, n_batches)
    batches = []

    start = 0
    for i in range(n_batches):
        size = q + 1 if i < r else q
        batches.append(seq[start:start + size])
        start += size

    return batches


# ============================================================
# 2. Load input data
# ============================================================

def load_applicant_rows():
    """Load classified applicant data."""
    fieldnames, rows = read_csv_rows(APPLICANT_CSV)

    require_columns(
        fieldnames,
        ["ida", "applicant", "applicant_seq", "type"],
        APPLICANT_CSV,
    )

    rows = [
        row for row in rows
        if is_nonempty(row.get("ida"))
        and is_nonempty(row.get("applicant"))
        and is_nonempty(row.get("applicant_seq"))
    ]

    rows = drop_duplicates_keep_first(rows, ["ida", "applicant_seq"])

    for row in rows:
        row["applicant_seq_num"] = parse_int_maybe(row.get("applicant_seq"))

    return rows


def load_address_rows():
    """
    Load patent-level address data and merge it with country/region codes.

    The country table is used as a filter so that only records with valid
    address and country/region code are kept.
    """
    address_fields, address_rows = read_csv_rows(ADDRESS_CSV)
    country_fields, country_rows = read_csv_rows(COUNTRYCODE_CSV)

    require_columns(address_fields, ["ida", "address"], ADDRESS_CSV)
    require_columns(country_fields, ["ida"], COUNTRYCODE_CSV)

    if "countrycode_detected" in country_fields:
        country_col = "countrycode_detected"
    elif "country/region_code" in country_fields:
        country_col = "country/region_code"
    else:
        raise ValueError(
            f"Required country-code column not found in {COUNTRYCODE_CSV}."
        )

    address_rows = [
        {
            "ida": safe_str(row.get("ida")),
            "address": safe_str(row.get("address")),
        }
        for row in address_rows
    ]

    country_rows = [
        {
            "ida": safe_str(row.get("ida")),
            "country": safe_str(row.get(country_col)),
        }
        for row in country_rows
    ]

    country_rows = drop_duplicates_keep_first(country_rows, ["ida"])
    country_by_ida = {
        row["ida"]: row["country"]
        for row in country_rows
        if is_nonempty(row.get("ida")) and is_nonempty(row.get("country"))
    }

    merged = []

    for row in address_rows:
        ida = row["ida"]

        if ida not in country_by_ida:
            continue

        if not is_nonempty(ida) or not is_nonempty(row.get("address")):
            continue

        merged.append({
            "ida": ida,
            "address": row["address"],
            "country": country_by_ida[ida],
        })

    merged = drop_duplicates_keep_first(merged, ["ida"])

    return merged


def load_adate_rows():
    """Load application dates and derive application year."""
    fieldnames, rows = read_csv_rows(ADATE_CSV)

    require_columns(fieldnames, ["ida", "adate"], ADATE_CSV)

    out = []

    for row in rows:
        ida = safe_str(row.get("ida"))
        adate_dt = parse_date_maybe(row.get("adate"))

        if not is_nonempty(ida) or adate_dt is None:
            continue

        out.append({
            "ida": ida,
            "adate_dt": adate_dt,
            "adate": adate_dt.strftime("%Y-%m-%d"),
            "ayear": adate_dt.year,
        })

    out = drop_duplicates_keep_first(out, ["ida"])

    return out


# ============================================================
# 3. Merge and split applicant groups
# ============================================================

def merge_applicant_with_adate(applicant_rows, adate_rows):
    """Inner join applicant rows with application dates."""
    adate_by_ida = {
        row["ida"]: row
        for row in adate_rows
        if is_nonempty(row.get("ida"))
    }

    merged = []

    for row in applicant_rows:
        ida = safe_str(row.get("ida"))

        if ida not in adate_by_ida:
            continue

        merged_row = dict(row)
        merged_row["adate_dt"] = adate_by_ida[ida]["adate_dt"]
        merged_row["adate"] = adate_by_ida[ida]["adate"]
        merged_row["ayear"] = adate_by_ida[ida]["ayear"]

        merged.append(merged_row)

    return merged


def split_applicant_groups(applicant_rows):
    """Split applicant rows by sequence number and applicant type."""
    first_non_individual = []
    other_non_individual = []
    first_individual = []
    other_individual = []

    for row in applicant_rows:
        seq = row.get("applicant_seq_num")
        applicant_type = safe_str(row.get("type"))

        if seq == 1 and applicant_type != "individual":
            first_non_individual.append(dict(row))
        elif seq != 1 and applicant_type != "individual":
            other_non_individual.append(dict(row))
        elif seq == 1 and applicant_type == "individual":
            first_individual.append(dict(row))
        elif seq != 1 and applicant_type == "individual":
            other_individual.append(dict(row))

    return (
        first_non_individual,
        other_non_individual,
        first_individual,
        other_individual,
    )


def merge_rows_with_address(rows, address_rows):
    """Inner join rows with patent-level address data."""
    address_by_ida = {
        row["ida"]: row["address"]
        for row in address_rows
        if is_nonempty(row.get("ida")) and is_nonempty(row.get("address"))
    }

    merged = []

    for row in rows:
        ida = safe_str(row.get("ida"))

        if ida not in address_by_ida:
            continue

        merged_row = dict(row)
        merged_row["address"] = address_by_ida[ida]
        merged.append(merged_row)

    return merged


# ============================================================
# 4. Fill addresses for other non-individual applicants
# ============================================================

def build_first_applicant_reference(first_non_individual_rows):
    """Build reference records from first non-individual applicants."""
    return [
        {
            "applicant": safe_str(row.get("applicant")),
            "adate_dt": row.get("adate_dt"),
            "address": safe_str(row.get("address")),
        }
        for row in first_non_individual_rows
        if is_nonempty(row.get("applicant"))
        and is_nonempty(row.get("address"))
        and row.get("adate_dt") is not None
    ]


def build_other_applicant_rows(other_non_individual_rows):
    """Build rows to be filled for other non-individual applicants."""
    return [
        {
            "ida": safe_str(row.get("ida")),
            "applicant_seq": row.get("applicant_seq_num"),
            "applicant": safe_str(row.get("applicant")),
            "adate_dt": row.get("adate_dt"),
        }
        for row in other_non_individual_rows
        if is_nonempty(row.get("ida"))
        and is_nonempty(row.get("applicant"))
        and row.get("adate_dt") is not None
    ]


def fill_other_non_individual_addresses(first_rows, other_rows, n_batches: int = 10):
    """
    Fill addresses for non-first, non-individual applicants.

    For each applicant, the address is copied from the same applicant name
    among first applicants. If multiple candidates exist, the one with the
    nearest application date is selected. If ties remain, the lexicographically
    smallest address is used.
    """
    first_by_applicant = defaultdict(list)

    for row in first_rows:
        first_by_applicant[row["applicant"]].append(row)

    unique_applicants = sorted({row["applicant"] for row in other_rows})
    applicant_batches = split_into_batches(unique_applicants, n_batches)

    filled_rows = []

    for applicant_batch in applicant_batches:
        batch_set = set(applicant_batch)

        other_part = [
            row for row in other_rows
            if row["applicant"] in batch_set
        ]

        candidates = []

        for other_row in other_part:
            applicant = other_row["applicant"]

            for first_row in first_by_applicant.get(applicant, []):
                date_diff = abs(other_row["adate_dt"] - first_row["adate_dt"])
                key = (other_row["ida"], other_row["applicant_seq"])

                candidates.append({
                    "key": key,
                    "address": first_row["address"],
                    "date_diff": date_diff,
                })

        min_diff_by_key = {}

        for row in candidates:
            key = row["key"]
            date_diff = row["date_diff"]

            if key not in min_diff_by_key or date_diff < min_diff_by_key[key]:
                min_diff_by_key[key] = date_diff

        best_address_by_key = {}

        for row in sorted(candidates, key=lambda x: (x["key"], x["address"])):
            key = row["key"]

            if row["date_diff"] != min_diff_by_key[key]:
                continue

            if key not in best_address_by_key:
                best_address_by_key[key] = row["address"]

        for row in other_part:
            key = (row["ida"], row["applicant_seq"])

            filled_rows.append({
                "ida": row["ida"],
                "applicant": row["applicant"],
                "applicant_seq": row["applicant_seq"],
                "address": best_address_by_key.get(key, ""),
            })

    return filled_rows


# ============================================================
# 5. Build final result
# ============================================================

def keep_final_columns(rows):
    """Keep final output columns."""
    out = []

    for row in rows:
        out.append({
            "ida": safe_str(row.get("ida")),
            "applicant": safe_str(row.get("applicant")),
            "applicant_seq": safe_str(row.get("applicant_seq")),
            "address": safe_str(row.get("address")),
        })

    return out


def build_final_result(
    first_non_individual,
    other_non_individual_filled,
    first_individual,
    other_individual,
):
    """Concatenate all applicant groups."""
    final_rows = []
    final_rows.extend(keep_final_columns(first_non_individual))
    final_rows.extend(keep_final_columns(other_non_individual_filled))
    final_rows.extend(keep_final_columns(first_individual))
    final_rows.extend(keep_final_columns(other_individual))

    return final_rows


# ============================================================
# 6. Main pipeline
# ============================================================

def main() -> None:
    applicant_rows = load_applicant_rows()
    address_rows = load_address_rows()
    adate_rows = load_adate_rows()

    applicant_rows = merge_applicant_with_adate(applicant_rows, adate_rows)

    (
        first_non_individual,
        other_non_individual,
        first_individual,
        other_individual,
    ) = split_applicant_groups(applicant_rows)

    first_non_individual = merge_rows_with_address(
        first_non_individual,
        address_rows,
    )

    first_individual = merge_rows_with_address(
        first_individual,
        address_rows,
    )

    first_reference = build_first_applicant_reference(first_non_individual)
    other_to_fill = build_other_applicant_rows(other_non_individual)

    other_non_individual_filled = fill_other_non_individual_addresses(
        first_rows=first_reference,
        other_rows=other_to_fill,
        n_batches=10,
    )

    final_rows = build_final_result(
        first_non_individual=first_non_individual,
        other_non_individual_filled=other_non_individual_filled,
        first_individual=first_individual,
        other_individual=other_individual,
    )

    write_csv_rows(
        path=FINAL_OUT,
        fieldnames=["ida", "applicant", "applicant_seq", "address"],
        rows=final_rows,
    )


if __name__ == "__main__":
    main()
