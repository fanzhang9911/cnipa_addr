#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：cnipa_address
@File    ：gcj2wgs.py
@IDE     ：PyCharm
@Machine : 12700KF + 4060Ti16G
@Author  ：Fan ZHANG
@Date    ：2026/04/20
@Note    : Utility for converting coordinates from GCJ-02 to WGS84.
'''


import csv
import math
import argparse


# Ellipsoid / projection parameters used by the standard GCJ-02 conversion formula.
PI = 3.14159265358979324
A = 6378245.0
EE = 0.00669342162296594323


def gcj02_to_wgs84(lon, lat):
    """
    Convert one coordinate pair from GCJ-02 to WGS84.

    Parameters
    ----------
    lon : float
        Longitude in GCJ-02.
    lat : float
        Latitude in GCJ-02.

    Returns
    -------
    tuple
        A tuple (wgs_lon, wgs_lat) in WGS84.

    Notes
    -----
    This function follows the standard one-step approximate inverse formula:
    1. Compute the longitude/latitude offsets.
    2. Transform the offsets based on ellipsoid parameters.
    3. Subtract the offsets from the input GCJ-02 coordinate to obtain
       an approximate WGS84 coordinate.
    """
    # Shift the input coordinate to the reference region used in the formula.
    x = lon - 105.0
    y = lat - 35.0

    # Compute longitude offset.
    dlon = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    dlon += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0
    dlon += (20.0 * math.sin(x * PI) + 40.0 * math.sin(x / 3.0 * PI)) * 2.0 / 3.0
    dlon += (150.0 * math.sin(x / 12.0 * PI) + 300.0 * math.sin(x / 30.0 * PI)) * 2.0 / 3.0

    # Compute latitude offset.
    dlat = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    dlat += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0
    dlat += (20.0 * math.sin(y * PI) + 40.0 * math.sin(y / 3.0 * PI)) * 2.0 / 3.0
    dlat += (160.0 * math.sin(y / 12.0 * PI) + 320.0 * math.sin(y * PI / 30.0)) * 2.0 / 3.0

    # Apply ellipsoid correction.
    radlat = lat / 180.0 * PI
    magic = math.sin(radlat)
    magic = 1 - EE * magic * magic
    sqrtmagic = math.sqrt(magic)

    dlat = (dlat * 180.0) / (((A * (1 - EE)) / (magic * sqrtmagic)) * PI)
    dlon = (dlon * 180.0) / ((A / sqrtmagic * math.cos(radlat)) * PI)

    # One-step approximate inverse transform.
    wgslon = lon - dlon
    wgslat = lat - dlat
    return wgslon, wgslat


def convert_csv(
    input_file,
    output_file,
    lon_col,
    lat_col,
    out_lon_col="lon_wgs84",
    out_lat_col="lat_wgs84",
    encoding="utf-8"
):
    """
    Read a CSV file, convert GCJ-02 coordinates to WGS84, and write a new CSV file.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file.
    output_file : str
        Path to the output CSV file.
    lon_col : str
        Name of the longitude column in the input file.
    lat_col : str
        Name of the latitude column in the input file.
    out_lon_col : str, default "lon_wgs84"
        Name of the output longitude column.
    out_lat_col : str, default "lat_wgs84"
        Name of the output latitude column.
    """
    with open(input_file, "r", encoding=encoding, newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []

        if lon_col not in fieldnames:
            raise ValueError(f"Longitude column not found: {lon_col}")
        if lat_col not in fieldnames:
            raise ValueError(f"Latitude column not found: {lat_col}")

        # Append output column names if they do not already exist.
        if out_lon_col not in fieldnames:
            fieldnames.append(out_lon_col)
        if out_lat_col not in fieldnames:
            fieldnames.append(out_lat_col)

        with open(output_file, "w", encoding=encoding, newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                lon_str = row.get(lon_col, "").strip()
                lat_str = row.get(lat_col, "").strip()

                # Leave output blank if coordinates are missing.
                if lon_str == "" or lat_str == "":
                    row[out_lon_col] = ""
                    row[out_lat_col] = ""
                else:
                    try:
                        lon = float(lon_str)
                        lat = float(lat_str)
                        wgs_lon, wgs_lat = gcj02_to_wgs84(lon, lat)

                        # Format to a stable decimal representation for CSV output.
                        row[out_lon_col] = f"{wgs_lon:.10f}"
                        row[out_lat_col] = f"{wgs_lat:.10f}"
                    except ValueError:
                        # Leave output blank if parsing fails.
                        row[out_lon_col] = ""
                        row[out_lat_col] = ""

                writer.writerow(row)


def main():
    """
    Command-line entry point.

    Example
    -------
    python gcj2wgs.py \
        --input input.csv \
        --output output.csv \
        --lon-col longitude \
        --lat-col latitude
    """
    parser = argparse.ArgumentParser(
        description="Convert GCJ-02 coordinates in a CSV file to WGS84."
    )
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--lon-col", default="lon", help="Name of the GCJ-02 longitude column")
    parser.add_argument("--lat-col", default="lat", help="Name of the GCJ-02 latitude column")
    parser.add_argument("--out-lon-col", default="lon_wgs84", help="Name of the output WGS84 longitude column")
    parser.add_argument("--out-lat-col", default="lat_wgs84", help="Name of the output WGS84 latitude column")
    parser.add_argument("--encoding", default="utf-8", help="CSV file encoding")
    args = parser.parse_args()

    convert_csv(
        input_file=args.input,
        output_file=args.output,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        out_lon_col=args.out_lon_col,
        out_lat_col=args.out_lat_col,
        encoding=args.encoding,
    )

    print(f"Done! Output saved to: {args.output}")


if __name__ == "__main__":
    main()
