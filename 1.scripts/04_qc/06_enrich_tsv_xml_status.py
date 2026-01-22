#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_check_xml_status.py
----------------------
For each PMCID in 2.data/meta_openaccess.tsv,
check whether 3.xml/{pmcid}.xml exists and record:

  xml_exists (1/0)
  xml_size (bytes)
  xml_md5 (MD5 checksum)

Outputs:
  2.data/meta_openaccess_enriched.tsv
"""

import hashlib
import pandas as pd
from pathlib import Path

META_FILE = Path("2.data/meta_openaccess.tsv")
XML_DIR   = Path("3.xml")
OUT_FILE  = Path("2.data/meta_openaccess_enriched.tsv")

def md5sum(file_path, blocksize=65536):
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    if not META_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {META_FILE}")
    if not XML_DIR.exists():
        raise FileNotFoundError(f"XML directory not found: {XML_DIR}")

    df = pd.read_csv(META_FILE, sep="\t", dtype=str)
    print(f"Loaded {len(df):,} records from {META_FILE}")

    xml_exists, xml_size, xml_md5 = [], [], []

    for pmcid in df["pmcid"].astype(str):
        xml_path = XML_DIR / f"{pmcid}.xml"
        if xml_path.exists():
            xml_exists.append(1)
            xml_size.append(xml_path.stat().st_size)
            xml_md5.append(md5sum(xml_path))
        else:
            xml_exists.append(0)
            xml_size.append(0)
            xml_md5.append("")

    df["xml_exists"] = xml_exists
    df["xml_size"]   = xml_size
    df["xml_md5"]    = xml_md5

    df.to_csv(OUT_FILE, sep="\t", index=False)
    print(f"Wrote enriched file: {OUT_FILE}")
    print(df["xml_exists"].value_counts())

if __name__ == "__main__":
    main()
