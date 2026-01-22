#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_validate_xml_integrity.py
----------------------------
Scan 3.xml/*.xml and flag files that appear truncated, HTML, or not well-formed OAI-PMH/JATS.

Outputs:
  - 3.xml/invalid_xml.tsv        (pmcid, path, size_bytes, reasons)
  - 3.xml/to_redownload.txt      (pmcid per line)

Heuristics:
  - Size threshold (default 10 KiB) to catch obvious truncation
  - Must contain closing </OAI-PMH> and parse as XML
  - Root should be OAI-PMH (any namespace)
  - Should contain a <article> element (any namespace)
  - Rejects files that appear to be HTML instead of XML

Usage:
  python 04_validate_xml_integrity.py
  python 04_validate_xml_integrity.py --dir 3.xml --min-kib 10 --workers 8
"""

import argparse
import concurrent.futures as cf
import os
from pathlib import Path
from typing import List, Tuple
import xml.etree.ElementTree as ET

def is_html_snippet(s: str) -> bool:
    head = s[:4096].lower()
    return ("<html" in head) or ("<!doctype html" in head)

def has_article_tag(root: ET.Element) -> bool:
    # Look for any-element named 'article' regardless of namespace
    for elem in root.iter():
        if elem.tag.endswith("article"):
            return True
    return False

def root_is_oai(root: ET.Element) -> bool:
    # Allow namespaced tags; check localname matches 'OAI-PMH'
    return root.tag.endswith("OAI-PMH")

def check_one(path: Path, min_bytes: int) -> Tuple[str, str, int, List[str]]:
    """
    Returns: (pmcid, str(path), size_bytes, reasons[])
    reasons empty => OK
    """
    pmcid = path.stem
    reasons: List[str] = []
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return (pmcid, str(path), 0, ["missing_file"])

    # Quick size gate
    if size < min_bytes:
        reasons.append("too_small")

    # Read content
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return (pmcid, str(path), size, [f"read_error:{e}"])

    # HTML masquerading as XML?
    if is_html_snippet(data):
        reasons.append("looks_like_html")

    # Missing closing tag often indicates truncation
    if "</OAI-PMH>" not in data:
        reasons.append("missing_closing_OAI-PMH")

    # Basic well-formedness
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        reasons.append(f"xml_parse_error:{e}")
        return (pmcid, str(path), size, reasons)

    # Root should be OAI-PMH
    if not root_is_oai(root):
        reasons.append("unexpected_root")

    # Expect a JATS <article> somewhere in metadata
    if not has_article_tag(root):
        reasons.append("no_article_element")

    return (pmcid, str(path), size, reasons)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="3.xml", help="Directory containing .xml files")
    ap.add_argument("--min-kib", type=int, default=10, help="Min file size in KiB to accept")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Parallel workers")
    args = ap.parse_args()

    in_dir = Path(args.dir)
    xml_paths = sorted(in_dir.glob("*.xml"))
    if not xml_paths:
        print(f"No XML files found under {in_dir}")
        return

    min_bytes = args.min_kib * 1024
    bad: List[Tuple[str, str, int, List[str]]] = []
    total = len(xml_paths)

    print(f"Scanning {total:,} XML files in {in_dir} ...")

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for pmcid, pstr, size, reasons in ex.map(lambda p: check_one(p, min_bytes), xml_paths):
            if reasons:
                bad.append((pmcid, pstr, size, reasons))

    # Write outputs
    out_tsv = in_dir / "invalid_xml.tsv"
    out_list = in_dir / "to_redownload.txt"

    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("pmcid\tpath\tsize_bytes\treasons\n")
        for pmcid, pstr, size, reasons in bad:
            f.write(f"{pmcid}\t{pstr}\t{size}\t{';'.join(reasons)}\n")

    with out_list.open("w", encoding="utf-8") as f:
        for pmcid, _, _, _ in bad:
            f.write(pmcid + "\n")

    # Console summary
    print(f"\n===== INTEGRITY CHECK SUMMARY =====")
    print(f"Total XMLs scanned:   {total:,}")
    print(f"Flagged as dubious:   {len(bad):,}")
    if bad:
        # show a few examples
        print("Examples:")
        for pmcid, _, size, reasons in bad[:10]:
            print(f"  {pmcid}  ({size} bytes)  ->  {', '.join(reasons)}")
    print(f"\nWrote: {out_tsv}")
    print(f"Wrote: {out_list}")
    print("You can remove these and re-run the downloader to refill them.")

if __name__ == "__main__":
    main()
