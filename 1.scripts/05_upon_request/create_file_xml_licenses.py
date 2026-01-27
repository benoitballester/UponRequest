#!/usr/bin/env python3
"""
Create xml_licenses.tsv for the XML files actually distributed in 3.xml/.

For each *.xml in 3.xml/, we:
  1) infer the PMCID from the filename (e.g., "PMC1234567.xml" or "1234567.xml")
  2) lookup PMCID in an Excel table (e.g., meta_under_request_tagged.xlsx)
  3) write a TSV with: pmcid, pmid, doi, license
"""

from pathlib import Path
import re
import pandas as pd


# --- paths (edit if needed) ---
XML_DIR = Path("3.xml")
XLS_PATH = Path("2.data/meta_under_request_tagged.xlsx")  # or the uploaded xlsx path
OUT_TSV = Path("xml_licenses.tsv")


def infer_pmcid_from_filename(name: str) -> str:
    """
    Accepts:
      - PMC1234567.xml
      - 1234567.xml
      - PMCID1234567.xml (rare)
    Returns PMCID as digits-only string, e.g. "1234567".
    """
    stem = Path(name).stem
    m = re.search(r"(\d+)$", stem)
    return m.group(1) if m else ""


def find_col(df: pd.DataFrame, candidates, contains=False) -> str:
    """
    Find a column in df by exact (case-insensitive) match in candidates,
    or by substring match if contains=True.
    """
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    if not contains:
        for cand in candidates:
            if cand.lower() in lower:
                return lower[cand.lower()]
        return ""

    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return ""


def main():
    # List XML files to be reported (only those distributed in 3.xml/)
    xml_files = sorted([p for p in XML_DIR.glob("*.xml") if p.is_file()])
    if not xml_files:
        raise SystemExit(f"No XML files found in: {XML_DIR.resolve()}")

    # Load XLS
    df = pd.read_excel(XLS_PATH)

    # Identify key columns (robust to slightly different column names)
    col_pmcid = find_col(df, ["pmcid"], contains=True) or find_col(df, ["pmcid"])
    col_pmid = find_col(df, ["pmid"], contains=True) or find_col(df, ["pmid"])
    col_doi = find_col(df, ["doi"], contains=True) or find_col(df, ["doi"])
    col_license = (
        find_col(df, ["license", "licence"], contains=True)
        or find_col(df, ["license", "licence"])
    )

    if not col_pmcid:
        raise SystemExit("Could not find a PMCID column in the XLS (expected something like 'pmcid').")

    # Normalize PMCID in XLS to digits-only strings for matching
    df = df.copy()
    df[col_pmcid] = df[col_pmcid].astype(str).str.replace(r"\D+", "", regex=True)

    # Build a simple lookup table (first match per PMCID)
    keep_cols = [c for c in [col_pmcid, col_pmid, col_doi, col_license] if c]
    lookup = df[keep_cols].dropna(subset=[col_pmcid]).drop_duplicates(subset=[col_pmcid], keep="first")
    lookup = lookup.set_index(col_pmcid)

    rows = []
    for xml_path in xml_files:
        pmcid = infer_pmcid_from_filename(xml_path.name)

        pmid = ""
        doi = ""
        lic = ""

        if pmcid and pmcid in lookup.index:
            rec = lookup.loc[pmcid]
            if col_pmid:
                pmid = "" if pd.isna(rec.get(col_pmid)) else str(rec.get(col_pmid))
            if col_doi:
                doi = "" if pd.isna(rec.get(col_doi)) else str(rec.get(col_doi))
            if col_license:
                lic = "" if pd.isna(rec.get(col_license)) else str(rec.get(col_license))

        rows.append(
            {
                "xml_file": xml_path.name,
                "pmcid": pmcid,
                "pmid": pmid,
                "doi": doi,
                "license": lic,
            }
        )

    out = pd.DataFrame(rows)

    # Write TSV
    out.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"Wrote {OUT_TSV} with {len(out)} rows (one per XML in {XML_DIR}).")


if __name__ == "__main__":
    main()
