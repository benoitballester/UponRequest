#!/usr/bin/env python3
"""
Move XML files from 3.xml/ to xmls_removed_from_3xml_which_where_no_cc/
when their License in meta_under_request_tagged.xlsx is "NO-CC CODE".

Run from the repository root:
  python 1.scripts/05_upon_request/move_no_cc_xmls.py
"""

from pathlib import Path
import re
import shutil
import pandas as pd


XML_DIR = Path("3.xml")
XLSX_PATH = Path("2.data/meta_under_request_tagged.xlsx")
DEST_DIR = Path("xmls_removed_from_3xml_which_where_no_cc")
LOG_TSV = DEST_DIR / "moved_no_cc_xmls.tsv"


def infer_pmcid_from_filename(filename: str) -> str:
    """
    Accepts filenames like:
      - 1234567.xml
      - PMC1234567.xml
    Returns digits-only PMCID (e.g. "1234567"), or "" if not found.
    """
    stem = Path(filename).stem
    m = re.search(r"(\d+)", stem)
    return m.group(1) if m else ""


def main():
    if not XML_DIR.exists():
        raise SystemExit(f"Missing directory: {XML_DIR}")

    # Requires openpyxl in your env:
    #   conda install -c conda-forge openpyxl
    df = pd.read_excel(XLSX_PATH, engine="openpyxl")

    # Column names in your file: pmcid, pmid, DOI, License
    if "pmcid" not in df.columns or "License" not in df.columns:
        raise SystemExit("Expected columns not found in the XLSX (need at least 'pmcid' and 'License').")

    df = df.copy()
    df["pmcid_norm"] = df["pmcid"].astype(str).str.replace(r"\D+", "", regex=True)
    df["license_norm"] = df["License"].fillna("").astype(str).str.strip()

    no_cc_pmcids = set(df.loc[df["license_norm"].str.upper() == "NO-CC CODE", "pmcid_norm"])

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    moved_rows = []
    n_seen = 0

    for xml_path in sorted(XML_DIR.glob("*.xml")):
        n_seen += 1
        pmcid = infer_pmcid_from_filename(xml_path.name)
        if not pmcid:
            continue

        if pmcid in no_cc_pmcids:
            dest = DEST_DIR / xml_path.name
            shutil.move(str(xml_path), str(dest))
            moved_rows.append({"xml_file": xml_path.name, "pmcid": pmcid})

    moved = pd.DataFrame(moved_rows)

    if not moved.empty:
        meta = (
            df.loc[df["pmcid_norm"].isin(moved["pmcid"]), ["pmcid_norm", "pmid", "DOI", "License"]]
            .drop_duplicates("pmcid_norm")
            .rename(columns={"pmcid_norm": "pmcid", "DOI": "doi", "License": "license"})
        )
        out = moved.merge(meta, on="pmcid", how="left")[["xml_file", "pmcid", "pmid", "doi", "license"]]
        out.to_csv(LOG_TSV, sep="\t", index=False)
    else:
        # still write an empty log with headers
        pd.DataFrame(columns=["xml_file", "pmcid", "pmid", "doi", "license"]).to_csv(LOG_TSV, sep="\t", index=False)

    print(f"Scanned {n_seen} XML files in {XML_DIR}")
    print(f"Moved {len(moved_rows)} NO-CC CODE XML files to {DEST_DIR}")
    print(f"Log written to {LOG_TSV}")


if __name__ == "__main__":
    main()
