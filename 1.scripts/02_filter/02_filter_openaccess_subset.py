#!/usr/bin/env python3
# ===============================================================
# 02_filter_openaccess_subset.py
# ---------------------------------------------------------------
# Filters the master metadata table (meta_all_papers.tsv) to keep
# only articles that are part of the official PMC Open Access (OA)
# Subset, based on the authoritative oa_file_list.csv downloaded
# from the PMC FTP.
#
# Input:
#   2.data/meta_all_papers.tsv
#   2.data/pmc_ftp/oa_file_list.csv
#
# Output:
#   2.data/meta_openaccess.tsv
#
# This file (meta_openaccess.tsv) will serve as the base corpus for
# downstream steps: XML download, parsing, and text-mining analyses.
# ===============================================================

import pandas as pd
from pathlib import Path

# --- Input paths ---
DATA_DIR = Path("2.data")
META_ALL = DATA_DIR / "meta_all_papers.tsv"
OA_LIST  = DATA_DIR / "pmc_ftp" / "oa_file_list.csv"

# --- Output path ---
META_OA = DATA_DIR / "meta_openaccess.tsv"

# --- Check presence of required files ---
if not META_ALL.exists():
    raise FileNotFoundError(f"Missing input file: {META_ALL}")
if not OA_LIST.exists():
    raise FileNotFoundError(f"Missing OA Subset index: {OA_LIST}")


print("Loading metadata...")
meta = pd.read_csv(META_ALL, sep="\t", dtype=str)
oa_df = pd.read_csv(OA_LIST, dtype=str)

# --- Normalize PMCID format (remove 'PMC' prefix) ---
meta["pmcid"] = meta["pmcid"].astype(str).str.replace("^PMC", "", regex=True)
oa_df["Accession ID"] = oa_df["Accession ID"].astype(str).str.replace("^PMC", "", regex=True)

# --- Intersect ---
oa_pmcs = set(oa_df["Accession ID"])
meta_oa = meta[meta["pmcid"].isin(oa_pmcs)].copy()

# --- Merge license info ---
meta_oa = meta_oa.merge(
    oa_df[["Accession ID", "License"]],
    left_on="pmcid", right_on="Accession ID", how="left"
)
meta_oa.drop(columns=["Accession ID"], inplace=True)



# --- Save filtered corpus ---
meta_oa.to_csv(META_OA, sep="\t", index=False)

# --- Summary ---
print("--------------------------------------------------")
print(f"Total articles in meta_all_papers.tsv : {len(meta):,}")
print(f"Articles in PMC OA Subset (kept)     : {len(meta_oa):,}")
print(f"Output written to                    : {META_OA}")
print("--------------------------------------------------")
