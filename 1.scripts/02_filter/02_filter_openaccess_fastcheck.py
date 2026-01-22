#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_filter_openaccess_fastcheck.py

Filter meta_all_papers.tsv on OA_subset==1 and validate against
the official PMC OA Subset list (oa_file_list.csv).

This is an initial version, replaced by '02_filter_openaccess_subset.py' 

"""

import pandas as pd

# Load metadata
meta = pd.read_csv("2.data/meta_all_papers.tsv", sep="\t", dtype=str)

# Normalize column types
meta["OA_subset"] = meta["OA_subset"].astype(int)
meta["pmcid"] = meta["pmcid"].astype(str).str.replace("^PMC", "", regex=True)

# Filter to OA Subset
meta_oa = meta[meta["OA_subset"] == 1]
meta_oa.to_csv("2.data/meta_openaccess.tsv", sep="\t", index=False)
print(f"Saved {len(meta_oa):,} OA Subset articles to 2.data/meta_openaccess.tsv")

# Validate against FTP index
ftp_oa = pd.read_csv("2.data/pmc_ftp/oa_file_list.csv", dtype=str)
ftp_pmcs = set(ftp_oa["Accession ID"].astype(str).str.replace("^PMC", "", regex=True))

discrepancies = meta_oa[~meta_oa["pmcid"].isin(ftp_pmcs)]
print(f"{len(discrepancies):,} articles flagged OA_subset=1 but missing from FTP index.")
