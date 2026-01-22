#!/usr/bin/env bash
# ===============================================================
# fetch_pmc_ftp.sh
# ---------------------------------------------------------------
# Downloads key PMC FTP index files for Open Access text-mining:
#   - oa_file_list.csv   : list of OA Subset articles + licenses
#   - PMC-ids.csv.gz     : mapping PMCID ↔ PMID ↔ DOI
#
# Files are stored in 2.data/pmc_ftp/
# A README.txt records the download date and provenance.
#
# Usage : 
# cd 2.data/
# or 
# # cd 2.data/pmc_ftp/
# bash ../1.scripts/00_fetch_pmc_ftp.sh
#
# 
# ===============================================================

set -euo pipefail

# --- Configuration ---
OA_URL="https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv"
IDS_URL="https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz"
OUTDIR="pmc_ftp"

# --- Create output directory ---
mkdir -p "${OUTDIR}"

# --- Download files ---
echo "Downloading oa_file_list.csv ..."
wget -q -N -P "${OUTDIR}" "${OA_URL}"

echo "Downloading PMC-ids.csv.gz ..."
wget -q -N -P "${OUTDIR}" "${IDS_URL}"

# --- Write README with provenance ---
DATE_LOCAL=$(date +"%Y-%m-%dT%H:%M:%S%z")

{
  echo "PMC FTP Snapshot"
  echo "================"
  echo "Download date (local): ${DATE_LOCAL}"
  echo
  echo "Files:"
  echo "  - oa_file_list.csv"
  echo "  - PMC-ids.csv.gz"
  echo
  echo "Sources:"
  echo "  ${OA_URL}"
  echo "  ${IDS_URL}"
  echo
  echo "Description:"
  echo "  oa_file_list.csv : lists all OA Subset articles and their licenses."
  echo "  PMC-ids.csv.gz   : provides mappings between PMCID, PMID, and DOI."
  echo
  echo "Downloaded with:"
  wget --version | head -n1
  echo
  echo "Notes:"
  echo "  - Files are updated periodically by NCBI."
  echo "  - Re-run this script to refresh local copies."
} > "${OUTDIR}/README.txt"

echo "Done. Files stored in ${OUTDIR}/"
