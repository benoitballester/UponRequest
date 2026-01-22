#!/usr/bin/env bash
#
# download_sjr.sh — Download latest SCImago journal ranking CSV
#
# Usage: ./download_sjr.sh [YEAR] [OUTPUT_FILE]
# Example: ./download_sjr.sh 2024 ../2.data/scimagojr_journals_2024.csv
#
# If YEAR not provided, defaults to current year minus one.
#
# Note : 
# In the end, this was not used in the latest analyses
#
set -euo pipefail

# Determine year
if [ $# -ge 1 ]; then
  YEAR="$1"
else
  # default to previous year
  YEAR=$(date +%Y)
  YEAR=$((YEAR - 1))
fi

# Determine output file name
if [ $# -ge 2 ]; then
  OUTFILE="$2"
else
  OUTFILE="scimagojr_journals_${YEAR}.csv"
fi

# Construct download URL (SCImago seems to support an XLS download via ‘out=xls’ parameter)
URL="https://www.scimagojr.com/journalrank.php?year=${YEAR}&out=xls"

echo "Downloading SCImago Journal & Country Rank data for year ${YEAR} ..."
echo "URL: ${URL}"
echo "Output file: ${OUTFILE}"

# Use curl or wget to download
if command -v curl >/dev/null 2>&1; then
  curl -L "${URL}" -o "${OUTFILE}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${OUTFILE}" "${URL}"
else
  echo "Error: Neither curl nor wget found on this system."
  exit 1
fi

echo "Download completed."

# Optional: if the downloaded file is .xls or .xlsx, convert to CSV (requires e.g. ssconvert or xlsx2csv)
# Example using ssconvert:
# if command -v ssconvert >/dev/null 2>&1; then
#   ssconvert "${OUTFILE}" "${OUTFILE%.xls}.csv"
#   echo "Converted to CSV: ${OUTFILE%.xls}.csv"
# fi

exit 0
