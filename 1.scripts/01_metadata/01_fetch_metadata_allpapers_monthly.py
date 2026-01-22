#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------------------------------------------
Fetch ALL research papers (OA + non-OA) month by month per journal,
avoid PubMed’s 9,999 record limit, print monthly progress, and
produce a rich TSV with PubMed metadata.

Output TSV columns:
  pmid    pmcid   OA  OA_subset  journal pub_date year
  PublicationType  DOI  Source  SO  Title

IMPORTANT — OA vs OA_subset:
  • OA = 1  -> article has a PMCID (present in PMC).
              Includes both:
                - “Free in PMC” (viewable but not reusable XML)
                - PMC Open Access Subset (true OA)
    OA = 0  -> no PMCID.

  • OA_subset = 1 -> PMCID appears in the official PMC Open Access Subset
                    list (oa_file_list.csv from NCBI FTP).
                    Means reusable, downloadable XML is available.
    OA_subset = 0 -> not in OA subset (often “Free in PMC”).

To enable OA_subset detection:
  1) Download from:
     ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv
  2) Save as: 2.data/oa_file_list.csv or 2.data/pmc_ftp/oa_file_list.csv 
  3) Script auto-detects correct column ("Accession ID" or "PMCID").
"""

from __future__ import annotations
import os, re, time, csv
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import pandas as pd
from urllib.error import HTTPError, URLError
from Bio import Entrez

# =========================
# CONFIG
# =========================
Entrez.email = os.environ.get("NCBI_EMAIL", "NAME@inserm.fr")
API_KEY_PATH = Path("0.env/ncbi_api.key")
if API_KEY_PATH.exists():
    Entrez.api_key = API_KEY_PATH.read_text().strip()

MAX_REQ_PER_SEC = 8 if getattr(Entrez, "api_key", None) else 3
MIN_INTERVAL = 1.0 / MAX_REQ_PER_SEC

# Date range
DATE_START = "2010/01/01"
DATE_END   = "2025/09/30"
START_YEAR = int(DATE_START.split("/")[0])
END_YEAR   = int(DATE_END.split("/")[0])

# Journals
JOURNALS = [
    "Lancet", "Cell", "Science", "Nature", "Nature Communications",
    "Nature Genetics", "Nature Medicine",
    "Nucleic Acids Research",
    "Genome Research", "Genome Biology", "Bioinformatics",
    "NAR Genomics and Bioinformatics", "eLife", "PLOS Genetics",
    "PLOS Biology", "Cell Genomics", "BMC Genomics",
    "BMC Bioinformatics", "Frontiers in Genetics",
    "EMBO Reports", "Molecular Systems Biology", "The EMBO Journal",
]

EXCLUDE_TYPES = {
    "Comment", "Editorial", "News", "Letter", "Published Erratum",
    "Correction", "Retracted Publication", "Retraction of Publication",
    "Expression of Concern", "Introductory Journal Article"
}

ESEARCH_RETMAX = 1000
ESUMMARY_BATCH = 200
MAX_WORKERS = 5

DATA_DIR = Path("2.data")
FTP_DIR = Path("pmc_ftp")
OUT_META = DATA_DIR / "meta_all_papers.tsv"
OUT_IDS  = DATA_DIR / "metadata_ids_all_papers.txt"
OA_CSV   = DATA_DIR / FTP_DIR / "oa_file_list.csv"

META_COLS = [
    "pmid", "pmcid", "OA", "OA_subset", "journal", "pub_date", "year",
    "PublicationType", "DOI", "Source", "SO", "Title"
]

WRITE_LOCK = Lock()
_RATE_LOCK = Lock()

OA_SET: set[str] = set()
OA_SET_LOADED: bool = False

def log(msg: str): print(msg, flush=True)

# =========================
# Helper functions
# =========================
_last_call_time = 0.0
def _polite_pause():
    global _last_call_time
    with _RATE_LOCK:
        now = time.time()
        delta = now - _last_call_time
        if delta < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - delta)
        _last_call_time = time.time()

def entrez_read_with_retry(func, *args, **kw):
    backoff = 6
    for i in range(1,7):
        try:
            _polite_pause()
            h = func(*args, **kw)
            r = Entrez.read(h)
            h.close()
            return r
        except (HTTPError, URLError) as e:
            log(f"⚠️ HTTP error {e} (try {i}/6), sleep {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
        except Exception as e:
            log(f"⚠️ Entrez error {e} (try {i}/6), sleep 5s")
            time.sleep(5)
    return None

def normalize_date_freeform(s: str) -> str:
    if not s: return ""
    s = s.strip()
    try:
        ts = pd.to_datetime(s, errors="raise")
        return f"{ts.year}-{ts.month}-{ts.day}"
    except Exception:
        return s

def chunk_iter(xs: List[str], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

def load_done_pmids() -> set[str]:
    done = set()
    if OUT_META.exists():
        try:
            df = pd.read_csv(OUT_META, sep="\t", usecols=["pmid"], dtype=str)
            done.update(df["pmid"].dropna())
        except Exception:
            pass
    return done

def load_oa_subset_csv(path: Path) -> None:
    """Load OA subset PMCIDs from oa_file_list.csv."""
    global OA_SET, OA_SET_LOADED
    if not path.exists():
        log(f"ℹ️ OA CSV not found at {path}. OA_subset will be 0 for all rows.")
        OA_SET_LOADED = False
        return

    try:
        df = pd.read_csv(path, dtype=str)
        col = None
        for c in ["Accession ID", "PMCID"]:
            if c in df.columns:
                col = c
                break
        if not col:
            log(f"⚠️ Neither 'Accession ID' nor 'PMCID' found in {path}. OA_subset=0.")
            OA_SET_LOADED = False
            return

        pmc_digits = df[col].astype(str).str.replace("^PMC", "", regex=True).str.replace(r"\D", "", regex=True)
        OA_SET = set(pmc_digits.dropna().tolist())
        OA_SET_LOADED = True
        log(f"✅ Loaded OA subset list from {path} (n={len(OA_SET):,})")

    except Exception as e:
        log(f"⚠️ Failed to load {path}: {e}")
        OA_SET_LOADED = False

# =========================
# Monthly PubMed search
# =========================
def esearch_monthly_pmids(journal: str) -> List[str]:
    all_ids = set()
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            start = f"{year}/{month:02d}/01"
            end   = f"{year}/{month:02d}/31"
            term  = f'("{journal}"[TA]) AND ("{start}"[PDAT] : "{end}"[PDAT])'
            rec   = entrez_read_with_retry(Entrez.esearch, db="pubmed", term=term, retmax=ESEARCH_RETMAX)
            if not rec:
                continue
            count = int(rec.get("Count", 0))
            if count == 0:
                continue
            log(f"   {journal} {year}-{month:02d}: {count} papers")
            all_ids.update(rec.get("IdList", []) or [])
            if count > len(rec.get("IdList", []) or []):
                retstart = len(rec.get("IdList", []) or [])
                while retstart < count and retstart < 9999:
                    rec2 = entrez_read_with_retry(Entrez.esearch, db="pubmed", term=term, retstart=retstart, retmax=ESEARCH_RETMAX)
                    if not rec2:
                        break
                    all_ids.update(rec2.get("IdList", []) or [])
                    retstart += ESEARCH_RETMAX
    return sorted(all_ids)

# =========================
# Parse summaries
# =========================
def parse_docsum(doc, fallback_journal_label: str) -> Dict[str, str]:
    pmid = str(doc.get("Id", "")).strip()
    title = str(doc.get("Title", "") or "").strip()
    source = str(doc.get("FullJournalName", doc.get("Source", fallback_journal_label)) or "").strip()
    so = str(doc.get("SO", "") or "").strip()
    pubdate = str(doc.get("PubDate", doc.get("EPubDate", "")) or "").strip()
    pub_date = normalize_date_freeform(pubdate)
    year = ""
    if pubdate:
        m = re.search(r"(\d{4})", pubdate)
        if m:
            year = m.group(1)

    pubtypes = doc.get("PubTypeList", []) or []
    pubtypes_str = ", ".join(pubtypes) if pubtypes else ""

    pmcid_digits = ""
    doi = ""
    ids = doc.get("ArticleIds", {})
    if isinstance(ids, dict):
        pmc_raw = str(ids.get("pmc", "") or "")
        pmcid_digits = re.sub(r"\D", "", pmc_raw)
        doi = str(ids.get("doi", "") or "").strip()

    for t in pubtypes:
        if any(t.lower() == bad.lower() for bad in EXCLUDE_TYPES):
            return {}

    OA = "1" if pmcid_digits else "0"
    if OA == "1" and OA_SET_LOADED:
        OA_subset = "1" if pmcid_digits in OA_SET else "0"
    else:
        OA_subset = "0"

    return {
        "pmid": pmid,
        "pmcid": pmcid_digits,
        "OA": OA,
        "OA_subset": OA_subset,
        "journal": source,
        "pub_date": pub_date,
        "year": year,
        "PublicationType": pubtypes_str,
        "DOI": doi,
        "Source": source,
        "SO": so,
        "Title": title,
    }

def esummary_to_rows(pmids: List[str], journal_label: str) -> List[Dict[str, str]]:
    if not pmids: return []
    rec = entrez_read_with_retry(Entrez.esummary, db="pubmed", id=",".join(pmids))
    if not rec: return []
    rows = []
    for doc in rec:
        try:
            row = parse_docsum(doc, journal_label)
            if row: rows.append(row)
        except Exception as e:
            log(f"⚠️ parse_docsum failed: {e}")
    return rows

# =========================
# Save
# =========================
def save_append(rows: List[Dict[str, str]]):
    if not rows: return
    df = pd.DataFrame(rows, columns=META_COLS).fillna("")
    with WRITE_LOCK:
        header = not OUT_META.exists()
        df.to_csv(OUT_META, sep="\t", index=False, mode="a", header=header, quoting=csv.QUOTE_MINIMAL)
        with open(OUT_IDS, "a") as f:
            for x in df["pmid"]: f.write(x + "\n")

# =========================
# Main
# =========================
def main():
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    load_oa_subset_csv(OA_CSV)
    done = load_done_pmids()
    log("------------------------------------------------------------")
    log(f"Fetching ALL papers ({DATE_START}–{DATE_END}) month-by-month from {len(JOURNALS)} journals")
    log(f"API key: {'yes' if getattr(Entrez, 'api_key', None) else 'no'}  |  rate ≤ {MAX_REQ_PER_SEC}/s")
    log("------------------------------------------------------------")

    stats = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut2j = {ex.submit(lambda j=j: process_journal(j, done), j): j for j in JOURNALS}
        for fut in as_completed(fut2j):
            j = fut2j[fut]
            try:
                stats[j] = fut.result()
            except Exception as e:
                log(f"❌ {j} failed: {e}")

    log("\n================ SUMMARY ================")
    tot_found = sum(s.get("found", 0) for s in stats.values())
    tot_written = sum(s.get("written", 0) for s in stats.values())
    log(f"TOTAL: found={tot_found:,}, written={tot_written:,}")
    log(f"Elapsed: {time.time()-t0:.1f}s")
    log("=========================================")

def process_journal(journal: str, done: set[str]) -> Dict[str, int]:
    log(f"\n🔎 {journal}")
    ids = esearch_monthly_pmids(journal)
    total = len(ids)
    new = [x for x in ids if x not in done]
    log(f"  Total for {journal}: {total} | new {len(new)}")
    written = 0
    for chunk in chunk_iter(new, ESUMMARY_BATCH):
        rows = esummary_to_rows(chunk, journal)
        save_append(rows)
        done.update([r["pmid"] for r in rows])
        written += len(rows)
        log(f"   • wrote {written}/{len(new)}")
    return {"found": total, "new": len(new), "written": written}

if __name__ == "__main__":
    main()
