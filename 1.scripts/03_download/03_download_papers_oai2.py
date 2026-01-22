#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_download_papers_oai2.py (with resume-enabled, live logging, 2025 endpoint)
-----------------------------------------------------------------------
Bulk download XML for PMC Open Access Subset items via OAI-PMH.

What this script does:
- Reads PMCIDs from 2.data/meta_openaccess.tsv (column: pmcid)
- Talks to the modern OAI-PMH endpoint:
    https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/
  using verb=GetRecord, identifier=oai:pubmedcentral.nih.gov:{pmcid},
  metadataPrefix=pmc (full text in NISO JATS Archiving & Interchange)
- Logs every event immediately to 3.xml/download_log.txt (timestamps)
- Resumes: skips OK/404 and existing files; retries FAIL/ERR
- Polite concurrency and retries with exponential backoff on 429
- Clean Ctrl+C: flushes logs and stops threads safely
"""

import os
import sys
import time
import random
import signal
import threading
import queue
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# -------------------------
# Config (as requested)
# -------------------------
META_FILE     = Path("2.data/meta_openaccess.tsv")
OUT_DIR       = Path("3.xml")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE      = OUT_DIR / "download_log.txt"

MAX_WORKERS   = 5            # ~3–5 req/s total
TIMEOUT       = 45
RETRY_LIMIT   = 3
SLEEP_BETWEEN = (0.6, 1.2)   # jitter between requests

EMAIL   = os.environ.get("NCBI_EMAIL", "NAME@inserm.fr")
HEADERS = {
    "User-Agent": f"PMC-OAI-Downloader/2025 (contact: {EMAIL})",
    "Accept": "application/xml",
}

# ------------------------------------------------------------------
# Base URL (Sept 2025)
# ------------------------------------------------------------------
BASE_URL = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"

# -------------------------
# Thread-safe logger
# -------------------------
_log_lock = threading.Lock()
def log_write(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
    print(msg, flush=True)

# -------------------------
# Resume helpers
# -------------------------
def parse_log_completed_and_failed():
    done, failed = set(), set()
    if LOG_FILE.exists():
        with open(LOG_FILE, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                # example: "[ts] OK   1234567"
                if len(parts) < 3:
                    continue
                status, pmc = parts[1], parts[2]
                if status in ("OK", "404", "SKIP"):
                    done.add(pmc)
                elif status in ("FAIL", "ERR"):
                    failed.add(pmc)
    existing = {p.stem for p in OUT_DIR.glob("*.xml")}
    done |= existing
    return done, failed

def load_pmcids_from_meta():
    df = pd.read_csv(META_FILE, sep="\t", dtype=str)
    if "pmcid" not in df.columns:
        raise RuntimeError("meta_openaccess.tsv is missing column 'pmcid'")
    pmcids = (
        df["pmcid"].dropna().astype(str).str.replace("^PMC", "", regex=True)
    )
    return list(pmcids)

# -------------------------
# Counters for this run
# -------------------------
_counts_lock = threading.Lock()
_counts = {"OK": 0, "SKIP": 0, "404": 0, "FAIL": 0, "429": 0}

def bump(status):
    with _counts_lock:
        if status in _counts:
            _counts[status] += 1

# -------------------------
# Downloader
# -------------------------
def fetch_one(session: requests.Session, pmcid: str):
    out_path = OUT_DIR / f"{pmcid}.xml"
    if out_path.exists():
        bump("SKIP")
        log_write(f"SKIP {pmcid} (exists)")
        return
    
    params = {
    "verb": "GetRecord",
    "identifier": f"oai:pubmedcentral.nih.gov:{pmcid}",
    "metadataPrefix": "pmc"
    }

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            r = session.get(BASE_URL, params=params, headers=HEADERS, timeout=TIMEOUT)
            status = r.status_code

            # Success with plausible XML
            if status == 200 and r.text and "<OAI-PMH" in r.text:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(r.text)
                bump("OK")
                log_write(f"OK   {pmcid}")
                return

            # Permanent not found
            if status == 404:
                bump("404")
                log_write(f"404  {pmcid}")
                return

            # Rate limited -> exponential backoff
            if status == 429:
                bump("429")
                wait = [90, 180, 300][min(attempt-1, 2)]
                log_write(f"429  {pmcid} (rate limit, sleeping {wait}s...)")
                time.sleep(wait)
                continue

            # Other HTTP codes -> brief pause then retry
            log_write(f"FAIL {pmcid}: HTTP {status}")
            if attempt < RETRY_LIMIT:
                time.sleep(1.0)

        except Exception as e:
            # Network/timeout/etc
            log_write(f"FAIL {pmcid}: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(1.0)

        # polite jitter every loop
        time.sleep(random.uniform(*SLEEP_BETWEEN))

    # If we exit the loop without success or permanent 404:
    bump("FAIL")
    log_write(f"FAIL {pmcid}: unexpected no-response")

def worker(q: queue.Queue):
    with requests.Session() as session:
        while True:
            pmc = q.get()
            if pmc is None:
                q.task_done()
                break
            try:
                fetch_one(session, pmc)
            finally:
                q.task_done()

# -------------------------
# proper shutdown, i think
# -------------------------
_stop_requested = False
def handle_sigint(sig, frame):
    global _stop_requested
    if not _stop_requested:
        _stop_requested = True
        log_write("Interrupted — flushing logs and stopping threads safely...")
    else:
        # second Ctrl+C forces exit
        log_write("Force exit requested.")
        os._exit(1)

signal.signal(signal.SIGINT, handle_sigint)

# -------------------------
# Main
# -------------------------
def main():
    start = time.time()

    pmcids = load_pmcids_from_meta()
    done, _failed = parse_log_completed_and_failed()

    pending = [p for p in pmcids if p not in done]
    log_write(f"Resuming: {len(pending):,} remaining of {len(pmcids):,} total.")

    if not pending:
        print("Nothing to do.")
        return

    q = queue.Queue()
    threads = []
    for _ in range(MAX_WORKERS):
        t = threading.Thread(target=worker, args=(q,), daemon=True)
        t.start()
        threads.append(t)

    try:
        for pmc in pending:
            if _stop_requested:
                break
            q.put(pmc)

        # Wait for queue to drain unless stop requested
        while not _stop_requested:
            try:
                q.join()
                break
            except KeyboardInterrupt:
                # handled by signal handler
                pass

    finally:
        # Signal workers to stop and join
        for _ in threads:
            q.put(None)
        for t in threads:
            t.join()

    elapsed_min = (time.time() - start) / 60.0
    # Summarize for this run only (based on counters)
    with _counts_lock:
        ok    = _counts["OK"]
        skip  = _counts["SKIP"]
        nf    = _counts["404"]
        fail  = _counts["FAIL"]
        rl    = _counts["429"]

    log_write(
        f"SUMMARY: OK={ok}, SKIP={skip}, 404={nf}, FAIL={fail}, 429={rl}, Elapsed={elapsed_min:.1f} min"
    )
    print(f"\nDone. See log at {LOG_FILE}\n")

if __name__ == "__main__":
    main()
