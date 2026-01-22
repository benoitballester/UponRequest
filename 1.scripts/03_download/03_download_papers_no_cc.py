#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_download_papers_no_cc.py
--------------------------------------------------------------
Download “NO-CC CODE” / “Free in PMC” XMLs via the slower
https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/?format=xml
endpoint.

Description:
  This script handles the restricted subset of articles not
  available through the OA-PMH interface.  It is intentionally
  conservative to respect PMC's rate limits.

Features:
- Reads entries from 2.data/meta_openaccess.tsv where License == "NO-CC CODE"
- Downloads XMLs from ?format=xml
- Polite: single-digit concurrency + long sleeps
- Full resume and logging identical to OA script
- Logs to 3.no_cc_code/download_log.txt
- Graceful Ctrl-C stop and retry for transient errors

Output:
  XMLs saved to 3.no_cc_code/{pmcid}.xml
"""

import os
import sys
import time
import queue
import random
import signal
import threading
import requests
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
META_FILE = Path("2.data/meta_openaccess.tsv")
OUT_DIR   = Path("3.no_cc_code")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE  = OUT_DIR / "download_log.txt"

MAX_WORKERS   = 1          # very conservative
TIMEOUT       = 45
RETRY_LIMIT   = 3
SLEEP_BETWEEN = (5.0, 8.0) # long sleeps to stay far below limits

EMAIL = os.environ.get("NCBI_EMAIL", "NAME@inserm.fr")
HEADERS = {
    "User-Agent": f"PMC-noCC-Downloader (contact: {EMAIL})",
    "Accept": "application/xml"
}
BASE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/?format=xml"

# ------------------------------------------------------------------
# Load metadata and filter
# ------------------------------------------------------------------
print(f"Loading PMCIDs from: {META_FILE}")
meta = pd.read_csv(META_FILE, sep="\t", dtype=str)
meta = meta[meta["License"].fillna("") == "NO-CC CODE"]

pmc_ids = (
    meta["pmcid"]
    .dropna()
    .astype(str)
    .str.replace("^PMC", "", regex=True)
    .tolist()
)

# ------------------------------------------------------------------
# Resume mode
# ------------------------------------------------------------------
completed, failed = set(), set()
if LOG_FILE.exists():
    for line in LOG_FILE.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        status, pmc = parts[0], parts[1]
        if status in {"OK", "404", "SKIP"}:
            completed.add(pmc)
        elif status in {"FAIL", "ERR"}:
            failed.add(pmc)

initial_total = len(pmc_ids)
pmc_ids = [p for p in pmc_ids if p not in completed]
remaining_total = len(pmc_ids)
print(f"Resuming: {remaining_total:,} remaining of {initial_total:,} total.")

# ------------------------------------------------------------------
# Worker class
# ------------------------------------------------------------------
class Downloader(threading.Thread):
    def __init__(self, q, lock, results):
        super().__init__()
        self.q = q
        self.lock = lock
        self.results = results

    def run(self):
        session = requests.Session()
        while True:
            try:
                pmcid = self.q.get(timeout=3)
            except queue.Empty:
                return
            try:
                msg = self.download_one(session, pmcid)
                with self.lock:
                    self.results.append(msg)
                    print(msg)
            except Exception as e:
                with self.lock:
                    self.results.append(f"ERR  {pmcid}: {e}")
            finally:
                self.q.task_done()

    def download_one(self, session, pmcid):
        out_path = OUT_DIR / f"{pmcid}.xml"
        if out_path.exists():
            return f"SKIP {pmcid} (exists)"

        url = BASE_URL.format(pmcid=pmcid)
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                r = session.get(url, headers=HEADERS, timeout=TIMEOUT)
                if r.status_code == 200 and r.text.strip():
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(r.text)
                    return f"OK   {pmcid}"
                elif r.status_code == 404:
                    return f"404  {pmcid}"
                elif r.status_code == 429:
                    wait = random.randint(120, 240)
                    print(f"429  {pmcid} (rate limit hit, backing off {wait}s...)")
                    time.sleep(wait)
                    continue
                else:
                    time.sleep(0.5)
            except Exception as e:
                if attempt == RETRY_LIMIT:
                    return f"FAIL {pmcid}: {e}"
                time.sleep(2.0)
            time.sleep(random.uniform(*SLEEP_BETWEEN))

        return f"FAIL {pmcid}: unexpected no-response"

# ------------------------------------------------------------------
# Prepare queue & launch
# ------------------------------------------------------------------
q = queue.Queue()
for pmcid in failed.union(set(pmc_ids)):
    q.put(pmcid)

results, lock = [], threading.Lock()
workers = [Downloader(q, lock, results) for _ in range(MAX_WORKERS)]

print(f"Starting {MAX_WORKERS} worker thread(s) ...")
start_time = time.time()
for w in workers:
    w.start()

# ------------------------------------------------------------------
# Graceful stop
# ------------------------------------------------------------------
def handle_interrupt(sig, frame):
    print("\nInterrupted. Flushing logs...")
    with open(LOG_FILE, "a") as log:
        for r in results:
            log.write(r + "\n")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

# ------------------------------------------------------------------
# Wait for completion
# ------------------------------------------------------------------
while any(w.is_alive() for w in workers):
    try:
        q.join()
        break
    except KeyboardInterrupt:
        handle_interrupt(None, None)

# ------------------------------------------------------------------
# Summary & logging
# ------------------------------------------------------------------
elapsed = time.time() - start_time
ok_count     = sum(1 for r in results if r.startswith("OK"))
skip_count   = sum(1 for r in results if r.startswith("SKIP"))
fail_count   = sum(1 for r in results if r.startswith("FAIL"))
notfound_cnt = sum(1 for r in results if r.startswith("404"))

print("\n================ SUMMARY ================")
print(f"Elapsed time: {elapsed/60:.1f} min")
print(f"OK:   {ok_count}")
print(f"SKIP: {skip_count}")
print(f"404:  {notfound_cnt}")
print(f"FAIL: {fail_count}")
print(f"Resumed from previous log: {len(completed):,} completed, {len(failed):,} failed.")
print("=========================================")

with open(LOG_FILE, "a") as log:
    for r in results:
        log.write(r + "\n")

print(f"Results logged in: {LOG_FILE}")
print(f"XML files saved to: {OUT_DIR}/")
