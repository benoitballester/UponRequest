#!/usr/bin/env python3
"""
08f_parse_open_science.py

Scan PMC JATS XML files to quantify Open Science support signals and append
results to the corpus TSV. This version is fully config driven (YAML) and
adds flexible text extraction modes and concurrent processing.

OVERVIEW
--------
For each PMCID in the input corpus, the script:

  1. Loads the corresponding JATS XML file from --xml-dir (named "{PMCID}.xml").
  2. Converts the XML to a single text string, either by:
       - serializing the full XML tree ("xml" mode, default), or
       - concatenating element text with itertext() ("itertext" mode).
  3. Applies a set of configurable regex patterns to detect Open Science
     "support signals", grouped into several buckets:
       - Data sharing (public repositories and accessions)
       - Controlled access repositories (dbGaP, EGA, PHS / EGA-like accessions)
       - Code availability (GitHub, GitLab, Bitbucket, CodeOcean, etc.)
       - Reproducible environments and workflows (Docker, Conda, Nextflow,
         Snakemake, CWL/WDL, notebooks, etc.)
       - Protocol sharing (protocols.io and related resources)
       - Identifier hygiene (RRIDs, ORCIDs, dataset DOIs)
       - Licensing (Creative Commons and common open-source licenses)
       - Figure "Source Data" mentions and Key Resources Table presence
  4. Aggregates these signals into a scalar Open Science Support Index (OSSI)
     score using weights from YAML, then assigns an OSSI tier:
       - Gold, Silver, Bronze, or None.

All regexes, repository lists, weights, and penalties live in a YAML config
file (default: 08f_parse_open_science.yaml) so you can tweak behaviour without
modifying this script.

SPECIAL RULE: "UPON REQUEST" PENALTY
------------------------------------
The script can adjust the OSSI score based on how "data available upon request"
is used, leveraging annotations from the upstream pipeline:

  - Input columns:
      under_request (0/1)  - global flag for any "under request" mention (upon request).
      genuine      (0/1)   - whether the UR statement is considered genuine
                             (e.g. patient data, huge MD trajectories, plasmids).

  - Penalty rule:
      If under_request == 1 AND there is at least one valid accession AND
      genuine == 0, then a penalty is applied to the final OSSI score:

          score = max(0, score - penalties["ur_with_accession_and_not_genuine"])

      If genuine == 1, no penalty is applied and the paper keeps full credit
      for its accessions.

This allows to distinguish:
  - Papers that genuinely cannot share some data openly (but still support OS, eg legal reason),
  - From papers that use "upon request" in place of proper data sharing despite
    having accessions and infrastructure.

INPUT / OUTPUT
--------------
Input TSV (default: "meta_under_request_tagged.tsv"):
  - Must contain a "pmcid" column (case-insensitive alias accepted).
  - May optionally contain:
      - "under_request" (0/1), produced by 08_tag_under_request_compliance.py.
      - "genuine"      (0/1), same provenance as above.

Output TSV (default: "meta_under_request_tagged_open_science.tsv"):
  - Contains all input columns plus the following appended features:

      parse_ok                       (0/1)
      data_public_repo               (0/1)
      data_controlled_access         (0/1)
      code_available                 (0/1)
      repro_env                      (0/1)
      protocol_shared                (0/1)
      key_resources_table            (0/1)
      source_data_present            (0/1)
      open_license_present           (0/1)
      rrid_present                   (0/1)
      orcid_present                  (0/1)
      dataset_doi_present            (0/1)
      data_avail_section_present     (0/1)

      n_repo_types                   (int)
      n_accessions                   (int)
      n_source_data_mentions         (int)
      n_rrids                        (int)
      n_orcids                       (int)
      n_dataset_dois                 (int)

      repos_found                    (pipe-separated string)
      accessions_found               (pipe-separated string)
      code_urls                      (pipe-separated string)
      dataset_dois                   (pipe-separated string)
      rrids                          (pipe-separated string)
      licenses_found                 (pipe-separated string)

      ossi_score                     (int)
      ossi_tier                      {"Gold", "Silver", "Bronze", "None"}

      scan_scope                     {"xml", "itertext"}
        - mirrors the chosen --text-mode, for provenance.

USAGE EXAMPLES
--------------
Typical full XML scan (serialized XML string):

  python 08f_parse_open_science.py \
    --in 2.data/meta_under_request_tagged.tsv \
    --out 2.data/meta_under_request_tagged_open_science.tsv \
    --xml-dir 3.xml \
    --config 1.scripts/08f_parse_open_science.yaml \
    --workers 10 \
    --executor process \
    --chunksize 200 \
    --text-mode xml \
    --progress-every 1000

Faster "itertext" scan (rather than xml scan) (concatenated element text, often less noisy):

  python 08f_parse_open_science.py \
    --in 2.data/meta_under_request_tagged.tsv \
    --out 2.data/meta_under_request_tagged_open_science.tsv \
    --xml-dir 3.xml \
    --config 1.scripts/08f_parse_open_science.yaml \
    --workers 10 \
    --executor process \
    --chunksize 200 \
    --text-mode itertext \
    --progress-every 1000

CONFIG (YAML)
-------------
The YAML file controls nearly everything:

  - repo_patterns: list of {name, regex} to recognise repository types.
  - accession_patterns: regex list to extract accessions (SRR, GSE, GSM, etc.).
  - code_host_regex:     detect code hosting services (GitHub, GitLab, etc.).
  - repro_env_regex:     detect reproducible environments or workflows.
  - protocol_regex:      detect protocol sharing resources.
  - license_regex:       detect open licenses (CC, MIT, BSD, Apache, GPL, etc.).
  - rrid_regex:          detect RRIDs.
  - orcid_regex:         detect ORCIDs.
  - doi_regex:           generic DOI detection.
  - dataset_doi_hint_regex:
                         detect DOIs in contexts suggesting dataset DOIs.
  - source_data_regex:   detect "Source Data"/"Underlying data" phrases.
  - key_resources_table_regex:
                         detect Key Resources Table.
  - sec_type_regex:      detect data/code availability sections.
  - under_request_regex: detect "available upon request" style language.

  - license_tags:        XML tag names to treat as license elements.
  - controlled_repo_types:
                         repository types counted as controlled access (dbGaP, EGA).
  - dataset_doi_infer_from_repos:
                         repo types that strongly imply dataset DOIs
                         (for example, Zenodo, Figshare).

  - score_weights:       dict mapping signal names to integer weights, used to
                         compute ossi_score.
  - penalties:           dict mapping penalty names to integer values. This
                         includes "ur_with_accession_and_not_genuine" for the
                         special UR penalty.

DEPENDENCIES
------------
  mamba/conda install -y pandas lxml pyyaml

PERFORMANCE NOTES
-----------------
  - Uses concurrent.futures with either a ThreadPoolExecutor or a
    ProcessPoolExecutor (chosen via --executor).
  - --workers controls the level of concurrency; --chunksize controls how many
    rows are batched per executor.map call.
  - The XML parser uses recover=True and huge_tree=True to tolerate mild
    malformations in PMC XML.
  - Full-text search is applied to a single text representation of the XML
    (serialized XML or itertext), which keeps the implementation 
    relatively fast.


    
───────────────────────────────────────────────────────────────────────────────
USAGE
───────────────────────────────────────────────────────────────────────────────
# Classic full XML scan => this takes 4h to run (--workers 10  --executor process )
python 1.scripts/08f_parse_open_science.py \
  --in 2.data/meta_under_request_tagged.tsv \
  --out 2.data/meta_under_request_tagged_open_science.tsv \
  --xml-dir 3.xml \
  --config 1.scripts/08f_parse_open_science.yaml \
  --workers 10 --executor process \
  --chunksize 200 \
  --text-mode xml \
  --progress-every 1000

# Faster run  (use concatenated element text via itertext (usually faster and less noisy)) 
python 1.scripts/08f_parse_open_science.py \
  --in 2.data/meta_under_request_tagged.tsv \
  --out 2.data/meta_under_request_tagged_open_science.tsv \
  --xml-dir 3.xml \
  --config 1.scripts/08f_parse_open_science.yaml \
  --workers 10 --executor process \
  --chunksize 200 \
  --text-mode itertext \
  --progress-every 1000

  
"""

from __future__ import annotations

import argparse
import dataclasses as dc
from functools import partial
from pathlib import Path
import os
import re
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

try:
    from lxml import etree
except Exception:
    print("[FATAL] Please install lxml (e.g., mamba/conda install lxml)", file=sys.stderr)
    raise

try:
    import pandas as pd
except Exception:
    print("[FATAL] Please install pandas (e.g., mamba/conda install pandas)", file=sys.stderr)
    raise

try:
    import yaml
except Exception:
    print("[FATAL] Please install pyyaml (e.g., mamba/conda install pyyaml)", file=sys.stderr)
    raise


# ---------------------------
# Config load
# ---------------------------

def _compile(pattern: str, flags=re.I):
    return re.compile(pattern, flags)


class Config:
    def __init__(self, d: dict):
        # Patterns
        self.repo_patterns: List[Tuple[str, re.Pattern]] = [
            (item["name"], _compile(item["regex"])) for item in d.get("repo_patterns", [])
        ]
        self.accession_patterns: List[re.Pattern] = [
            _compile(rx) for rx in d.get("accession_patterns", [])
        ]
        self.code_host_re: re.Pattern = _compile(d.get("code_host_regex", r"(?:github|gitlab|bitbucket)\.com/[^\s)]+"))
        self.repro_env_re: re.Pattern = _compile(
            d.get(
                "repro_env_regex",
                r"environment\.ya?ml|requirements\.txt|Dockerfile|Singularity|nextflow\.config|Snakefile|\.cwl|\.wdl|\.ipynb",
            )
        )
        self.protocol_re: re.Pattern = _compile(
            d.get(
                "protocol_regex",
                r"protocols\.io|Nature\s+Protocol\s+Exchange|Registered\s+protocol|preregistration|PROSPERO",
            )
        )
        self.license_re: re.Pattern = _compile(
            d.get(
                "license_regex",
                r"Creative\s+Commons|\bCC\s?BY(?:\-SA|\-NC|\-ND)?\b|\bCC0\b|MIT\s+License|BSD\s+(?:2\-Clause|3\-Clause)|Apache\s+License|GPL\-?v?\d",
            )
        )
        self.rrid_re: re.Pattern = _compile(d.get("rrid_regex", r"RRID:[A-Z_]+:\S+"))
        self.orcid_re: re.Pattern = _compile(
            d.get("orcid_regex", r"\b(?:ORCID:)?\s?\d{4}\-\d{4}\-\d{4}\-\d{3}[\dX]\b")
        )
        self.doi_re: re.Pattern = _compile(d.get("doi_regex", r"\b10\.\d{4,9}/\S+\b"))
        # special flags: DOTALL for dataset_doi_hint_re
        self.dataset_doi_hint_re: re.Pattern = re.compile(
            d.get("dataset_doi_hint_regex", r"(DataCite|dataset|data\s+set).*?10\.\d{4,9}/\S+"),
            flags=re.I | re.S,
        )
        self.source_data_re: re.Pattern = _compile(
            d.get("source_data_regex", r"Source\s+Data|Underlying\s+data|Raw\s+data\s+for\s+Fig")
        )
        self.krt_re: re.Pattern = _compile(d.get("key_resources_table_regex", r"Key\s+Resources\s+Table"))
        self.sec_type_re: re.Pattern = _compile(
            d.get("sec_type_regex", r"\b(data-availability|availability|code-availability)\b")
        )
        self.ur_re: re.Pattern = _compile(
            d.get("under_request_regex", r"available\s+upon\s+request|upon\s+request|available\s+on\s+request")
        )
        self.license_tags: Set[str] = set(d.get("license_tags", ["license", "license-p", "license-ref", "permissions"]))
        self.controlled_repo_types: Set[str] = set(d.get("controlled_repo_types", ["dbGaP", "EGA"]))
        self.dataset_doi_infer_from_repos: Set[str] = set(d.get("dataset_doi_infer_from_repos", ["Zenodo", "Figshare"]))

        # Scoring
        self.score_weights: Dict[str, int] = d.get("score_weights", {})
        self.penalties: Dict[str, int] = d.get("penalties", {})


def load_config(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    return Config(d)


# ---------------------------
# Data structures
# ---------------------------

@dc.dataclass
class OSFeatures:
    pmcid: str
    parse_ok: int = 1

    # Signals (booleans)
    data_public_repo: int = 0
    data_controlled_access: int = 0
    code_available: int = 0
    repro_env: int = 0
    protocol_shared: int = 0
    key_resources_table: int = 0
    source_data_present: int = 0
    open_license_present: int = 0
    rrid_present: int = 0
    orcid_present: int = 0
    dataset_doi_present: int = 0
    data_avail_section_present: int = 0

    # Counts
    n_repo_types: int = 0
    n_accessions: int = 0
    n_source_data_mentions: int = 0
    n_rrids: int = 0
    n_orcids: int = 0
    n_dataset_dois: int = 0

    # Lists (pipe-separated in output)
    repos_found: str = ""
    accessions_found: str = ""
    code_urls: str = ""
    dataset_dois: str = ""
    rrids: str = ""
    licenses_found: str = ""

    # Derived
    ossi_score: int = 0
    ossi_tier: str = ""

    # Provenance / meta
    scan_scope: str = "full-xml"


# ---------------------------
# Utilities
# ---------------------------

def xml_to_text(root: etree._Element, mode: str) -> str:
    """Return the string to be scanned by regex, depending on text mode."""
    if mode == "itertext":
        try:
            return "".join(root.itertext())
        except Exception:
            return ""
    # default 'xml' mode: scan serialized XML (backward-compatible)
    try:
        return etree.tostring(root, encoding="unicode", with_tail=False)
    except Exception:
        return ""


def get_xml_root(xml_path: Path) -> Optional[etree._Element]:
    try:
        parser = etree.XMLParser(recover=True, huge_tree=True)
        with xml_path.open("rb") as fh:
            return etree.parse(fh, parser=parser).getroot()
    except Exception:
        return None


# ---------------------------
# Extraction helpers (config-driven)
# ---------------------------

def find_repo_types_and_accessions(text: str, cfg: Config) -> Tuple[Set[str], Set[str]]:
    repo_types: Set[str] = set()
    for name, rx in cfg.repo_patterns:
        if rx.search(text):
            repo_types.add(name)

    accessions: Set[str] = set()
    for rx in cfg.accession_patterns:
        accessions.update(m.group(0).upper() for m in rx.finditer(text))

    return repo_types, accessions


def find_code_urls(text: str, cfg: Config) -> Set[str]:
    return set(m.group(0) for m in cfg.code_host_re.finditer(text))


def any_re(pattern: re.Pattern, text: str) -> bool:
    return bool(pattern.search(text))


def findall_re(pattern: re.Pattern, text: str) -> List[str]:
    return [m.group(0) for m in pattern.finditer(text)]


def extract_features(pmcid: str, xml_dir: Path, cfg: Config, text_mode: str) -> OSFeatures:
    feat = OSFeatures(pmcid=pmcid)
    xml_path = xml_dir / f"{pmcid}.xml"

    if not xml_path.exists():
        feat.parse_ok = 0
        return feat

    root = get_xml_root(xml_path)
    if root is None:
        feat.parse_ok = 0
        return feat

    full_text = xml_to_text(root, text_mode)

    # Section-type presence (data/code availability)
    feat.data_avail_section_present = 1 if any_re(cfg.sec_type_re, full_text) else 0

    # License tags or textual license mentions
    license_nodes = root.xpath("//*[local-name() = 'license' or local-name() = 'license-p' or local-name() = 'license-ref' or local-name() = 'permissions']")
    textual_license = any_re(cfg.license_re, full_text)
    feat.open_license_present = 1 if (license_nodes or textual_license) else 0

    # Source Data mentions
    src_hits = findall_re(cfg.source_data_re, full_text)
    feat.source_data_present = 1 if src_hits else 0
    feat.n_source_data_mentions = len(src_hits)

    # Key Resources Table
    feat.key_resources_table = 1 if any_re(cfg.krt_re, full_text) else 0

    # Protocols
    feat.protocol_shared = 1 if any_re(cfg.protocol_re, full_text) else 0

    # Code repos
    code_urls = find_code_urls(full_text, cfg)
    feat.code_available = 1 if code_urls else 0
    feat.code_urls = "|".join(sorted(code_urls))

    # Reproducible env/workflow
    feat.repro_env = 1 if any_re(cfg.repro_env_re, full_text) else 0

    # Repositories and accessions
    repo_types, accessions = find_repo_types_and_accessions(full_text, cfg)
    feat.n_repo_types = len(repo_types)
    feat.repos_found = "|".join(sorted(repo_types))

    feat.n_accessions = len(accessions)
    feat.accessions_found = "|".join(sorted(accessions))

    # Controlled-access
    feat.data_controlled_access = 1 if (cfg.controlled_repo_types & repo_types or any(a.startswith("PHS") or a.startswith("EGA") for a in accessions)) else 0

    # Public data repo if any non-controlled repository present
    public_repo_present = bool(repo_types - cfg.controlled_repo_types)
    feat.data_public_repo = 1 if public_repo_present else 0

    # Identifier hygiene
    rrids = findall_re(cfg.rrid_re, full_text)
    feat.rrid_present = 1 if rrids else 0
    feat.n_rrids = len(rrids)
    feat.rrids = "|".join(sorted(set(rrids)))

    orcids = findall_re(cfg.orcid_re, full_text)
    feat.orcid_present = 1 if orcids else 0
    feat.n_orcids = len(orcids)

    dois = findall_re(cfg.doi_re, full_text)
    dataset_doi_hits = findall_re(cfg.dataset_doi_hint_re, full_text)
    dataset_doi_present = 1 if dataset_doi_hits or (cfg.dataset_doi_infer_from_repos & repo_types) else 0
    feat.dataset_doi_present = dataset_doi_present
    feat.dataset_dois = "|".join(sorted(set(dois)))

    # Licenses found (textual) for provenance
    licenses_found = findall_re(cfg.license_re, full_text)
    feat.licenses_found = "|".join(sorted(set(licenses_found)))

    feat.scan_scope = "itertext" if text_mode == "itertext" else "full-xml"
    return feat


def compute_score(feat: OSFeatures, under_request: Optional[int], genuine: Optional[int], cfg: Config) -> Tuple[int, str]:
    W = cfg.score_weights
    P = cfg.penalties
    score = 0

    if feat.data_public_repo:
        score += W.get("data_public_repo", 4)
        if feat.n_repo_types >= 2:
            score += W.get("data_multi_repo_bonus", 1)

    if feat.data_controlled_access:
        score += W.get("data_controlled_access", 2)

    if feat.code_available:
        score += W.get("code_available", 2)

    if feat.repro_env:
        score += W.get("repro_env", 1)

    if feat.protocol_shared:
        score += W.get("protocol_shared", 1)

    identifier_hygiene = 1 if (feat.rrid_present or feat.orcid_present or feat.dataset_doi_present) else 0
    if identifier_hygiene:
        score += W.get("identifier_hygiene", 1)

    if feat.open_license_present:
        score += W.get("open_license_present", 1)

    if feat.source_data_present:
        score += W.get("source_data_present", 1)

    # Special rule: if "upon request" co-occurs with a valid accession
    if under_request == 1 and feat.n_accessions > 0:
        if genuine == 0:
            score = max(0, score - P.get("ur_with_accession_and_not_genuine", 2))
        # if genuine==1: no penalty

    # Tiering
    if score >= 8:
        tier = "Gold"
    elif score >= 5:
        tier = "Silver"
    elif score >= 2:
        tier = "Bronze"
    else:
        tier = "None"

    return score, tier


# ---------------------------
# Orchestration
# ---------------------------

def process_row(row: "pd.Series", xml_dir: Path, cfg: Config, text_mode: str) -> Dict[str, object]:
    pmcid = str(row.get("pmcid", "")).strip()
    if not pmcid:
        return {"pmcid": pmcid, "_skip": 1}

    feat = extract_features(pmcid, xml_dir, cfg, text_mode=text_mode)

    def to01(x):
        try:
            v = int(x)
            return 1 if v == 1 else 0
        except Exception:
            return 0

    under_request = to01(row.get("under_request", 0))
    genuine = to01(row.get("genuine", 0))

    score, tier = compute_score(feat, under_request, genuine, cfg)
    feat.ossi_score = score
    feat.ossi_tier = tier

    return dc.asdict(feat)


def run(args: argparse.Namespace) -> None:
    in_path = Path(args.in_tsv)
    out_path = Path(args.out_tsv)
    xml_dir = Path(args.xml_dir)
    cfg_path = Path(args.config)

    if not in_path.exists():
        print(f"[FATAL] Input TSV not found: {in_path}", file=sys.stderr)
        sys.exit(2)
    if not xml_dir.exists():
        print(f"[FATAL] XML directory not found: {xml_dir}", file=sys.stderr)
        sys.exit(2)
    if not cfg_path.exists():
        print(f"[FATAL] Config YAML not found: {cfg_path}", file=sys.stderr)
        sys.exit(2)

    cfg = load_config(cfg_path)

    df = pd.read_csv(in_path, sep="\t", dtype=str).fillna("")
    if "pmcid" not in df.columns:
        alt_cols = [c for c in df.columns if c.lower() == "pmcid"]
        if alt_cols:
            df.rename(columns={alt_cols[0]: "pmcid"}, inplace=True)
        else:
            raise SystemExit("Input TSV must contain a 'pmcid' column.")

    rows = [r[1] for r in df.iterrows()]

    results: List[Dict[str, object]] = []

    # Executor selection
    if args.executor == "process":
        from concurrent.futures import ProcessPoolExecutor as Exec
    else:
        from concurrent.futures import ThreadPoolExecutor as Exec

    worker_fn = partial(process_row, xml_dir=xml_dir, cfg=cfg, text_mode=args.text_mode)

    start = time.time()
    total = len(rows)

    with Exec(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(worker_fn, rows, chunksize=args.chunksize), start=1):
            results.append(res)
            if i % args.progress_every == 0 or i == total:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0.0
                remaining = (total - i) / rate if rate > 0 else float('inf')
                print(f"[{i}/{total}] {rate:.1f} rows/s | ETA ~ {remaining/60:.1f} min", file=sys.stderr)

    res_df = pd.DataFrame(results)
    if "_skip" in res_df.columns:
        res_df = res_df[res_df["_skip"] != 1].drop(columns=["_skip"]).copy()

    merged = df.merge(res_df, on="pmcid", how="left")

    new_cols = [c for c in res_df.columns if c != "pmcid"]
    final_cols = list(df.columns) + [c for c in new_cols if c not in df.columns]
    merged = merged[final_cols]

    merged.to_csv(out_path, sep="\t", index=False)

    n = len(merged)
    gold = (merged.get("ossi_tier", pd.Series([])) == "Gold").sum()
    silver = (merged.get("ossi_tier", pd.Series([])) == "Silver").sum()
    bronze = (merged.get("ossi_tier", pd.Series([])) == "Bronze").sum()
    none = (merged.get("ossi_tier", pd.Series([])) == "None").sum()

    print(f"[DONE] Wrote {n} rows → {out_path}", file=sys.stderr)
    print(f"        Tiers: Gold={gold} Silver={silver} Bronze={bronze} None={none}", file=sys.stderr)


# ---------------------------
# CLI
# ---------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scan PMC XML for Open Science signals and score OSSI (config-driven).")
    p.add_argument("--in", dest="in_tsv", default="meta_under_request_tagged.tsv", help="Input TSV with at least a pmcid column.")
    p.add_argument("--out", dest="out_tsv", default="meta_under_request_tagged_open_science.tsv", help="Output TSV (input + OS features).")
    p.add_argument("--xml-dir", dest="xml_dir", default="3.xml", help="Directory containing PMC XML files named {PMCID}.xml")
    p.add_argument("--config", dest="config", default="08f_parse_open_science.yaml", help="YAML config with regexes, weights, and options.")
    p.add_argument("--workers", dest="workers", type=int, default=os.cpu_count() or 4, help="Number of parallel workers.")
    p.add_argument("--executor", choices=["thread", "process"], default="thread",
                   help="thread = ThreadPoolExecutor (default). process = ProcessPoolExecutor for CPU-bound speedups.")
    p.add_argument("--chunksize", dest="chunksize", type=int, default=50,
                   help="Batch size for executor.map (larger = less overhead, more latency)."
                   )
    p.add_argument("--text-mode", dest="text_mode", choices=["xml", "itertext"], default="xml",
                   help="How to create the searchable string: 'xml' (serialized XML) or 'itertext' (concatenated text)."
                   )
    p.add_argument("--progress-every", dest="progress_every", type=int, default=1000,
                   help="Print progress every N rows (stderr)."
                   )
    return p


def main():
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
