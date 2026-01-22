#!/usr/bin/env python3
# --------------------------------------------------
# 08_tag_under_request_compliance.py
# --------------------------------------------------
# PURPOSE
# -------
# Post-process the parser output (meta_under_request.tsv) to classify each
# "upon request" sentence according to compliance rules.
#
# This script:
#   1) Assigns a "domain" (data, materials, code, other) to each under_request
#      sentence.
#   2) Assigns a "reason" describing why access is restricted or how it is
#      offered (privacy/ethics, controlled_access_repo, MTA/proprietary,
#      materials_not_text, code_shareable, data_should_be_open, size_only, none).
#   3) Assigns a "genuine" flag (1/0) indicating whether the under_request
#      formulation is acceptable in context (genuine constraints or materials)
#      versus potentially non sharing.
#   4) Renames the original "restricted" from the XML parser to
#      "restricted_initial_parse" and recomputes a new "restricted" flag as:
#           restricted = 1  if  restricted_initial_parse == 1 and genuine == 0
#           restricted = 0  otherwise
#
# INPUT
# -----
#   2.data/meta_under_request.tsv
#   (This is the TSV produced by the XML parser. It must contain at least:
#    - pmcid
#    - under_request  (0/1)
#    - matching       (exact phrase or "xml_error")
#    - phrase         (full sentence containing the match)
#    - section        (section title, if available)
#    - journal, year  (if available)
#    - restricted     (0/1, from 07_parse_under_request_with_sections_restricted.py)
#   All other columns are preserved.)
#
# OUTPUT (ENRICHED)
# -----------------
#   2.data/meta_under_request_tagged.tsv
#       Same as input, plus:
#         - domain
#         - reason
#         - genuine
#         - restricted_initial_parse
#         - restricted  (new, derived as above)
#
#   3.analyses/under_request_tagging/summary_by_journal.tsv
#   3.analyses/under_request_tagging/summary_by_year.tsv      (if year present)
#   3.analyses/under_request_tagging/summary_by_section.tsv   (if section present)
#
# --------------------------------------------------

import os
import re
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

IN_FILE  = "2.data/meta_under_request.tsv"            # Parser output (TSV)
OUT_FILE = "2.data/meta_under_request_tagged.tsv"     # Enriched table
OUT_DIR  = "3.analyses/under_request_tagging"         # Folder for summaries

SEP_IN   = "\t"
SEP_OUT  = "\t"

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# Regex cue sets
# --------------------------------------------------
# Domain cues

DATA_CUES = re.compile(
    r"\b("
    r"data|dataset|datasets|raw data|data set|data sets|"
    r"individual level|participant level|patient level|"
    r"clinical data|clinical information|patient data|"
    r"phenotype|genotype|"
    r"sequence data|sequencing data|reads?|fastq|bam|cram|vcf|"
    r"count matrix|counts matrix|expression matrix|"
    r"rna-?seq|chip-?seq|atac-?seq|wgs|whole genome sequencing|"
    r"geo|sra|ena|ega|dbgap|arrayexpress"
    r")\b",
    re.IGNORECASE,
)

MATERIALS_CUES = re.compile(
    r"\b("
    r"plasmid|plasmids|vector|vectors|construct|constructs|"
    r"strain|strains|cell line|cell lines|"
    r"mutant|mutants|knockout|transgenic|transgenic lines?|"
    r"mouse line|mouse lines|"
    r"primer|primers|probe|probes|oligo|oligos|oligonucleotide|oligonucleotides|"
    r"blueprint|blueprints|"
    r"arabidopsis mutants?|transgenic arabidopsis lines?"
    r")\b",
    re.IGNORECASE,
)

CODE_CUES = re.compile(
    r"\b("
    r"code|codes|source code|software|package|packages|"
    r"script|scripts|pipeline|pipelines"
    r")\b",
    re.IGNORECASE,
)

DATA_AVAIL_SECTION = re.compile(
    r"(data availability|availability of data|data and materials)",
    re.IGNORECASE,
)

# Reason cue sets

PRIVACY_ETHICS_CUES = re.compile(
    r"(patient(?:s)?\b|patient level|patient level data|"
    r"clinical data|clinical information|hospital\b|"
    r"human genetic resources?|human subjects?|human participants?|"
    r"anonymity\b|anonymous\b|anonymized\b|anonymised\b|"
    r"de[- ]?identified|pseudo[- ]?anonym)",
    re.IGNORECASE,
)

CONTROLLED_ACCESS_CUES = re.compile(
    r"(dbgap\b|ega\b|controlled access\b|controlled-access\b|"
    r"data access committee|data access request|DAC\b)",
    re.IGNORECASE,
)

THIRD_PARTY_CUES = re.compile(
    r"(third party|third-party|data provider|data providers|"
    r"provider\b|providers\b|data source|data sources|"
    r"consortium|consortia|governance|regulator|regulators|"
    r"ethics committee|ethical committee|institutional review board|IRB\b)",
    re.IGNORECASE,
)

MTA_PROPRIETARY_CUES = re.compile(
    r"(material transfer agreement|MTA\b|"
    r"data sharing agreement|project agreement|"
    r"non[- ]disclosure agreement|nondisclosure agreement|NDA\b|"
    r"licence conditions|license conditions|licence\b|license\b|"
    r"licensing\b|proprietary\b|commercially sensitive)",
    re.IGNORECASE,
)

EXTREME_SIZE_CUES = re.compile(
    r"(PetaLibrary|external hard drive|external hard drives|"
    r"hard drives?|shipping of external hard drives?|tape library)",
    re.IGNORECASE,
)

# Open source licenses (for code) that are not genuine constraints

OPEN_LICENSE_CUES = re.compile(
    r"(mit license|gnu general public license|gpl\b|apache license|apache 2\.0|"
    r"bsd license|bsd-2-clause|bsd-3-clause)",
    re.IGNORECASE,
)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def detect_domain(phrase: str, section: str) -> str:
    """
    Decide whether a sentence concerns data, materials, code, or other.

    Priority:
        1) If data cues present, treat as "data".
        2) Else if materials cues present, treat as "materials".
        3) Else if code cues present, treat as "code".
        4) Else, if in a Data Availability section, treat as "data".
        5) Otherwise "other".
    """
    text = phrase or ""
    sec = section or ""

    data = bool(DATA_CUES.search(text))
    mats = bool(MATERIALS_CUES.search(text))
    code = bool(CODE_CUES.search(text))

    if not (data or mats or code) and DATA_AVAIL_SECTION.search(sec):
        data = True

    if data:
        return "data"
    if mats:
        return "materials"
    if code:
        return "code"
    return "other"


def classify_reason(domain: str, phrase: str) -> str:
    """
    Assign a single reason label for the given domain and phrase.

    For data:
        privacy/ethics > controlled_access_repo > third_party_governance >
        MTA/proprietary > size_only > data_should_be_open

    For materials:
        MTA/proprietary > privacy/ethics > controlled_access_repo >
        third_party_governance > size_only > materials_not_text

    For code:
        MTA/proprietary (except open-source licenses) > privacy/ethics >
        controlled_access_repo > third_party_governance > size_only >
        code_shareable

    For other:
        same priority as data, with fallback "none".
    """
    text = phrase or ""

    reasons = []

    if PRIVACY_ETHICS_CUES.search(text):
        reasons.append("privacy/ethics")
    if CONTROLLED_ACCESS_CUES.search(text):
        reasons.append("controlled_access_repo")
    if THIRD_PARTY_CUES.search(text):
        reasons.append("third_party_governance")
    if MTA_PROPRIETARY_CUES.search(text):
        reasons.append("MTA/proprietary")
    if EXTREME_SIZE_CUES.search(text):
        reasons.append("size_only")

    # For code, do not treat standard open source licenses as MTA/proprietary
    if domain == "code" and "MTA/proprietary" in reasons and OPEN_LICENSE_CUES.search(text):
        reasons = [r for r in reasons if r != "MTA/proprietary"]

    if domain == "data":
        for label in [
            "privacy/ethics",
            "controlled_access_repo",
            "third_party_governance",
            "MTA/proprietary",
            "size_only",
        ]:
            if label in reasons:
                return label
        return "data_should_be_open"

    if domain == "materials":
        if "MTA/proprietary" in reasons:
            return "MTA/proprietary"
        for label in [
            "privacy/ethics",
            "controlled_access_repo",
            "third_party_governance",
            "size_only",
        ]:
            if label in reasons:
                return label
        return "materials_not_text"

    if domain == "code":
        if "MTA/proprietary" in reasons:
            return "MTA/proprietary"
        for label in [
            "privacy/ethics",
            "controlled_access_repo",
            "third_party_governance",
            "size_only",
        ]:
            if label in reasons:
                return label
        return "code_shareable"

    # domain == "other"
    if reasons:
        for label in [
            "privacy/ethics",
            "controlled_access_repo",
            "third_party_governance",
            "MTA/proprietary",
            "size_only",
        ]:
            if label in reasons:
                return label
    return "none"


def decide_genuine(domain: str, reason: str, phrase: str) -> bool:
    """
    Decide whether an under_request sentence is genuine (True) or not (False).

    Data:
      - True for privacy/ethics, controlled_access_repo, third_party_governance,
        MTA/proprietary, size_only.
      - For data_should_be_open, only True when there is an explicit
        "article + supplementary AND/ALSO from corresponding author" pattern.

    Materials:
      - True for materials_not_text, MTA/proprietary, privacy/ethics,
        controlled_access_repo, third_party_governance, size_only.

    Code:
      - If code is available "upon (reasonable) request" (in any form),
        treat as non genuine, even if a repository URL is also present.
      - Otherwise, True if there is a code repository URL or a strong
        reason (privacy/ethics, controlled access, third party, MTA,
        size_only).
      - Plain code_shareable with no repository is non genuine.

    Other:
      - True only for strong reasons and size_only.
    """
    text = (phrase or "").lower()

    if domain == "data":
        if reason in {
            "privacy/ethics",
            "controlled_access_repo",
            "third_party_governance",
            "MTA/proprietary",
            "size_only",
        }:
            return True

        if reason == "data_should_be_open":
            has_article_supp = bool(
                re.search(
                    r"within the (article|paper)\s+and\s+(its\s+)?supplementary\s+"
                    r"(information|materials?|files?)",
                    text,
                )
            )
            has_also = (
                " and also " in text
                or " is also available" in text
                or " are also available" in text
            )
            has_and_from_corr = " and from the corresponding author" in text
            has_or_request = bool(
                re.search(
                    r"\bor\b[^.]{0,80}\bcorresponding author\b",
                    text,
                )
            )

            if has_article_supp and (has_also or has_and_from_corr) and not has_or_request:
                return True
            return False

        return False

    if domain == "materials":
        if reason in {
            "materials_not_text",
            "MTA/proprietary",
            "privacy/ethics",
            "controlled_access_repo",
            "third_party_governance",
            "size_only",
        }:
            return True
        return False

    if domain == "code":
        # Any code that is (fully or partly) "available upon (reasonable) request"
        # is treated as non genuine, even if some code or tools are on GitHub/GitLab.
        if re.search(r"upon\s+\w*\s*request", text):
            return False

        has_repo = (
            "github.com" in text or
            "gitlab.com" in text or
            "bitbucket.org" in text
        )
        if has_repo:
            return True

        if reason in {
            "privacy/ethics",
            "controlled_access_repo",
            "third_party_governance",
            "MTA/proprietary",
            "size_only",
        }:
            return True

        return False

    # domain == "other"
    if reason in {
        "privacy/ethics",
        "controlled_access_repo",
        "third_party_governance",
        "MTA/proprietary",
        "size_only",
    }:
        return True
    return False


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    df = pd.read_csv(IN_FILE, sep=SEP_IN, dtype=str, keep_default_na=False)

    if "under_request" not in df.columns:
        raise ValueError("Column 'under_request' not found in input.")

    df["under_request_num"] = (
        pd.to_numeric(df["under_request"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    if "phrase" not in df.columns:
        df["phrase"] = ""
    if "section" not in df.columns:
        df["section"] = ""
    if "matching" not in df.columns:
        df["matching"] = ""

    # Handle restricted from parser
    if "restricted_initial_parse" in df.columns:
        pass
    elif "restricted" in df.columns:
        df.rename(columns={"restricted": "restricted_initial_parse"}, inplace=True)
    else:
        df["restricted_initial_parse"] = 0

    # Initialize or reset classification columns
    if "domain" not in df.columns:
        df["domain"] = "none"
    if "reason" not in df.columns:
        df["reason"] = "none"
    if "genuine" not in df.columns:
        df["genuine"] = ""

    mask = (df["under_request_num"] == 1) & (
        df["matching"].str.lower() != "xml_error"
    )

    for idx, row in df[mask].iterrows():
        phrase = row.get("phrase", "")
        section = row.get("section", "")

        domain = detect_domain(phrase, section)
        reason = classify_reason(domain, phrase)
        genuine = decide_genuine(domain, reason, phrase)

        df.at[idx, "domain"] = domain
        df.at[idx, "reason"] = reason
        df.at[idx, "genuine"] = int(bool(genuine))

    df.drop(columns=["under_request_num"], inplace=True)

    # Normalize restricted_initial_parse and compute new restricted
    df["restricted_initial_parse"] = (
        pd.to_numeric(df["restricted_initial_parse"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    genuine_int = (
        pd.to_numeric(df["genuine"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df["restricted"] = 0
    restricted_mask = (
        (df["restricted_initial_parse"] == 1)
        & (genuine_int == 1)
    )
    df.loc[restricted_mask, "restricted"] = 1



    df.to_csv(OUT_FILE, sep=SEP_OUT, index=False)

    def build_summary(group_keys, out_name):
        cols_present = [k for k in group_keys if k in df.columns]
        if not cols_present:
            return
        g = df.groupby(cols_present, dropna=False)

        s = g.agg(
            total_rows=("pmcid", "count"),
            under_request_1=(
                "under_request",
                lambda x: (pd.to_numeric(x, errors="coerce") == 1).sum(),
            ),
            non_compliant_data=(
                "pmcid",
                lambda ix: (
                    (df.loc[ix.index, "under_request"].astype(str) == "1")
                    & (df.loc[ix.index, "domain"] == "data")
                    & (df.loc[ix.index, "genuine"].astype(str) == "0")
                ).sum(),
            ),
            genuine_constraints=(
                "pmcid",
                lambda ix: (
                    (df.loc[ix.index, "under_request"].astype(str) == "1")
                    & (df.loc[ix.index, "genuine"].astype(str) == "1")
                ).sum(),
            ),
        ).reset_index()

        s.to_csv(os.path.join(OUT_DIR, out_name), sep=SEP_OUT, index=False)

    build_summary(["journal"], "summary_by_journal.tsv")
    if "year" in df.columns:
        build_summary(["year"], "summary_by_year.tsv")
    if "section" in df.columns:
        build_summary(["section"], "summary_by_section.tsv")

    print(f"Enriched table written to: {OUT_FILE}")
    print(f"Summaries written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
