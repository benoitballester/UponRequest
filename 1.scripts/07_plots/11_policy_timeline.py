#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
11_policy_timeline.py
This writes a small, citable policy timeline for overlaying on year-based plots.

Output:
  2.data/policy_timeline.tsv
Columns:
  year, publisher, journal_scope, policy, notes, url
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "2.data"
OUT_TSV = DATA_DIR / "policy_timeline.tsv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

rows = [
    # PLOS
    dict(
        year=2014,
        publisher="PLOS",
        journal_scope="All PLOS journals",
        policy="Data Availability Statements required",
        notes="Portfolio-wide policy starting March 2014; data should be publicly available with limited exceptions.",
        url="https://journals.plos.org/plosone/s/data-availability",
    ),
    # Cell Press
    dict(
        year=2016,
        publisher="Cell Press",
        journal_scope="Cell + life-science titles",
        policy="STAR Methods introduced",
        notes="Structured, Transparent, Accessible Reporting; standardised methods reporting rolled out across Cell Press.",
        url="https://www.cell.com/star-methods",
    ),
    # Nature portfolio (rollout)
    dict(
        year=2016,
        publisher="Nature Portfolio",
        journal_scope="Nature (flagship) + initial pilot journals",
        policy="Mandatory Data Availability Statements (pilot/launch)",
        notes="DAS piloted at 5 journals (Mar 2016) and adopted at Nature (Sept 2016).",
        url="",  # add specific announcement link you prefer
    ),
    dict(
        year=2017,
        publisher="Nature Portfolio",
        journal_scope="Nature Research journals",
        policy="Mandatory Data Availability Statements (rollout)",
        notes="Rollout across the portfolio after 2016 pilot/adoption.",
        url="",  # add portfolio policy link you prefer
    ),
    # AAAS / Science (long-standing)
    dict(
        year=2010,
        publisher="AAAS",
        journal_scope="Science (policy long-standing)",
        policy="Data & materials availability required",
        notes="Policy predates 2016–2017; specify article-type nuances as needed.",
        url="https://www.science.org/content/page/science-journals-editorial-policies",
    ),
    # Oxford / NAR examples (domain-specific deposition)
    #dict(
    #    year=1992,
    #    publisher="Oxford Univ. Press",
    #    journal_scope="Nucleic Acids Research",
    #    policy="Mandatory deposition of structural data (early exemplar)",
    #    notes="Strict domain deposition norms existed long before modern DAS.",
    #    url="",  # add archival/policy link if you want
    #),
]

df = pd.DataFrame(rows, columns=["year", "publisher", "journal_scope", "policy", "notes", "url"])
df.to_csv(OUT_TSV, sep="\t", index=False)
print(f"Wrote: {OUT_TSV}")
