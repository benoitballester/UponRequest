#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% ------------------------------------------------------------
# 09_plots.py
#
# Robust plotting script that runs from CLI or cell-by-cell in VS Code.
# Plots:
#   P1 (p1): Papers by year (2010–2025): total vs. used in analysis (ok_analysis==1)
#   P2 (p2): Among ok_analysis==1: stacked under_request==0/1 (counts)
#   P3 (p3): Among ok_analysis==1: stacked under_request==0/1 (percent per year)
#   P4 (p4): Among ok_analysis==1: stacked under_request==0/1 (percent per journal; horizontal)
#
# Inputs (auto-detected):
#   2.data/meta_under_request_tagged.tsv  (preferred)
#   2.data/meta_under_request.tsv         (fallback with warning)
#
# Outputs:
#   3.analyses/papers_by_year/
#       papers_by_year.tsv
#       papers_by_year.png
#       papers_by_year.pdf
#   3.analyses/under_request_among_ok_analysis/
#       under_request_among_ok_analysis.tsv
#       under_request_among_ok_analysis.png
#       under_request_among_ok_analysis.pdf
#   3.analyses/under_request_among_ok_analysis_pct/
#       under_request_among_ok_analysis_pct.tsv
#       under_request_among_ok_analysis_pct.png
#       under_request_among_ok_analysis_pct.pdf
#   3.analyses/under_request_among_ok_analysis_pct_by_journal/
#       under_request_among_ok_analysis_pct_by_journal.tsv
#       under_request_among_ok_analysis_pct_by_journal.png
#       under_request_among_ok_analysis_pct_by_journal.pdf
# ---------------------------------------------------------------

# %% Imports
from pathlib import Path
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# %% Constants
YEAR_MIN, YEAR_MAX = 2010, 2025
FIGSIZE = (10, 6)
DPI = 300

# Try to use Arial; silently fall back if not available
try:
    matplotlib.rcParams["font.family"] = "Arial"
except Exception:
    pass


# %% Utilities
def find_project_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]
    cursor = Path.cwd().resolve()
    for _ in range(5):
        if (cursor / "1.scripts").exists() and (cursor / "2.data").exists():
            return cursor
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    return Path.cwd().resolve()


def resolve_input_file(project_root: Path, override: str | None = None) -> Path:
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p

    tagged = project_root / "2.data" / "meta_under_request_tagged.tsv"
    untagged = project_root / "2.data" / "meta_under_request.tsv"

    if tagged.exists():
        print(f"Reading tagged input: {tagged}")
        return tagged
    if untagged.exists():
        print(f"Warning: using untagged input: {untagged}\n"
              f"Run 1.scripts/05_tag_under_request_compliance.py to generate the tagged table.")
        return untagged

    raise FileNotFoundError(
        "Neither 2.data/meta_under_request_tagged.tsv nor 2.data/meta_under_request.tsv was found.\n"
        "If you need the tagged table, run: python 1.scripts/05_tag_under_request_compliance.py"
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_df(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)
    if "year" not in df.columns:
        raise ValueError("Expected a 'year' column in the input TSV.")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    ok_col = "ok_analysis" if "ok_analysis" in df.columns else ("analysis_ok" if "analysis_ok" in df.columns else None)
    if ok_col is None:
        df["ok_analysis_num"] = 0
    else:
        df["ok_analysis_num"] = pd.to_numeric(df[ok_col], errors="coerce").fillna(0).astype(int)

    if "under_request" in df.columns:
        df["under_request_num"] = pd.to_numeric(df["under_request"], errors="coerce").fillna(0).astype(int)
    else:
        df["under_request_num"] = 0

    # clip years for year-based plots
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    return df


# %% Plot 1: Papers by year (total vs ok_analysis==1)
def plot_papers_by_year(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = out_dir / "papers_by_year"
    ensure_dir(out_dir)

    total = df.groupby("year").size().rename("total")
    ok1 = df[df["ok_analysis_num"] == 1].groupby("year").size().rename("ok_analysis_1")

    years = pd.Series(range(YEAR_MIN, YEAR_MAX + 1), name="year")
    summary = pd.DataFrame({"year": years})
    summary = summary.merge(total.reset_index(), on="year", how="left")
    summary = summary.merge(ok1.reset_index(), on="year", how="left")

    summary["total"] = summary["total"].fillna(0).astype(int)
    summary["ok_analysis_1"] = summary["ok_analysis_1"].fillna(0).astype(int)
    summary["ok_analysis_0"] = (summary["total"] - summary["ok_analysis_1"]).clip(lower=0).astype(int)
    summary["ok_share"] = np.where(summary["total"] > 0,
                                   summary["ok_analysis_1"] / summary["total"], np.nan)

    summary.to_csv(out_dir / "papers_by_year.tsv", sep="\t", index=False)

    plt.figure(figsize=FIGSIZE)
    x = summary["year"].values
    base = summary["ok_analysis_0"].values
    top = summary["ok_analysis_1"].values
    plt.bar(x, base, label="Not used (ok_analysis=0)")
    plt.bar(x, top, bottom=base, label="Used (ok_analysis=1)")
    plt.xlabel("Year")
    plt.ylabel("Number of papers")
    plt.title("Papers by Year (2010–2025)\nSubset used in analysis highlighted")
    plt.xticks(x, rotation=90)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "papers_by_year.png", dpi=DPI)
    plt.savefig(out_dir / "papers_by_year.pdf")
    plt.close()
    print(f"Wrote: {out_dir / 'papers_by_year.tsv'}")
    print(f"Wrote: {out_dir / 'papers_by_year.png'}")
    print(f"Wrote: {out_dir / 'papers_by_year.pdf'}")


# %% Plot 2: Among ok_analysis==1, stacked under_request==0/1 (counts)
def plot_under_request_among_ok(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = out_dir / "under_request_among_ok_analysis"
    ensure_dir(out_dir)

    df_ok = df[df["ok_analysis_num"] == 1].copy()
    ok1_total = df_ok.groupby("year").size().rename("ok_analysis_1_total")
    ur1_total = df_ok[df_ok["under_request_num"] == 1].groupby("year").size().rename("under_request_1_total")

    years = pd.Series(range(YEAR_MIN, YEAR_MAX + 1), name="year")
    summary = pd.DataFrame({"year": years})
    summary = summary.merge(ok1_total.reset_index(), on="year", how="left")
    summary = summary.merge(ur1_total.reset_index(), on="year", how="left")

    summary["ok_analysis_1_total"] = summary["ok_analysis_1_total"].fillna(0).astype(int)
    summary["under_request_1_total"] = summary["under_request_1_total"].fillna(0).astype(int)
    summary["ok_analysis_1_not_ur"] = (summary["ok_analysis_1_total"] - summary["under_request_1_total"]).clip(lower=0).astype(int)
    summary["ur_share"] = np.where(summary["ok_analysis_1_total"] > 0,
                                   summary["under_request_1_total"] / summary["ok_analysis_1_total"], np.nan)

    summary.to_csv(out_dir / "under_request_among_ok_analysis.tsv", sep="\t", index=False)

    plt.figure(figsize=FIGSIZE)
    x = summary["year"].values
    base = summary["ok_analysis_1_not_ur"].values
    top = summary["under_request_1_total"].values
    plt.bar(x, base, label="ok_analysis==1 & under_request==0")
    plt.bar(x, top, bottom=base, label="ok_analysis==1 & under_request==1")
    plt.xlabel("Year")
    plt.ylabel("Number of papers")
    plt.title("‘Upon request’ among papers used in analysis (ok_analysis==1)")
    plt.xticks(x, rotation=90)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "under_request_among_ok_analysis.png", dpi=DPI)
    plt.savefig(out_dir / "under_request_among_ok_analysis.pdf")
    plt.close()
    print(f"Wrote: {out_dir / 'under_request_among_ok_analysis.tsv'}")
    print(f"Wrote: {out_dir / 'under_request_among_ok_analysis.png'}")
    print(f"Wrote: {out_dir / 'under_request_among_ok_analysis.pdf'}")


# %% Plot 3: Among ok_analysis==1, stacked under_request==0/1 (percent per year)
def plot_under_request_among_ok_pct(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = out_dir / "under_request_among_ok_analysis_pct"
    ensure_dir(out_dir)

    df_ok = df[df["ok_analysis_num"] == 1].copy()
    ok1_total = df_ok.groupby("year").size().rename("ok_analysis_1_total")
    ur1_total = df_ok[df_ok["under_request_num"] == 1].groupby("year").size().rename("under_request_1_total")

    years = pd.Series(range(YEAR_MIN, YEAR_MAX + 1), name="year")
    summary = pd.DataFrame({"year": years})
    summary = summary.merge(ok1_total.reset_index(), on="year", how="left")
    summary = summary.merge(ur1_total.reset_index(), on="year", how="left")

    summary["ok_analysis_1_total"] = summary["ok_analysis_1_total"].fillna(0).astype(int)
    summary["under_request_1_total"] = summary["under_request_1_total"].fillna(0).astype(int)

    summary["ur_share"] = np.where(
        summary["ok_analysis_1_total"] > 0,
        summary["under_request_1_total"] / summary["ok_analysis_1_total"],
        np.nan
    )
    summary["not_ur_share"] = np.where(
        summary["ok_analysis_1_total"] > 0,
        1.0 - summary["ur_share"],
        np.nan
    )

    summary["ur_pct"] = 100.0 * summary["ur_share"]
    summary["not_ur_pct"] = 100.0 * summary["not_ur_share"]

    summary.to_csv(out_dir / "under_request_among_ok_analysis_pct.tsv", sep="\t", index=False)

    plt.figure(figsize=FIGSIZE)
    x = summary["year"].values
    base = summary["not_ur_pct"].values
    top = summary["ur_pct"].values
    plt.bar(x, base, label="ok_analysis==1 & under_request==0 (%)")
    plt.bar(x, top, bottom=base, label="ok_analysis==1 & under_request==1 (%)")
    plt.xlabel("Year")
    plt.ylabel("Percent of ok_analysis==1")
    plt.title("‘Upon request’ among papers used in analysis (ok_analysis==1) — Percent per year")
    plt.xticks(x, rotation=90)
    plt.ylim(0, 100)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "under_request_among_ok_analysis_pct.png", dpi=DPI)
    plt.savefig(out_dir / "under_request_among_ok_analysis_pct.pdf")
    plt.close()
    print(f"Wrote: {out_dir / 'under_request_among_ok_analysis_pct.tsv'}")
    print(f"Wrote: {out_dir / 'under_request_among_ok_analysis_pct.png'}")
    print(f"Wrote: {out_dir / 'under_request_among_ok_analysis_pct.pdf'}")


# %% Plot 4: Among ok_analysis==1, stacked under_request==0/1 (percent per journal; horizontal bars)
def plot_under_request_among_ok_pct_by_journal(df: pd.DataFrame, out_dir: Path) -> None:
    """
    For each journal, compute percentages among ok_analysis==1:
      - not_ur_share = (ok_analysis==1 & under_request==0) / (ok_analysis==1 total)
      - ur_share     = (ok_analysis==1 & under_request==1) / (ok_analysis==1 total)
    Plot horizontal stacked bars (journals on y-axis), sorted by ur_pct (desc).
    """
    out_dir = out_dir / "under_request_among_ok_analysis_pct_by_journal"
    ensure_dir(out_dir)

    if "journal" not in df.columns:
        raise ValueError("Expected a 'journal' column in the input TSV.")

    df_ok = df[df["ok_analysis_num"] == 1].copy()

    ok1_total = df_ok.groupby("journal", dropna=False).size().rename("ok_analysis_1_total")
    ur1_total = df_ok[df_ok["under_request_num"] == 1].groupby("journal", dropna=False).size().rename("under_request_1_total")

    summary = pd.concat([ok1_total, ur1_total], axis=1).fillna(0)
    summary["ok_analysis_1_total"] = summary["ok_analysis_1_total"].astype(int)
    summary["under_request_1_total"] = summary["under_request_1_total"].astype(int)

    summary["ur_share"] = np.where(
        summary["ok_analysis_1_total"] > 0,
        summary["under_request_1_total"] / summary["ok_analysis_1_total"],
        np.nan
    )
    summary["not_ur_share"] = np.where(
        summary["ok_analysis_1_total"] > 0,
        1.0 - summary["ur_share"],
        np.nan
    )

    summary["ur_pct"] = 100.0 * summary["ur_share"]
    summary["not_ur_pct"] = 100.0 * summary["not_ur_share"]

    # Sort journals by ur_pct descending; drop NaN totals if any
    summary = summary.sort_values(by=["ur_pct", "ok_analysis_1_total"], ascending=[False, False])

    # Save TSV
    summary.reset_index().to_csv(out_dir / "under_request_among_ok_analysis_pct_by_journal.tsv",
                                 sep="\t", index=False)

    # Plot horizontal stacked bars
    plt.figure(figsize=(10, max(6, 0.35 * len(summary))))  # auto-height for many journals
    y = np.arange(len(summary))
    base = summary["not_ur_pct"].values
    top = summary["ur_pct"].values
    labels = summary.index.astype(str).tolist()

    plt.barh(y, base, label="ok_analysis==1 & under_request==0 (%)")
    plt.barh(y, top, left=base, label="ok_analysis==1 & under_request==1 (%)")
    plt.yticks(y, labels)
    plt.xlabel("Percent of ok_analysis==1")
    plt.ylabel("Journal")
    plt.title("‘Upon request’ among papers used in analysis (ok_analysis==1) — Percent by journal")
    plt.xlim(0, 100)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "under_request_among_ok_analysis_pct_by_journal.png", dpi=DPI)
    plt.savefig(out_dir / "under_request_among_ok_analysis_pct_by_journal.pdf")
    plt.close()
    print(f"Wrote: {out_dir / 'under_request_among_ok_analysis_pct_by_journal.tsv'}")
    print(f"Wrote: {out_dir / 'under_request_among_ok_analysis_pct_by_journal.png'}")
    print(f"Wrote: {out_dir / 'under_request_among_ok_analysis_pct_by_journal.pdf'}")


# %% Main (CLI)
def main():
    parser = argparse.ArgumentParser(description="Generate summary plots from meta_under_request tables.")
    parser.add_argument("--input", type=str, default=None,
                        help="Optional path to input TSV. Defaults to 2.data/meta_under_request_tagged.tsv (or fallback).")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Optional base output directory. Defaults to 3.analyses/")
    parser.add_argument("--which", type=str, default="all", choices=["all", "p1", "p2", "p3", "p4"],
                        help="Which plot to produce: p1, p2, p3, p4, or all.")
    args = parser.parse_args()

    project_root = find_project_root()
    input_path = resolve_input_file(project_root, args.input)

    out_base = Path(args.outdir).resolve() if args.outdir else (project_root / "3.analyses")
    ensure_dir(out_base)

    df = load_df(input_path)

    if args.which in ("p1", "all"):
        plot_papers_by_year(df, out_base)
    if args.which in ("p2", "all"):
        plot_under_request_among_ok(df, out_base)
    if args.which in ("p3", "all"):
        plot_under_request_among_ok_pct(df, out_base)
    if args.which in ("p4", "all"):
        plot_under_request_among_ok_pct_by_journal(df, out_base)


# %% Entry point for CLI, while still allowing cell-by-cell execution in VS Code
if __name__ == "__main__":
    main()

# %% Example cell usage in VS Code
# project_root = find_project_root()
# input_path = resolve_input_file(project_root)
# df = load_df(input_path)
# plot_papers_by_year(df, project_root / "3.analyses")
# plot_under_request_among_ok(df, project_root / "3.analyses")
# plot_under_request_among_ok_pct(df, project_root / "3.analyses")
# plot_under_request_among_ok_pct_by_journal(df, project_root / "3.analyses")
