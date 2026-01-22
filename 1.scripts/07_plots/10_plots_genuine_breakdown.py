#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------------------------------
# 10_plots_genuine_breakdown.py
#
# Reproduces P2, P3, P4 with a finer breakdown:
#   Instead of a single stack for (ok_analysis==1 & under_request==1),
#   we split it into:
#       - ok_analysis==1 & under_request==1 & genuine==1
#       - ok_analysis==1 & under_request==1 & genuine==0
#
# Plots generated:
#   G2: Counts per year (2010–2025), stacked:
#         [ok_analysis==1 & under_request==0] (base)
#         [ok_analysis==1 & UR==1 & genuine==0]
#         [ok_analysis==1 & UR==1 & genuine==1]
#
#   G3: Percent per year (within ok_analysis==1), stacked:
#         [% not_ur], [% UR & g==0], [% UR & g==1]
#
#   G4: Percent per journal (within ok_analysis==1), horizontal stacked:
#         [% not_ur], [% UR & g==0], [% UR & g==1]
#
# Inputs (auto-detected):
#   2.data/meta_under_request_tagged.tsv  (preferred)
#   2.data/meta_under_request.tsv         (fallback, but requires 'genuine' column)
#
# Outputs:
#   3.analyses/under_request_genuine_count_by_year/
#       under_request_genuine_count_by_year.tsv/.png/.pdf
#   3.analyses/under_request_genuine_pct_by_year/
#       under_request_genuine_pct_by_year.tsv/.png/.pdf
#   3.analyses/under_request_genuine_pct_by_journal/
#       under_request_genuine_pct_by_journal.tsv/.png/.pdf
# ------------------------------------------------------------

from pathlib import Path
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# -------------------- Config --------------------
YEAR_MIN, YEAR_MAX = 2010, 2025
FIGSIZE = (10, 6)
DPI = 300

# Try to use Arial; silently fall back if not available
try:
    matplotlib.rcParams["font.family"] = "Arial"
except Exception:
    pass


# -------------------- Utils --------------------
def find_project_root() -> Path:
    """Locate repo root whether run from CLI or VS Code cells."""
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
    """Prefer tagged table; fallback to untagged if necessary."""
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
              f"Run 1.scripts/05_tag_under_request_compliance.py to generate the tagged table including 'genuine'.")
        return untagged

    raise FileNotFoundError(
        "Neither 2.data/meta_under_request_tagged.tsv nor 2.data/meta_under_request.tsv was found."
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_df(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)

    # Required columns
    for col in ["year", "under_request", "genuine"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in the input TSV.")

    # Normalize types
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["under_request_num"] = pd.to_numeric(df["under_request"], errors="coerce").fillna(0).astype(int)
    df["genuine_num"] = pd.to_numeric(df["genuine"], errors="coerce").fillna(0).astype(int)

    # ok_analysis / analysis_ok handling
    ok_col = "ok_analysis" if "ok_analysis" in df.columns else ("analysis_ok" if "analysis_ok" in df.columns else None)
    if ok_col is None:
        df["ok_analysis_num"] = 0
    else:
        df["ok_analysis_num"] = pd.to_numeric(df[ok_col], errors="coerce").fillna(0).astype(int)

    # Clip year range for year-based plots
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    return df


# -------------------- G2: Counts per year --------------------
def plot_counts_by_year_genuine(df: pd.DataFrame, out_base: Path) -> None:
    out_dir = out_base / "under_request_genuine_count_by_year"
    ensure_dir(out_dir)

    df_ok = df[df["ok_analysis_num"] == 1].copy()
    # Totals among ok_analysis==1
    ok_total = df_ok.groupby("year").size().rename("ok_analysis_1_total")

    # Breakdowns among ok_analysis==1:
    not_ur = df_ok[df_ok["under_request_num"] == 0].groupby("year").size().rename("not_ur")
    ur_g1 = df_ok[(df_ok["under_request_num"] == 1) & (df_ok["genuine_num"] == 1)].groupby("year").size().rename("ur_genuine_1")
    ur_g0 = df_ok[(df_ok["under_request_num"] == 1) & (df_ok["genuine_num"] == 0)].groupby("year").size().rename("ur_genuine_0")

    years = pd.Series(range(YEAR_MIN, YEAR_MAX + 1), name="year")
    summary = pd.DataFrame({"year": years})
    for s in (ok_total, not_ur, ur_g0, ur_g1):
        summary = summary.merge(s.reset_index(), on="year", how="left")

    # Fill + types
    for c in ["ok_analysis_1_total", "not_ur", "ur_genuine_0", "ur_genuine_1"]:
        summary[c] = summary[c].fillna(0).astype(int)

    # Save TSV
    summary.to_csv(out_dir / "under_request_genuine_count_by_year.tsv", sep="\t", index=False)

    # Plot stacked counts
    plt.figure(figsize=FIGSIZE)
    x = summary["year"].values
    base = summary["not_ur"].values
    mid = summary["ur_genuine_0"].values
    top = summary["ur_genuine_1"].values

    plt.bar(x, base, label="ok_analysis==1 & under_request==0")
    plt.bar(x, mid, bottom=base, label="UR & genuine=0")
    plt.bar(x, top, bottom=base + mid, label="UR & genuine=1")

    plt.xlabel("Year")
    plt.ylabel("Number of papers (ok_analysis==1)")
    plt.title("ok_analysis==1 — Counts by year\n(not_UR vs UR&g=0 vs UR&g=1)")
    plt.xticks(x, rotation=90)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "under_request_genuine_count_by_year.png", dpi=DPI)
    plt.savefig(out_dir / "under_request_genuine_count_by_year.pdf")
    plt.close()

    print(f"Wrote: {out_dir / 'under_request_genuine_count_by_year.tsv'}")
    print(f"Wrote: {out_dir / 'under_request_genuine_count_by_year.png'}")
    print(f"Wrote: {out_dir / 'under_request_genuine_count_by_year.pdf'}")


# -------------------- G3: Percent per year --------------------
def plot_percent_by_year_genuine(df: pd.DataFrame, out_base: Path) -> None:
    out_dir = out_base / "under_request_genuine_pct_by_year"
    ensure_dir(out_dir)

    df_ok = df[df["ok_analysis_num"] == 1].copy()
    ok_total = df_ok.groupby("year").size().rename("ok_analysis_1_total")

    not_ur = df_ok[df_ok["under_request_num"] == 0].groupby("year").size().rename("not_ur")
    ur_g1 = df_ok[(df_ok["under_request_num"] == 1) & (df_ok["genuine_num"] == 1)].groupby("year").size().rename("ur_genuine_1")
    ur_g0 = df_ok[(df_ok["under_request_num"] == 1) & (df_ok["genuine_num"] == 0)].groupby("year").size().rename("ur_genuine_0")

    years = pd.Series(range(YEAR_MIN, YEAR_MAX + 1), name="year")
    summary = pd.DataFrame({"year": years})
    for s in (ok_total, not_ur, ur_g0, ur_g1):
        summary = summary.merge(s.reset_index(), on="year", how="left")

    for c in ["ok_analysis_1_total", "not_ur", "ur_genuine_0", "ur_genuine_1"]:
        summary[c] = summary[c].fillna(0).astype(int)

    # Percentages within ok_analysis==1
    denom = summary["ok_analysis_1_total"].replace(0, np.nan)
    summary["not_ur_pct"] = 100.0 * (summary["not_ur"] / denom)
    summary["ur_g0_pct"] = 100.0 * (summary["ur_genuine_0"] / denom)
    summary["ur_g1_pct"] = 100.0 * (summary["ur_genuine_1"] / denom)

    # Save TSV
    summary.to_csv(out_dir / "under_request_genuine_pct_by_year.tsv", sep="\t", index=False)

    # Plot stacked % (0–100)
    plt.figure(figsize=FIGSIZE)
    x = summary["year"].values
    b = np.nan_to_num(summary["not_ur_pct"].values)
    m = np.nan_to_num(summary["ur_g0_pct"].values)
    t = np.nan_to_num(summary["ur_g1_pct"].values)

    plt.bar(x, b, label="not_UR (%)")
    plt.bar(x, m, bottom=b, label="UR & genuine=0 (%)")
    plt.bar(x, t, bottom=b + m, label="UR & genuine=1 (%)")

    plt.xlabel("Year")
    plt.ylabel("Percent of ok_analysis==1")
    plt.title("ok_analysis==1 — Percent by year\n(not_UR vs UR&g=0 vs UR&g=1)")
    plt.xticks(x, rotation=90)
    plt.ylim(0, 100)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "under_request_genuine_pct_by_year.png", dpi=DPI)
    plt.savefig(out_dir / "under_request_genuine_pct_by_year.pdf")
    plt.close()

    print(f"Wrote: {out_dir / 'under_request_genuine_pct_by_year.tsv'}")
    print(f"Wrote: {out_dir / 'under_request_genuine_pct_by_year.png'}")
    print(f"Wrote: {out_dir / 'under_request_genuine_pct_by_year.pdf'}")


# -------------------- G4: Percent by journal (horizontal) --------------------
def plot_percent_by_journal_genuine(df: pd.DataFrame, out_base: Path) -> None:
    """
    For each journal, compute percentages among ok_analysis==1:
      - not_ur_pct = (ok_analysis==1 & under_request==0) / (ok_analysis==1 total) * 100
      - ur_g0_pct  = (ok_analysis==1 & under_request==1 & genuine==0) / (ok_analysis==1 total) * 100
      - ur_g1_pct  = (ok_analysis==1 & under_request==1 & genuine==1) / (ok_analysis==1 total) * 100
    Sort by (ur_g0_pct + ur_g1_pct) descending. Horizontal stacked bars.
    """
    out_dir = out_base / "under_request_genuine_pct_by_journal"
    ensure_dir(out_dir)

    if "journal" not in df.columns:
        raise ValueError("Expected a 'journal' column in the input TSV.")

    df_ok = df[df["ok_analysis_num"] == 1].copy()

    ok_total = df_ok.groupby("journal", dropna=False).size().rename("ok_analysis_1_total")
    not_ur   = df_ok[df_ok["under_request_num"] == 0] \
                 .groupby("journal", dropna=False).size().rename("not_ur")
    ur_g1    = df_ok[(df_ok["under_request_num"] == 1) & (df_ok["genuine_num"] == 1)] \
                 .groupby("journal", dropna=False).size().rename("ur_genuine_1")
    ur_g0    = df_ok[(df_ok["under_request_num"] == 1) & (df_ok["genuine_num"] == 0)] \
                 .groupby("journal", dropna=False).size().rename("ur_genuine_0")

    summary = pd.concat([ok_total, not_ur, ur_g0, ur_g1], axis=1).fillna(0)
    for c in ["ok_analysis_1_total", "not_ur", "ur_genuine_0", "ur_genuine_1"]:
        summary[c] = summary[c].astype(int)

    denom = summary["ok_analysis_1_total"].replace(0, np.nan)
    summary["not_ur_pct"] = 100.0 * (summary["not_ur"] / denom)
    summary["ur_g0_pct"]  = 100.0 * (summary["ur_genuine_0"] / denom)
    summary["ur_g1_pct"]  = 100.0 * (summary["ur_genuine_1"] / denom)
    summary["total_ur_pct"] = summary["ur_g0_pct"] + summary["ur_g1_pct"]

    # Sort by total UR percentage (desc), tie-breaker = larger journal first
    summary = summary.sort_values(by=["total_ur_pct", "ok_analysis_1_total"], ascending=[False, False])

    # Save TSV (include total_ur_pct)
    summary.reset_index().to_csv(
        out_dir / "under_request_genuine_pct_by_journal.tsv",
        sep="\t", index=False
    )

    # Plot horizontal stacked bars (% that sum to ~100)
    plt.figure(figsize=(10, max(6, 0.35 * len(summary))))
    y = np.arange(len(summary))
    b = np.nan_to_num(summary["not_ur_pct"].values)
    m = np.nan_to_num(summary["ur_g0_pct"].values)
    t = np.nan_to_num(summary["ur_g1_pct"].values)
    labels = summary.index.astype(str).tolist()

    plt.barh(y, b, label="not_UR (%)")
    plt.barh(y, m, left=b, label="UR & genuine=0 (%)")
    plt.barh(y, t, left=b + m, label="UR & genuine=1 (%)")

    plt.yticks(y, labels)
    plt.xlabel("Percent of ok_analysis==1")
    plt.ylabel("Journal")
    plt.title("ok_analysis==1 — Percent by journal\n(sorted by total ‘upon request’ share)")
    plt.xlim(0, 100)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "under_request_genuine_pct_by_journal.png", dpi=DPI)
    plt.savefig(out_dir / "under_request_genuine_pct_by_journal.pdf")
    plt.close()

    print(f"Wrote: {out_dir / 'under_request_genuine_pct_by_journal.tsv'}")
    print(f"Wrote: {out_dir / 'under_request_genuine_pct_by_journal.png'}")
    print(f"Wrote: {out_dir / 'under_request_genuine_pct_by_journal.pdf'}")


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Genuine breakdown plots (counts per year, % per year, % per journal).")
    parser.add_argument("--input", type=str, default=None,
                        help="Optional path to input TSV. Defaults to 2.data/meta_under_request_tagged.tsv (or fallback).")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Optional base output directory. Defaults to 3.analyses/")
    parser.add_argument("--which", type=str, default="all", choices=["all", "g2", "g3", "g4"],
                        help="Which plot: g2 (counts/year), g3 (percent/year), g4 (percent/journal).")
    args = parser.parse_args()

    project_root = find_project_root()
    input_path = resolve_input_file(project_root, args.input)
    out_base = Path(args.outdir).resolve() if args.outdir else (project_root / "3.analyses")
    ensure_dir(out_base)

    df = load_df(input_path)

    if args.which in ("g2", "all"):
        plot_counts_by_year_genuine(df, out_base)
    if args.which in ("g3", "all"):
        plot_percent_by_year_genuine(df, out_base)
    if args.which in ("g4", "all"):
        plot_percent_by_journal_genuine(df, out_base)


if __name__ == "__main__":
    main()

# Example (VS Code cells):
# project_root = find_project_root()
# input_path = resolve_input_file(project_root)
# out_base = project_root / "3.analyses"
# df = load_df(input_path)
# plot_counts_by_year_genuine(df, out_base)
# plot_percent_by_year_genuine(df, out_base)
# plot_percent_by_journal_genuine(df, out_base)
