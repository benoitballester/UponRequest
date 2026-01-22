#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 09_plots.py  (with policy overlay support)
# Plots:
#   P1 (p1): Papers by year — total vs ok_analysis==1
#   P2 (p2): Among ok_analysis==1 — stacked under_request==0/1 (counts)
#   P3 (p3): Among ok_analysis==1 — stacked under_request==0/1 (percent per year)
#   P4 (p4): Among ok_analysis==1 — stacked under_request==0/1 (percent per journal; horizontal)
# Optional: overlay policy timeline (vertical lines) on P1–P3

from pathlib import Path
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

YEAR_MIN, YEAR_MAX = 2010, 2025
FIGSIZE = (10, 6)
DPI = 300

try:
    matplotlib.rcParams["font.family"] = "Arial"
except Exception:
    pass

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
        print(f"Warning: using untagged input: {untagged}")
        return untagged
    raise FileNotFoundError("No meta_under_request TSV found in 2.data/")

def resolve_timeline_file(project_root: Path, override: str | None = None) -> Path | None:
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        return p if p.exists() else None
    default = project_root / "2.data" / "policy_timeline.tsv"
    return default if default.exists() else None

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_df(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)
    if "year" not in df.columns:
        raise ValueError("Expected 'year' column.")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    ok_col = "ok_analysis" if "ok_analysis" in df.columns else ("analysis_ok" if "analysis_ok" in df.columns else None)
    df["ok_analysis_num"] = pd.to_numeric(df[ok_col], errors="coerce").fillna(0).astype(int) if ok_col else 0
    df["under_request_num"] = pd.to_numeric(df.get("under_request", 0), errors="coerce").fillna(0).astype(int)
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    return df

def load_timeline(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    try:
        tl = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        tl["year"] = pd.to_numeric(tl["year"], errors="coerce")
        tl = tl.dropna(subset=["year"])
        return tl
    except Exception:
        return None

def overlay_policy_lines(ax, timeline_df: pd.DataFrame | None):
    """Draw vertical lines for policy years on a year-based plot."""
    if timeline_df is None or timeline_df.empty:
        return
    ymin, ymax = ax.get_ylim()
    for _, row in timeline_df.iterrows():
        yr = row["year"]
        label = f'{row.get("publisher","")}: {row.get("policy","")}'
        ax.axvline(x=yr, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(yr + 0.05, ymax * 0.98, str(int(yr)),
                rotation=90, va="top", ha="left", fontsize=8, color="gray")

# ---- P1
def plot_papers_by_year(df: pd.DataFrame, out_dir: Path, timeline_df=None, overlay=False) -> None:
    out_dir = out_dir / "papers_by_year"
    ensure_dir(out_dir)
    total = df.groupby("year").size().rename("total")
    ok1 = df[df["ok_analysis_num"] == 1].groupby("year").size().rename("ok_analysis_1")
    years = pd.Series(range(YEAR_MIN, YEAR_MAX + 1), name="year")
    summary = pd.DataFrame({"year": years}).merge(total.reset_index(), on="year", how="left") \
                                          .merge(ok1.reset_index(), on="year", how="left")
    summary["total"] = summary["total"].fillna(0).astype(int)
    summary["ok_analysis_1"] = summary["ok_analysis_1"].fillna(0).astype(int)
    summary["ok_analysis_0"] = (summary["total"] - summary["ok_analysis_1"]).clip(lower=0).astype(int)
    summary["ok_share"] = np.where(summary["total"] > 0, summary["ok_analysis_1"] / summary["total"], np.nan)
    summary.to_csv(out_dir / "papers_by_year2.tsv", sep="\t", index=False)

    plt.figure(figsize=FIGSIZE)
    x = summary["year"].values
    base = summary["ok_analysis_0"].values
    top = summary["ok_analysis_1"].values
    ax = plt.gca()
    ax.bar(x, base, label="Not used (ok_analysis=0)")
    ax.bar(x, top, bottom=base, label="Used (ok_analysis=1)")
    ax.set_xlabel("Year"); ax.set_ylabel("Number of papers")
    ax.set_title("Papers by Year (2010–2025)\nSubset used in analysis highlighted")
    ax.set_xticks(x); ax.set_xticklabels(x, rotation=90)
    if overlay: overlay_policy_lines(ax, timeline_df)
    ax.legend(frameon=False); plt.tight_layout()
    plt.savefig(out_dir / "papers_by_year2.png", dpi=DPI)
    plt.savefig(out_dir / "papers_by_year2.pdf"); plt.close()

# ---- P2
def plot_under_request_among_ok(df: pd.DataFrame, out_dir: Path, timeline_df=None, overlay=False) -> None:
    out_dir = out_dir / "under_request_among_ok_analysis"; ensure_dir(out_dir)
    df_ok = df[df["ok_analysis_num"] == 1].copy()
    ok1_total = df_ok.groupby("year").size().rename("ok_analysis_1_total")
    ur1_total = df_ok[df_ok["under_request_num"] == 1].groupby("year").size().rename("under_request_1_total")
    years = pd.Series(range(YEAR_MIN, YEAR_MAX + 1), name="year")
    summary = pd.DataFrame({"year": years}).merge(ok1_total.reset_index(), on="year", how="left") \
                                          .merge(ur1_total.reset_index(), on="year", how="left")
    summary["ok_analysis_1_total"] = summary["ok_analysis_1_total"].fillna(0).astype(int)
    summary["under_request_1_total"] = summary["under_request_1_total"].fillna(0).astype(int)
    summary["ok_analysis_1_not_ur"] = (summary["ok_analysis_1_total"] - summary["under_request_1_total"]).clip(lower=0).astype(int)
    summary["ur_share"] = np.where(summary["ok_analysis_1_total"] > 0,
                                   summary["under_request_1_total"] / summary["ok_analysis_1_total"], np.nan)
    summary.to_csv(out_dir / "under_request_among_ok_analysis2.tsv", sep="\t", index=False)

    plt.figure(figsize=FIGSIZE); ax = plt.gca()
    x = summary["year"].values
    base = summary["ok_analysis_1_not_ur"].values
    top = summary["under_request_1_total"].values
    ax.bar(x, base, label="ok_analysis==1 & under_request==0")
    ax.bar(x, top, bottom=base, label="ok_analysis==1 & under_request==1")
    ax.set_xlabel("Year"); ax.set_ylabel("Number of papers")
    ax.set_title("‘Upon request’ among papers used in analysis (ok_analysis==1)")
    ax.set_xticks(x); ax.set_xticklabels(x, rotation=90)
    if overlay: overlay_policy_lines(ax, timeline_df)
    ax.legend(frameon=False); plt.tight_layout()
    plt.savefig(out_dir / "under_request_among_ok_analysis2.png", dpi=DPI)
    plt.savefig(out_dir / "under_request_among_ok_analysis2.pdf"); plt.close()

# ---- P3
def plot_under_request_among_ok_pct(df: pd.DataFrame, out_dir: Path, timeline_df=None, overlay=False) -> None:
    out_dir = out_dir / "under_request_among_ok_analysis_pct"; ensure_dir(out_dir)
    df_ok = df[df["ok_analysis_num"] == 1].copy()
    ok1_total = df_ok.groupby("year").size().rename("ok_analysis_1_total")
    ur1_total = df_ok[df_ok["under_request_num"] == 1].groupby("year").size().rename("under_request_1_total")
    years = pd.Series(range(YEAR_MIN, YEAR_MAX + 1), name="year")
    summary = pd.DataFrame({"year": years}).merge(ok1_total.reset_index(), on="year", how="left") \
                                          .merge(ur1_total.reset_index(), on="year", how="left")
    summary["ok_analysis_1_total"] = summary["ok_analysis_1_total"].fillna(0).astype(int)
    summary["under_request_1_total"] = summary["under_request_1_total"].fillna(0).astype(int)
    summary["ur_share"] = np.where(summary["ok_analysis_1_total"] > 0,
                                   summary["under_request_1_total"] / summary["ok_analysis_1_total"], np.nan)
    summary["not_ur_share"] = np.where(summary["ok_analysis_1_total"] > 0, 1.0 - summary["ur_share"], np.nan)
    summary["ur_pct"] = 100.0 * summary["ur_share"]; summary["not_ur_pct"] = 100.0 * summary["not_ur_share"]
    summary.to_csv(out_dir / "under_request_among_ok_analysis_pct2.tsv", sep="\t", index=False)

    plt.figure(figsize=FIGSIZE); ax = plt.gca()
    x = summary["year"].values
    base = summary["not_ur_pct"].values
    top = summary["ur_pct"].values
    ax.bar(x, base, label="ok_analysis==1 & under_request==0 (%)")
    ax.bar(x, top, bottom=base, label="ok_analysis==1 & under_request==1 (%)")
    ax.set_xlabel("Year"); ax.set_ylabel("Percent of ok_analysis==1"); ax.set_ylim(0, 100)
    ax.set_title("‘Upon request’ among papers used in analysis (ok_analysis==1) — Percent per year")
    ax.set_xticks(x); ax.set_xticklabels(x, rotation=90)
    if overlay: overlay_policy_lines(ax, timeline_df)
    ax.legend(frameon=False); plt.tight_layout()
    plt.savefig(out_dir / "under_request_among_ok_analysis_pct2.png", dpi=DPI)
    plt.savefig(out_dir / "under_request_among_ok_analysis_pct2.pdf"); plt.close()

# ---- P4 (unchanged; journal-level, no overlay)
def plot_under_request_among_ok_pct_by_journal(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = out_dir / "under_request_among_ok_analysis_pct_by_journal"
    ensure_dir(out_dir)
    if "journal" not in df.columns:
        raise ValueError("Expected 'journal' column.")
    df_ok = df[df["ok_analysis_num"] == 1].copy()
    ok1_total = df_ok.groupby("journal", dropna=False).size().rename("ok_analysis_1_total")
    ur1_total = df_ok[df_ok["under_request_num"] == 1].groupby("journal", dropna=False).size().rename("under_request_1_total")
    summary = pd.concat([ok1_total, ur1_total], axis=1).fillna(0).astype(int)
    denom = summary["ok_analysis_1_total"].replace(0, np.nan)
    summary["ur_pct"] = 100.0 * (summary["under_request_1_total"] / denom)
    summary["not_ur_pct"] = 100.0 * (1.0 - summary["under_request_1_total"] / denom)
    summary = summary.sort_values(by=["ur_pct", "ok_analysis_1_total"], ascending=[False, False])
    summary.reset_index().to_csv(out_dir / "under_request_among_ok_analysis_pct_by_journal2.tsv", sep="\t", index=False)

    plt.figure(figsize=(12, max(6, 0.35 * len(summary))))
    y = np.arange(len(summary)); labels = summary.index.astype(str).tolist()
    b = np.nan_to_num(summary["not_ur_pct"].values); t = np.nan_to_num(summary["ur_pct"].values)
    plt.barh(y, b, label="ok_analysis==1 & under_request==0 (%)")
    plt.barh(y, t, left=b, label="ok_analysis==1 & under_request==1 (%)")
    plt.yticks(y, labels); plt.xlabel("Percent of ok_analysis==1"); plt.ylabel("Journal")
    plt.title("‘Upon request’ among papers used in analysis — Percent by journal")
    plt.xlim(0, 100); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(out_dir / "under_request_among_ok_analysis_pct_by_journal2.png", dpi=DPI)
    plt.savefig(out_dir / "under_request_among_ok_analysis_pct_by_journal2.pdf"); plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate summary plots from meta_under_request tables (with optional policy overlays).")
    parser.add_argument("--input", type=str, default=None, help="Path to input TSV (default: 2.data/meta_under_request_tagged.tsv or fallback).")
    parser.add_argument("--outdir", type=str, default=None, help="Base output dir (default: 3.analyses/)")
    parser.add_argument("--which", type=str, default="all", choices=["all", "p1", "p2", "p3", "p4"], help="Which plot(s) to run")
    parser.add_argument("--timeline", type=str, default="2.data/policy_timeline.tsv", help="Path to policy_timeline.tsv (default: 2.data/policy_timeline.tsv if present)")
    parser.add_argument("--overlay", action="store_true", help="Add vertical policy lines to year-based plots (p1–p3)")
    args = parser.parse_args()

    project_root = find_project_root()
    input_path = resolve_input_file(project_root, args.input)
    out_base = Path(args.outdir).resolve() if args.outdir else (project_root / "3.analyses")
    ensure_dir(out_base)

    df = load_df(input_path)
    timeline_path = resolve_timeline_file(project_root, args.timeline)
    timeline_df = load_timeline(timeline_path)

    if args.which in ("p1", "all"): plot_papers_by_year(df, out_base, timeline_df, args.overlay)
    if args.which in ("p2", "all"): plot_under_request_among_ok(df, out_base, timeline_df, args.overlay)
    if args.which in ("p3", "all"): plot_under_request_among_ok_pct(df, out_base, timeline_df, args.overlay)
    if args.which in ("p4", "all"): plot_under_request_among_ok_pct_by_journal(df, out_base)

if __name__ == "__main__":
    main()
