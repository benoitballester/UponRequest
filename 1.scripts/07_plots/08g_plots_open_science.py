#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08g_plot_open_science.py
Create figures showing the evolution of OSSI (Open Science Support Index) between 2010 and 2025.

Plot design (recap)
    Median + IQR line: robust central trend with spread; annotated with yearly N.
    Yearly boxplots: distributional view (no outliers) to see compression/shift.
    Tier composition: stacked 100% bars for Gold/Silver/Bronze/None per year.


Defaults:
  --in      2.data/meta_under_request_tagged_open_science.tsv
  --outdir  4.analyses/open_science

OUTPUT (all written in --outdir)
  - ossi_score_median_iqr_by_year.png / .pdf
  - ossi_score_median_iqr_by_year.tsv    (year, n, q25, ossi_score_median, q75)
  - ossi_score_boxplot_by_year.png / .pdf
  - ossi_score_boxplot_by_year.tsv       (long table: year, ossi_score)
  - ossi_tier_stacked_pct_by_year.png / .pdf
  - ossi_tier_stacked_pct_by_year.tsv    (wide % table: year, Gold, Silver, Bronze, None)
  - ossi_summary.pdf                     (multi-page PDF with all plots)



python 01.script/08g_plots_open_science.py  \
  --in 2.data/meta_under_request_tagged_open_science.tsv  \
  --outdir 3.analyses/figures_ossi

/Users/benoit/miniforge3/envs/bio/bin/python /Users/benoit/work/under_request/1.scripts/08g_plots_open_science.py   --in 2.data/meta_under_request_tagged_open_science.tsv    --outdir 3.analyses/figures_ossi

it will produce (in 3.analyses/figures_ossi/)
    ossi_score_median_iqr_by_year.png and .pdf
    ossi_score_median_iqr_by_year.tsv (year, n, q25, ossi_score_median, q75)
    ossi_score_boxplot_by_year.png and .pdf
    ossi_score_boxplot_by_year.tsv (long format: year, ossi_score)
    ossi_tier_stacked_pct_by_year.png and .pdf
    ossi_tier_stacked_pct_by_year.tsv (wide table: year, Gold, Silver, Bronze, None)
    ossi_summary.pdf (multi-page PDF assembling all plots)

"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

def infer_year_column(df):
    # Prefer 'year' or 'pub_year'; otherwise parse from 'pubdate'/'date'/'publication_date'
    for c in df.columns:
        if c.lower() in ("year","pub_year"):
            return c
    for c in df.columns:
        if c.lower() in ("pubdate","date","publication_date"):
            ser = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            if ser.notna().any():
                df["_year"] = ser.dt.year
                return "_year"
    # Fallback: try regex-based year extraction from any column
    for c in df.columns:
        ser = df[c].astype(str).str.extract(r"(\d{4})", expand=False)
        if ser.notna().any():
            yy = pd.to_numeric(ser, errors="coerce")
            if yy.notna().any():
                df["_year"] = yy
                return "_year"
    raise SystemExit("Could not infer a 'year' column. Add a 'year' or 'pub_year' column to the TSV.")

def ensure_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def summarize_score_by_year(df, year_col, min_year=2010, max_year=2025):
    d = df[[year_col,"ossi_score"]].copy()
    d["ossi_score"] = ensure_numeric(d["ossi_score"])
    d = d.dropna()
    d = d[(d[year_col] >= min_year) & (d[year_col] <= max_year)]
    grp = d.groupby(year_col)["ossi_score"]
    summary = grp.agg(n="size", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
    summary = summary.reset_index().rename(columns={year_col:"year"}).sort_values("year")
    return summary, d  # also return the filtered long data used for boxplot

def tier_percent_by_year(df, year_col, min_year=2010, max_year=2025):
    d = df[[year_col, "ossi_tier"]].copy()
    d = d[(d[year_col] >= min_year) & (d[year_col] <= max_year)]
    d["ossi_tier"] = d["ossi_tier"].fillna("None")
    order = ["Gold","Silver","Bronze","None"]
    d.loc[~d["ossi_tier"].isin(order), "ossi_tier"] = "None"
    ct = d.groupby([year_col, "ossi_tier"]).size().unstack(fill_value=0)
    for t in order:
        if t not in ct.columns:
            ct[t] = 0
    ct = ct[order]
    pct = (ct.T / ct.sum(axis=1)).T * 100.0
    pct = pct.reset_index().rename(columns={year_col:"year"}).sort_values("year")
    return pct

def plot_median_iqr(outdir, summary):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(summary["year"], summary["median"], marker="o")
    ax.fill_between(summary["year"], summary["q25"], summary["q75"], alpha=0.2, label="IQR")
    ax.set_xlabel("Year")
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_ylabel("OSSI score")
    ax.set_title("OSSI score — median and IQR by year")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    # annotate N
    for _, row in summary.iterrows():
        ax.text(row["year"], row["median"], f" n={int(row['n'])}", fontsize=8, ha="left", va="bottom")
    ax.legend(loc="best", frameon=False)
    png = Path(outdir)/"ossi_score_median_iqr_by_year.png"
    pdf = Path(outdir)/"ossi_score_median_iqr_by_year.pdf"
    fig.tight_layout()
    ax.tick_params(axis="x", labelrotation=90)
    plt.setp(ax.get_xticklabels(), ha="center")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf

def plot_boxplot_by_year(outdir, df, year_col, min_year=2010, max_year=2025):
    """
    Draw boxplots of ossi_score for each year present in df, restricted to [min_year, max_year].
    Ensures a tick label (year) is drawn under each xtick, rotated vertically and centered.
    Prevents extra ticks beyond max_year (no 2026).
    """
    import numpy as np

    # Ensure numeric and filter year range
    d = df.copy()
    d["ossi_score"] = pd.to_numeric(d["ossi_score"], errors="coerce")
    d[year_col] = pd.to_numeric(d[year_col], errors="coerce")
    d = d.dropna(subset=[year_col, "ossi_score"])
    d = d[(d[year_col] >= min_year) & (d[year_col] <= max_year)].copy()

    # Find the years actually present (sorted)
    years_present = sorted(d[year_col].unique())
    if len(years_present) == 0:
        raise SystemExit("No data in the requested year range to plot boxplots.")

    # Build list-of-arrays for boxplot for each present year
    data_by_year = [d.loc[d[year_col] == y, "ossi_score"].values for y in years_present]

    # Figure size scaled modestly to number of years to avoid cramped labels
    fig_w = max(10, 0.4 * len(years_present))
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    # Use 'tick_labels' param (matplotlib >=3.9) to avoid deprecation warnings.
    # Positions will be 1..N so we can control xlim precisely.
    positions = list(range(1, len(years_present) + 1))
    ax.boxplot(data_by_year, positions=positions, tick_labels=[str(int(y)) for y in years_present], showfliers=False)

    # Explicit ticks and tick labels (safe cross-version)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(int(y)) for y in years_present], rotation=90, ha="center")

    # Prevent extra tick beyond last year (no 2026)
    ax.set_xlim(0.5, len(years_present) + 0.5)

    # Axis labels and title
    ax.set_xlabel("Year", labelpad=12)
    ax.tick_params(axis="x", labelrotation=90)  # ensure vertical labels
    ax.set_ylabel("OSSI score")
    ax.set_title("OSSI score — distribution by year (boxplot)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Save PNG + PDF
    png = Path(outdir) / "ossi_score_boxplot_by_year.png"
    pdf = Path(outdir) / "ossi_score_boxplot_by_year.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_tier_stacked_pct(outdir, pct, min_year=2010, max_year=2025):
    """
    Stacked 100% bar chart of OSSI tiers by year.
    - Uses only the years present in `pct["year"]` (already filtered to min/max).
    - Places a vertical, centered label under each xtick.
    - Prevents Matplotlib from adding an extra tick beyond the last year.
    """
    import numpy as np

    # Ensure year is numeric and sorted
    pct = pct.copy()
    pct["year"] = pd.to_numeric(pct["year"], errors="coerce")
    pct = pct.dropna(subset=["year"])
    pct = pct[(pct["year"] >= min_year) & (pct["year"] <= max_year)].copy()
    if pct.empty:
        raise SystemExit("No data for tier plot in the requested year range.")

    years_present = list(pct["year"].astype(int).sort_values().values)
    n = len(years_present)

    # positions 1..n gives us complete control over ticks/limits
    positions = np.arange(1, n + 1)

    tiers = ["Gold", "Silver", "Bronze", "None"]
    # color-blind friendly palette (Option A)
    palette = {
        "Gold":   "#E69F00",
        "Silver": "#7F7F7F",
        "Bronze": "#8C510A",
        "None":   "#BDBDBD",
    }

    # Build arrays for each tier in the order we will stack them
    vals = {}
    for t in tiers:
        if t in pct.columns:
            # keep same order as years_present
            vals[t] = pct.set_index("year").reindex(years_present)[t].fillna(0).values
        else:
            vals[t] = np.zeros(n)

    # Make figure width scale with number of years so labels are ok
    fig_w = max(10, 0.5 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    bottom = np.zeros(n)
    for t in tiers:
        ax.bar(positions, vals[t], bottom=bottom, label=t, color=palette[t], edgecolor="none", linewidth=0.25)
        bottom += vals[t]

    # Ticks: one tick per bar, label is actual year
    ax.set_xticks(positions)
    ax.set_xticklabels([str(y) for y in years_present], rotation=90, ha="center")

    # Prevent extra tick beyond last year (no 2026)
    ax.set_xlim(0.5, n + 0.5)

    ax.set_xlabel("Year", labelpad=12)
    ax.set_ylabel("% of papers")
    ax.set_title("OSSI tiers — stacked percentages by year")
    ax.legend(loc="upper left", ncol=2, frameon=False)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    png = Path(outdir) / "ossi_tier_stacked_pct_by_year.png"
    pdf = Path(outdir) / "ossi_tier_stacked_pct_by_year.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_tsv", default="2.data/meta_under_request_tagged_open_science.tsv",
                    help="Input TSV (default: 2.data/meta_under_request_tagged_open_science.tsv)")
    ap.add_argument("--outdir", default="3.analyses/figures_ossi",
                    help="Output directory for figures and TSVs (default: 3.analyses/figures_ossi)")
    ap.add_argument("--min-year", type=int, default=2010)
    ap.add_argument("--max-year", type=int, default=2025)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_tsv, sep="\t", dtype=str).fillna("")
    # coerce ossi_score
    df["ossi_score"] = pd.to_numeric(df.get("ossi_score", pd.Series(dtype=float)), errors="coerce")

    year_col = infer_year_column(df)
    # filter year range
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df[(df[year_col] >= args.min_year) & (df[year_col] <= args.max_year)].copy()

    # summaries + data for plots
    summary, d_box = summarize_score_by_year(df, year_col, args.min_year, args.max_year)
    pct = tier_percent_by_year(df, year_col, args.min_year, args.max_year)

    # save TSV backing each plot
    summary_tsv = outdir / "ossi_score_median_iqr_by_year.tsv"

    # Don't mutate `summary` before plotting — make a copy for saving
    summary_to_save = summary.copy()
    summary_to_save.rename(columns={"median": "ossi_score_median"}, inplace=True)
    summary_to_save.to_csv(summary_tsv, sep="\t", index=False)

    box_tsv = outdir / "ossi_score_boxplot_by_year.tsv"
    d_box[[year_col, "ossi_score"]].rename(columns={year_col: "year"}).to_csv(box_tsv, sep="\t", index=False)

    pct_tsv = outdir / "ossi_tier_stacked_pct_by_year.tsv"
    pct.to_csv(pct_tsv, sep="\t", index=False)

    # plots (PNG + PDF) — now `summary` still has 'median'
    p1_png, p1_pdf = plot_median_iqr(outdir, summary)
    p2_png, p2_pdf = plot_boxplot_by_year(outdir, d_box, year_col)
    p3_png, p3_pdf = plot_tier_stacked_pct(outdir, pct)


    # multi-page PDF
    pdf_path = Path(outdir)/"ossi_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        for path in (p1_png, p2_png, p3_png):
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(11,8.5))
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("Saved:")
    for p in (p1_png, p1_pdf, summary_tsv, p2_png, p2_pdf, box_tsv, p3_png, p3_pdf, pct_tsv, pdf_path):
        print(f"- {p}")

if __name__ == "__main__":
    main()
