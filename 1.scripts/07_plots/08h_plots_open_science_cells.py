#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# 08h_plots_open_science_cells.py

 Cell plotting  for OSSI (Open Science Support Index).
 - Each cell  can be executed independently (VS Code / Jupyter style).
 - Every plot cell writes PNG, PDF, and the exact TSV behind the plot.
 - Defaults: input TSV = `2.data/meta_under_request_tagged_open_science.tsv`,
             output dir = `3.analyses/figures_ossi/`.

Create figures showing the evolution of OSSI (Open Science Support Index) between 2010 and 2025.

Plot design (recap)
    Median + IQR line: robust central trend with spread; annotated with yearly N.
    Yearly boxplots: distributional view (no outliers) to see compression/shift.
    Tier composition: stacked 100% bars for Gold/Silver/Bronze/None per year.


Defaults:
  --in      2.data/meta_under_request_tagged_open_science.tsv
  --outdir  3.analyses/figures_ossi

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


# %%
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
# -----------------------
# Configuration (edited)
# -----------------------
#IN_TSV  = Path("2.data/meta_under_request_tagged_open_science.tsv")
IN_TSV = Path("/Users/benoit/work/under_request/2.data/meta_under_request_tagged_open_science.tsv")
#OUTDIR  = Path("3.analyses/figures_ossi")
OUTDIR = Path("/Users/benoit/work/under_request/3.analyses/figures_ossi")

YEAR_MIN = 2010
YEAR_MAX = 2025
OUTDIR.mkdir(parents=True, exist_ok=True)
# %%
# Helpers
def infer_year_column(df: pd.DataFrame) -> str:
    """
    Return the column name containing the publication year.
    Prefers 'year' or 'pub_year', else parses from 'pubdate'/'date'/'publication_date'.
    As a last resort, extracts a 4-digit year with regex into '_year'.
    """
    for c in df.columns:
        if c.lower() in ("year", "pub_year"):
            return c
    for c in df.columns:
        if c.lower() in ("pubdate", "date", "publication_date"):
            ser = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            if ser.notna().any():
                df["_year"] = ser.dt.year
                return "_year"
    # fallback: regex on any column
    for c in df.columns:
        ser = df[c].astype(str).str.extract(r"(\d{4})", expand=False)
        yy = pd.to_numeric(ser, errors="coerce")
        if yy.notna().any():
            df["_year"] = yy
            return "_year"
    raise SystemExit("Could not infer a 'year' column. Add a 'year' or 'pub_year' column to the TSV.")
def load_and_filter(tsv: Path, year_min: int, year_max: int):
    """
    Load TSV, coerce ossi_score numeric, infer year, filter year range.
    Returns (df_filtered, year_col).
    """
    df = pd.read_csv(tsv, sep="\t", dtype=str).fillna("")
    df["ossi_score"] = pd.to_numeric(df.get("ossi_score", pd.Series(dtype=float)), errors="coerce")
    year_col = infer_year_column(df)
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df[(df[year_col] >= year_min) & (df[year_col] <= year_max)].copy()
    return df, year_col
def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
# %%
# Load data (run this once; re-run after changing IN_TSV/YEAR_MIN/YEAR_MAX)
df, YEAR_COL = load_and_filter(IN_TSV, YEAR_MIN, YEAR_MAX)
print(f"Loaded {len(df)} rows from {IN_TSV} with year column '{YEAR_COL}' filtered to [{YEAR_MIN}, {YEAR_MAX}]")
# %%
# Plot A — OSSI score: median + IQR by year
# Produces: PNG, PDF, TSV (ossi_score_median_iqr_by_year.*)
def summarize_score_by_year(df: pd.DataFrame, year_col: str):
    d = df[[year_col, "ossi_score"]].copy()
    d = d.dropna(subset=["ossi_score", year_col])
    grp = d.groupby(year_col)["ossi_score"]
    summary = grp.agg(n="size", q25=lambda s: s.quantile(0.25), median="median", q75=lambda s: s.quantile(0.75))
    summary = summary.reset_index().rename(columns={year_col: "year"}).sort_values("year")
    return summary
summary = summarize_score_by_year(df, YEAR_COL)
summary_tsv = OUTDIR / "ossi_score_median_iqr_by_year.tsv"
summary_to_save = summary.copy()
summary_to_save.rename(columns={"median": "ossi_score_median"}, inplace=True)
summary_to_save.to_csv(summary_tsv, sep="\t", index=False)
print(f"Wrote {summary_tsv}")
# Plot
fig_w = max(10, 0.4 * len(summary))
fig, ax = plt.subplots(figsize=(fig_w, 5))
ax.plot(summary["year"], summary["median"], marker="o")
ax.fill_between(summary["year"], summary["q25"], summary["q75"], alpha=0.2, label="IQR")
# Ticks: one per year present, vertical, centered
ax.set_xticks(summary["year"].values)
ax.set_xticklabels([str(int(y)) for y in summary["year"].values], rotation=90, ha="center")
ax.set_xlim(summary["year"].min() - 0.5, summary["year"].max() + 0.5)
ax.set_xlabel("Year", labelpad=12)
ax.set_ylabel("OSSI score")
ax.set_title("OSSI score — median and IQR by year")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
for _, row in summary.iterrows():
    ax.text(row["year"], row["median"], f" n={int(row['n'])}", fontsize=8, ha="left", va="bottom")
ax.legend(loc="best", frameon=False)
png = OUTDIR / "ossi_score_median_iqr_by_year.png"
pdf = OUTDIR / "ossi_score_median_iqr_by_year.pdf"
fig.tight_layout()
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)
print(f"Wrote {png} and {pdf}")
# %%
# Plot B — OSSI score: boxplot by year
# Produces: PNG, PDF, TSV (ossi_score_boxplot_by_year.*)
# backing TSV (long format)
box_tsv = OUTDIR / "ossi_score_boxplot_by_year.tsv"
df[[YEAR_COL, "ossi_score"]].rename(columns={YEAR_COL: "year"}).to_csv(box_tsv, sep="\t", index=False)
print(f"Wrote {box_tsv}")
# Prepare data by year present
d = df[[YEAR_COL, "ossi_score"]].dropna().sort_values(YEAR_COL)
years_present = sorted(d[YEAR_COL].unique())
data_by_year = [d.loc[d[YEAR_COL] == y, "ossi_score"].values for y in years_present]
fig_w = max(12, 0.4 * len(years_present))
fig, ax = plt.subplots(figsize=(fig_w, 6))
positions = list(range(1, len(years_present) + 1))
ax.boxplot(data_by_year, positions=positions, tick_labels=[str(int(y)) for y in years_present], showfliers=False)
ax.set_xticks(positions)
ax.set_xticklabels([str(int(y)) for y in years_present], rotation=90, ha="center")
ax.set_xlim(0.5, len(years_present) + 0.5)
ax.set_xlabel("Year", labelpad=12)
ax.set_ylabel("OSSI score")
ax.set_title("OSSI score — distribution by year (boxplot)")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
png = OUTDIR / "ossi_score_boxplot_by_year.png"
pdf = OUTDIR / "ossi_score_boxplot_by_year.pdf"
fig.tight_layout()
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)
print(f"Wrote {png} and {pdf}")
# %%
# Plot C — OSSI tiers: stacked 100% bar by year
# Produces: PNG, PDF, TSV (ossi_tier_stacked_pct_by_year.*)
def tier_percent_by_year(df: pd.DataFrame, year_col: str):
    d = df[[year_col, "ossi_tier"]].copy()
    d["ossi_tier"] = d["ossi_tier"].fillna("None")
    order = ["Gold", "Silver", "Bronze", "None"]
    d.loc[~d["ossi_tier"].isin(order), "ossi_tier"] = "None"
    ct = d.groupby([year_col, "ossi_tier"]).size().unstack(fill_value=0)
    for t in order:
        if t not in ct.columns:
            ct[t] = 0
    ct = ct[order]
    pct = (ct.T / ct.sum(axis=1)).T * 100.0
    pct = pct.reset_index().rename(columns={year_col: "year"}).sort_values("year")
    return pct
pct = tier_percent_by_year(df, YEAR_COL)
pct_tsv = OUTDIR / "ossi_tier_stacked_pct_by_year.tsv"
pct.to_csv(pct_tsv, sep="\t", index=False)
print(f"Wrote {pct_tsv}")
years_present = list(pct["year"].astype(int).sort_values().values)
n = len(years_present)
positions = np.arange(1, n + 1)
tiers = ["Gold", "Silver", "Bronze", "None"]
palette = {
    "Gold":   "#E69F00",
    "Silver": "#7F7F7F",
    "Bronze": "#8C510A",
    "None":   "#BDBDBD",
}
vals = {}
for t in tiers:
    if t in pct.columns:
        vals[t] = pct.set_index("year").reindex(years_present)[t].fillna(0).values
    else:
        vals[t] = np.zeros(n)
fig_w = max(12, 0.5 * n)
fig, ax = plt.subplots(figsize=(fig_w, 6))
bottom = np.zeros(n)
for t in tiers:
    ax.bar(positions, vals[t], bottom=bottom, label=t, color=palette[t], edgecolor="none")
    bottom += vals[t]
ax.set_xticks(positions)
ax.set_xticklabels([str(y) for y in years_present], rotation=90, ha="center")
ax.set_xlim(0.5, n + 0.5)
ax.set_xlabel("Year", labelpad=12)
ax.set_ylabel("% of papers")
ax.set_title("OSSI tiers — stacked percentages by year")
ax.legend(loc="upper left", ncol=2, frameon=False)
ax.set_ylim(0, 100)
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
png = OUTDIR / "ossi_tier_stacked_pct_by_year.png"
pdf = OUTDIR / "ossi_tier_stacked_pct_by_year.pdf"
fig.tight_layout()
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)
print(f"Wrote {png} and {pdf}")
# %%
# Plot D — Single PDF assembling the three plots (optional)
pdf_path = OUTDIR / "ossi_summary.pdf"
with PdfPages(pdf_path) as pdf:
    for path in (
        OUTDIR / "ossi_score_median_iqr_by_year.png",
        OUTDIR / "ossi_score_boxplot_by_year.png",
        OUTDIR / "ossi_tier_stacked_pct_by_year.png",
    ):
        img = plt.imread(path)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.imshow(img)
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
print(f"Wrote {pdf_path}")


# %%
# Plot E — OSSI score: violin by year (distribution view like the boxplot,
# but showing the full density). Produces PNG, PDF, TSV.

# Backing TSV (long format, same structure as the boxplot one but kept separate for provenance)
violin_tsv = OUTDIR / "ossi_score_violin_by_year.tsv"
df[[YEAR_COL, "ossi_score"]].rename(columns={YEAR_COL: "year"}).to_csv(violin_tsv, sep="\t", index=False)
print(f"Wrote {violin_tsv}")

# Prepare data by year present
d = df[[YEAR_COL, "ossi_score"]].dropna().sort_values(YEAR_COL)
years_present = sorted(d[YEAR_COL].unique())
data_by_year = [d.loc[d[YEAR_COL] == y, "ossi_score"].values for y in years_present]

# Figure width scales with the number of years for readability
fig_w = max(12, 0.4 * len(years_present))
fig, ax = plt.subplots(figsize=(fig_w, 6))

positions = list(range(1, len(years_present) + 1))

# Matplotlib violin plot (no explicit colors; show median line, hide extrema caps)
vp = ax.violinplot(
    data_by_year,
    positions=positions,
    widths=0.8,
    showmeans=False,
    showmedians=True,
    showextrema=False,
)

# Optional: lighten the violins a touch (still no explicit colors set)
for b in vp['bodies']:
    b.set_alpha(0.7)

# Overlay the median as points for clarity (computed directly)
medians = [np.median(v) if len(v) else np.nan for v in data_by_year]
ax.plot(positions, medians, marker="o", linestyle="None")  # default style, no explicit color

# Ticks: one per violin, vertical, centered; avoid extra tick (e.g., 2026)
ax.set_xticks(positions)
ax.set_xticklabels([str(int(y)) for y in years_present], rotation=90, ha="center")
ax.set_xlim(0.5, len(years_present) + 0.5)

ax.set_xlabel("Year", labelpad=12)
ax.set_ylabel("OSSI score")
ax.set_title("OSSI score — violin plot by year")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)

# Save outputs
png = OUTDIR / "ossi_score_violin_by_year.png"
pdf = OUTDIR / "ossi_score_violin_by_year.pdf"
fig.tight_layout()
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)
print(f"Wrote {png} and {pdf}")




# %%
# Plot G — OSSI tiers: stacked 100% bars by JOURNAL (like the by-year plot)
# Produces: PNG, PDF, TSV (ossi_tier_stacked_pct_by_journal.*)

def infer_journal_column(df: pd.DataFrame) -> str:
    """Best-effort guess of the journal column."""
    candidates = [
        "journal", "journal_name", "journal_title", "journal_full", "source",
        "journal_abbrev", "journal_abbreviation", "journal-title"
    ]
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    # last resort: pick the first column with 'journal' in its name
    for c in df.columns:
        if "journal" in c.lower():
            return c
    raise SystemExit("Could not infer a journal column. Please add one of: "
                     "'journal', 'journal_name', 'journal_title', 'source', ...")

JOURNAL_COL = infer_journal_column(df)

def tier_percent_by_group(df: pd.DataFrame, group_col: str):
    d = df[[group_col, "ossi_tier"]].copy()
    d["ossi_tier"] = d["ossi_tier"].fillna("None")
    order = ["Gold", "Silver", "Bronze", "None"]
    d.loc[~d["ossi_tier"].isin(order), "ossi_tier"] = "None"

    # counts and 100% percentages
    ct = d.groupby([group_col, "ossi_tier"]).size().unstack(fill_value=0)
    for t in order:
        if t not in ct.columns:
            ct[t] = 0
    ct = ct[order]
    pct = (ct.T / ct.sum(axis=1)).T * 100.0

    # optional sorting: more OS support first (Gold then Silver, then Bronze)
    sort_cols = ["Gold", "Silver", "Bronze", "None"]
    pct = pct.sort_values(by=sort_cols, ascending=[False, False, False, True])

    pct = pct.reset_index().rename(columns={group_col: "journal"})
    ct = ct.reset_index().rename(columns={group_col: "journal"})
    return pct, ct  # return percentages and counts (counts not plotted but useful to inspect)

pctJ, ctJ = tier_percent_by_group(df, JOURNAL_COL)

# Save the TSV backing the plot (wide table: journal, Gold, Silver, Bronze, None)
pct_tsv = OUTDIR / "ossi_tier_stacked_pct_by_journal.tsv"
pctJ.to_csv(pct_tsv, sep="\t", index=False)
print(f"Wrote {pct_tsv}")

# (Optional) also save counts for transparency/debugging
ct_tsv = OUTDIR / "ossi_tier_counts_by_journal.tsv"
ctJ.to_csv(ct_tsv, sep="\t", index=False)
print(f"Wrote {ct_tsv}")

# Build the stacked bars
journals = pctJ["journal"].astype(str).tolist()
n = len(journals)
positions = np.arange(1, n + 1)

tiers = ["Gold", "Silver", "Bronze", "None"]
palette = {
    "Gold":   "#E69F00",  # orange-gold
    "Silver": "#7F7F7F",  # gray
    "Bronze": "#8C510A",  # bronze/brown
    "None":   "#BDBDBD",  # light gray
}

# values per tier, in sorted journal order
vals = {t: pctJ[t].values if t in pctJ.columns else np.zeros(n) for t in tiers}

fig_w = max(12, 0.5 * n)
fig, ax = plt.subplots(figsize=(fig_w, 6))

bottom = np.zeros(n)
for t in tiers:
    ax.bar(
        positions, vals[t], bottom=bottom,
        label=t, color=palette[t],
        edgecolor="none"  # no borders as requested
    )
    bottom += vals[t]

# ticks: vertical, centered; widen margins to avoid clipping long names
ax.set_xticks(positions)
ax.set_xticklabels(journals, rotation=90, ha="center")
ax.set_xlim(0.5, n + 0.5)

ax.set_xlabel("Journal", labelpad=12)
ax.set_ylabel("% of papers")
ax.set_title("OSSI tiers — stacked percentages by journal")
ax.legend(loc="upper right", ncol=2, frameon=False)
ax.set_ylim(0, 100)
ax.grid(True, axis="y", linestyle=":", alpha=0.5)

png = OUTDIR / "ossi_tier_stacked_pct_by_journal.png"
pdf = OUTDIR / "ossi_tier_stacked_pct_by_journal.pdf"
fig.tight_layout()
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)
print(f"Wrote {png} and {pdf}")




# %%
# Plot H — OSSI tiers: stacked 100% bars by YEAR, one panel per JOURNAL (vertical stack)
# Produces: PNG, PDF, TSV (ossi_tier_stacked_pct_by_year_by_journal.*)
# Notes:
# - Pure matplotlib, color-blind friendly palette, no bar borders.
# - By default, x tick labels are shown only on the bottom panel to save space; set SHOW_XLABELS_ALL=True to show on all.

from collections import defaultdict

def infer_journal_column(df: pd.DataFrame) -> str:
    candidates = [
        "journal", "journal_name", "journal_title", "journal_full", "source",
        "journal_abbrev", "journal_abbreviation", "journal-title"
    ]
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    for c in df.columns:
        if "journal" in c.lower():
            return c
    raise SystemExit("Could not infer a journal column. Add one of: "
                     "'journal', 'journal_name', 'journal_title', 'source', ...")

JOURNAL_COL = infer_journal_column(df)

# ------- compute percent table: (journal, year) -> % Gold/Silver/Bronze/None -------
def tier_percent_by_year_and_journal(df: pd.DataFrame, year_col: str, journal_col: str):
    d = df[[year_col, journal_col, "ossi_tier"]].copy()
    d["ossi_tier"] = d["ossi_tier"].fillna("None")
    order = ["Gold", "Silver", "Bronze", "None"]
    d.loc[~d["ossi_tier"].isin(order), "ossi_tier"] = "None"

    # counts
    ct = d.groupby([journal_col, year_col, "ossi_tier"]).size().unstack(fill_value=0)
    for t in order:
        if t not in ct.columns:
            ct[t] = 0
    ct = ct[order]
    total = ct.sum(axis=1).rename("n_total")
    pct = (ct.T / total).T * 100.0
    pct = pct.reset_index().rename(columns={journal_col: "journal", year_col: "year"})
    ct  = ct.reset_index().rename(columns={journal_col: "journal", year_col: "year"})
    pct = pct.merge(total.reset_index().rename(columns={journal_col: "journal", year_col: "year"}), on=["journal","year"])
    return pct.sort_values(["journal","year"]), ct.sort_values(["journal","year"])

pctJY, ctJY = tier_percent_by_year_and_journal(df, YEAR_COL, JOURNAL_COL)

# Save TSV backing this plot
pct_tsv = OUTDIR / "ossi_tier_stacked_pct_by_year_by_journal.tsv"
pctJY.to_csv(pct_tsv, sep="\t", index=False)
print(f"Wrote {pct_tsv}")

# ------- figure layout & drawing -------
# Years present across the filtered dataset (same positions for all panels)
years_all = sorted(pd.to_numeric(df[YEAR_COL], errors="coerce").dropna().unique().astype(int))
positions = np.arange(1, len(years_all) + 1)

# Order journals by overall OS support (Gold + Silver across all years), desc
order_cols = ["Gold","Silver","Bronze","None"]
agg_os = (pctJY.groupby("journal")[["Gold","Silver"]].mean().sum(axis=1)).sort_values(ascending=False)
journals = agg_os.index.tolist()

tiers = ["Gold", "Silver", "Bronze", "None"]
palette = {
    "Gold":   "#E69F00",
    "Silver": "#7F7F7F",
    "Bronze": "#8C510A",
    "None":   "#BDBDBD",
}

# figure size: width scales with #years, height ~1.2" per journal
fig_w = max(6, 0.45 * len(years_all))
fig_h = max(6, 1.2 * len(journals))
fig, axes = plt.subplots(nrows=len(journals), ncols=1, figsize=(fig_w, fig_h), sharex=True)

# If only one journal, axes may not be an array
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

SHOW_XLABELS_ALL = False  # set True if I want x labels on every panel

for i, journal in enumerate(journals):
    ax = axes[i]
    sub = pctJY[pctJY["journal"] == journal].copy()
    # align values to the fixed year axis; missing years -> 0
    vals = {}
    for t in tiers:
        s = sub.set_index("year")[t] if t in sub.columns else pd.Series(dtype=float)
        vals[t] = pd.Series(0.0, index=years_all)
        vals[t].loc[s.index] = s.values

    # stacked bars
    bottom = np.zeros(len(years_all))
    for t in tiers:
        ax.bar(positions, vals[t].values, bottom=bottom, color=palette[t], edgecolor="none", label=t if i==0 else None)
        bottom += vals[t].values

    # panel title on the left
    ax.text(0.01, 0.75, journal, transform=ax.transAxes, ha="left", va="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # X ticks: vertical labels; only draw tick labels on bottom by default
    ax.set_xlim(0.5, len(years_all) + 0.5)
    ax.set_xticks(positions)
    if SHOW_XLABELS_ALL or i == len(journals)-1:
        ax.set_xticklabels([str(y) for y in years_all], rotation=90, ha="center")
    else:
        ax.set_xticklabels([])

# Axis labels and legend
axes[-1].set_xlabel("Year", labelpad=12)
fig.text(0.04, 0.5, "% of papers", va="center", rotation="vertical")
# Single shared legend at top-right
handles = [plt.Rectangle((0,0),1,1, color=palette[t]) for t in tiers]
fig.legend(handles, tiers, loc="upper right", ncol=4, frameon=False)

fig.tight_layout(rect=[0.06, 0.03, 0.98, 0.96])  # leave space for legend and y-label

png = OUTDIR / "ossi_tier_stacked_pct_by_year__per_journal_panels.png"
pdf = OUTDIR / "ossi_tier_stacked_pct_by_year__per_journal_panels.pdf"
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)
print(f"Wrote {png} and {pdf}")


# %%
# Plot I — % of papers per year with each open-science indicator (6 lines)
# Produces: PNG, PDF, TSV (ossi_indicator_pct_by_year.*)

INDICATOR_COLS = [
    "data_public_repo",
    "dataset_doi_present",
    "code_available",
    "protocol_shared",
    "source_data_present",
]

# Legend labels (one column) with brief examples
LEGEND_LABELS = {
    "data_public_repo":   "data_public_repo (GEO/SRA/ENA, ArrayExpress...)",
    "dataset_doi_present":"dataset_doi_present (DataCite DOI: Zenodo, Figshare)",
    "code_available":     "code_available (GitHub/GitLab/Bitbucket/CodeOcean)",
    "protocol_shared":    "protocol_shared (protocols.io, registered protocol/PROSPERO)",
    "source_data_present":"source_data_present (“Source Data” files)",
}

def to_bool_series(s: pd.Series) -> pd.Series:
    """
    Convert heterogeneous values to boolean.
    - numeric: True if > 0
    - strings: True if in {'1','true','yes','y'} (case-insensitive)
               False if empty/NA/'0'/'false'/'no'/'n'
               Otherwise: non-empty -> True
    """
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce").fillna(0) > 0
    low = s.astype(str).str.strip().str.lower()
    true_set  = {"1","true","yes","y"}
    false_set = {"","0","false","no","n","na","nan","none"}
    out = pd.Series(False, index=s.index)
    out = out.where(~low.isin(true_set), True)
    out = out.where(~low.isin(false_set), False)
    # anything else non-empty -> True
    out = out.where(~(~low.isin(true_set | false_set) & (low != "")), True)
    return out

# Build long table: year × metric -> % yes (and N)
year_vals = sorted(pd.to_numeric(df[YEAR_COL], errors="coerce").dropna().astype(int).unique())
rows = []
for metric in INDICATOR_COLS:
    if metric not in df.columns:
        print(f"[warn] column not found: {metric} — skipping")
        continue
    b = to_bool_series(df[metric])
    tmp = pd.DataFrame({"year": pd.to_numeric(df[YEAR_COL], errors="coerce"), "ok": b})
    tmp = tmp.dropna(subset=["year"]).copy()
    tmp["year"] = tmp["year"].astype(int)
    grp = tmp.groupby("year")["ok"]
    pct = (grp.mean() * 100.0).reindex(year_vals, fill_value=0)
    n   = grp.size().reindex(year_vals, fill_value=0)
    for y in year_vals:
        rows.append({"year": y, "metric": metric, "pct_yes": pct.loc[y], "n": int(n.loc[y])})

pct_long = pd.DataFrame(rows).sort_values(["metric","year"])

# Save TSV
out_tsv = OUTDIR / "ossi_indicator_pct_by_year.tsv"
pct_long.to_csv(out_tsv, sep="\t", index=False)
print(f"Wrote {out_tsv}")

# Pivot to wide for plotting
wide = pct_long.pivot(index="year", columns="metric", values="pct_yes").reindex(year_vals)

# Plot — six lines, vertical year labels, legend in 1 column with detailed labels
fig_w = max(12, 0.45 * len(year_vals))
fig, ax = plt.subplots(figsize=(fig_w, 6))

for metric in INDICATOR_COLS:
    if metric in wide.columns:
        ax.plot(wide.index.values, wide[metric].values, marker="o",
                label=LEGEND_LABELS.get(metric, metric))

ax.set_xticks(year_vals)
ax.set_xticklabels([str(y) for y in year_vals], rotation=90, ha="center")
ax.set_xlim(min(year_vals) - 0.5, max(year_vals) + 0.5)
ax.set_ylim(0, 100)

ax.set_xlabel("Year", labelpad=12)
ax.set_ylabel("% of papers with indicator")
ax.set_title("Open-science indicators — % of papers per year")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)

# One-column legend (left/top works well for many lines)
ax.legend(loc="upper left", frameon=False, ncol=1)

png = OUTDIR / "ossi_indicator_pct_by_year.png"
pdf = OUTDIR / "ossi_indicator_pct_by_year.pdf"
fig.tight_layout()
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)
print(f"Wrote {png} and {pdf}")



# %%
# Plot J — % of papers by JOURNAL with each open-science indicator (six-line plot over journals)
# Produces: PNG, PDF, TSV (ossi_indicator_pct_by_journal.*)
# - Uses the legend text (one column).
# - Journals on x-axis (vertical labels), 6 indicator lines on y (%).
# - Sorted by composite OS support (mean across indicators), highest first.

# Use the legend labels
LEGEND_LABELS = {
    "data_public_repo":   "data_public_repo (GEO/SRA/ENA, ArrayExpress...)",
    "dataset_doi_present":"dataset_doi_present (DataCite DOI: Zenodo, Figshare)",
    "code_available":     "code_available (GitHub/GitLab/Bitbucket/CodeOcean)",
    "protocol_shared":    "protocol_shared (protocols.io, registered protocol/PROSPERO)",
    "source_data_present":"source_data_present (“Source Data” files)",
}

# Indicators to plot (derived from legend keys to keep them in sync)
INDICATOR_COLS = list(LEGEND_LABELS.keys())

# Helper: booleanize mixed columns
def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce").fillna(0) > 0
    low = s.astype(str).str.strip().str.lower()
    true_set  = {"1","true","yes","y"}
    false_set = {"","0","false","no","n","na","nan","none"}
    out = pd.Series(False, index=s.index)
    out = out.where(~low.isin(true_set), True)
    out = out.where(~low.isin(false_set), False)
    # anything else non-empty -> True
    out = out.where(~(~low.isin(true_set | false_set) & (low != "")), True)
    return out

# Infer journal column (reuse if already defined)
if "infer_journal_column" not in globals():
    def infer_journal_column(df: pd.DataFrame) -> str:
        candidates = [
            "journal", "journal_name", "journal_title", "journal_full", "source",
            "journal_abbrev", "journal_abbreviation", "journal-title"
        ]
        low = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in low:
                return low[cand]
        for c in df.columns:
            if "journal" in c.lower():
                return c
        raise SystemExit("Could not infer a journal column. Add one of: "
                         "'journal', 'journal_name', 'journal_title', 'source', ...")

JOURNAL_COL = infer_journal_column(df)

# Build long table: journal × metric -> % yes (and N)
journals = sorted(df[JOURNAL_COL].astype(str).unique())
rows = []
for metric in INDICATOR_COLS:
    if metric not in df.columns:
        print(f"[warn] column not found: {metric} — skipping")
        continue
    b = to_bool_series(df[metric])
    tmp = pd.DataFrame({ "journal": df[JOURNAL_COL].astype(str), "ok": b })
    grp = tmp.groupby("journal")["ok"]
    pct = (grp.mean() * 100.0).reindex(journals, fill_value=0)
    n   = grp.size().reindex(journals, fill_value=0)
    for j in journals:
        rows.append({"journal": j, "metric": metric, "pct_yes": pct.loc[j], "n": int(n.loc[j])})

pct_long_j = pd.DataFrame(rows)

# Sort journals by composite OS support (mean across indicators), descending
comp = (pct_long_j.pivot(index="journal", columns="metric", values="pct_yes")
        .reindex(columns=INDICATOR_COLS)
        .mean(axis=1)
        .sort_values(ascending=False))
journals_sorted = comp.index.tolist()

# Save TSV
out_tsv = OUTDIR / "ossi_indicator_pct_by_journal.tsv"
pct_long_j.sort_values(["metric","journal"]).to_csv(out_tsv, sep="\t", index=False)
print(f"Wrote {out_tsv}")

# Pivot to wide for plotting in the sorted journal order
wide_j = (pct_long_j
          .pivot(index="journal", columns="metric", values="pct_yes")
          .reindex(journals_sorted))

# Plot — six lines across journals (x categorical -> positions 1..N)
xpos = np.arange(1, len(journals_sorted) + 1)
fig_w = max(14, 0.45 * len(journals_sorted))
fig, ax = plt.subplots(figsize=(fig_w, 6))

for metric in INDICATOR_COLS:
    if metric in wide_j.columns:
        ax.plot(xpos, wide_j[metric].values, marker="o",
                label=LEGEND_LABELS.get(metric, metric))

ax.set_xticks(xpos)
ax.set_xticklabels(journals_sorted, rotation=90, ha="center")
ax.set_xlim(0.5, len(journals_sorted) + 0.5)
ax.set_ylim(0, 100)

ax.set_xlabel("Journal", labelpad=12)
ax.set_ylabel("% of papers with indicator")
ax.set_title("Open-science indicators — % of papers by journal")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)

# One-column legend with the detailed labels
ax.legend(loc="upper right", frameon=False, ncol=1)

png = OUTDIR / "ossi_indicator_pct_by_journal.png"
pdf = OUTDIR / "ossi_indicator_pct_by_journal.pdf"
fig.tight_layout()
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)
print(f"Wrote {png} and {pdf}")




# %%
# Plot J2 — % of papers by JOURNAL with each indicator (thin grouped bars)
# Reuses pct_long_j / wide_j / journals_sorted from Plot J. If missing, it rebuilds them.
# Outputs: PNG/PDF (ossi_indicator_pct_by_journal_bars.*)
try:
    wide_j, journals_sorted, INDICATOR_COLS, LEGEND_LABELS  # noqa: F823
except NameError:
    # Rebuild from df if Plot J hasn't been run in this session
    LEGEND_LABELS = {
        "data_public_repo":   "data_public_repo (GEO/SRA/ENA, ArrayExpress...)",
        "dataset_doi_present":"dataset_doi_present (DataCite DOI: Zenodo, Figshare)",
        "code_available":     "code_available (GitHub/GitLab/Bitbucket/CodeOcean)",
        "protocol_shared":    "protocol_shared (protocols.io, registered protocol/PROSPERO)",
        "source_data_present":"source_data_present (“Source Data” files)",
    }
    INDICATOR_COLS = list(LEGEND_LABELS.keys())

    def to_bool_series(s: pd.Series) -> pd.Series:
        if s.dtype.kind in "biufc":
            return pd.to_numeric(s, errors="coerce").fillna(0) > 0
        low = s.astype(str).str.strip().str.lower()
        true_set  = {"1","true","yes","y"}
        false_set = {"","0","false","no","n","na","nan","none"}
        out = pd.Series(False, index=s.index)
        out = out.where(~low.isin(true_set), True)
        out = out.where(~low.isin(false_set), False)
        out = out.where(~(~low.isin(true_set | false_set) & (low != "")), True)
        return out

    def infer_journal_column(df: pd.DataFrame) -> str:
        candidates = [
            "journal","journal_name","journal_title","journal_full","source",
            "journal_abbrev","journal_abbreviation","journal-title"
        ]
        low = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in low:
                return low[cand]
        for c in df.columns:
            if "journal" in c.lower():
                return c
        raise SystemExit("Could not infer a journal column.")
    JOURNAL_COL = infer_journal_column(df)

    journals = sorted(df[JOURNAL_COL].astype(str).unique())
    rows = []
    for metric in INDICATOR_COLS:
        if metric not in df.columns:
            continue
        b = to_bool_series(df[metric])
        tmp = pd.DataFrame({"journal": df[JOURNAL_COL].astype(str), "ok": b})
        grp = tmp.groupby("journal")["ok"]
        pct = (grp.mean() * 100.0).reindex(journals, fill_value=0)
        n   = grp.size().reindex(journals, fill_value=0)
        for j in journals:
            rows.append({"journal": j, "metric": metric, "pct_yes": pct.loc[j], "n": int(n.loc[j])})
    pct_long_j = pd.DataFrame(rows)
    comp = (pct_long_j.pivot(index="journal", columns="metric", values="pct_yes")
            .reindex(columns=INDICATOR_COLS)
            .mean(axis=1)
            .sort_values(ascending=False))
    journals_sorted = comp.index.tolist()
    wide_j = (pct_long_j.pivot(index="journal", columns="metric", values="pct_yes")
              .reindex(journals_sorted))

import numpy as np
xpos = np.arange(1, len(journals_sorted) + 1)
fig_w = max(14, 0.5 * len(journals_sorted))
fig, ax = plt.subplots(figsize=(fig_w, 6))

# Thin grouped bars: small offsets per indicator around each journal's x position.
k = len(INDICATOR_COLS)
bar_width = min(0.12, 0.6 / max(1, k))   # really thin bars; cap total group width
# Center the group around each x: offsets symmetric around 0
offsets = (np.arange(k) - (k - 1) / 2.0) * (bar_width + 0.02)

for i, metric in enumerate(INDICATOR_COLS):
    if metric in wide_j.columns:
        ax.bar(
            xpos + offsets[i],
            wide_j[metric].values,
            width=bar_width,
            label=LEGEND_LABELS.get(metric, metric),
            edgecolor="none",   # clean look
        )

ax.set_xticks(xpos)
ax.set_xticklabels(journals_sorted, rotation=90, ha="center")
ax.set_xlim(0.5, len(journals_sorted) + 0.5)
ax.set_ylim(0, 100)

ax.set_xlabel("Journal", labelpad=12)
ax.set_ylabel("% of papers with indicator")
ax.set_title("Open-science indicators — % of papers by journal (thin grouped bars)")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)

# One-column legend with the detailed labels
ax.legend(loc="upper right", frameon=False, ncol=1)

png = OUTDIR / "ossi_indicator_pct_by_journal_bars.png"
pdf = OUTDIR / "ossi_indicator_pct_by_journal_bars.pdf"
fig.tight_layout()
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)
print(f"Wrote {png} and {pdf}")



# %%
# Plot L — OSSI tiers stacked 100% bars by YEAR, split into:
#   - not_UR           (under_request == 0)
#   - UR & genuine==0  (under_request == 1 and genuine == 0)
# For each year: two adjacent stacked bars.
# Produces:
#   - TSV (percent): ossi_tier_stacked_pct_by_year_notUR_vs_URg0.tsv
#   - TSV (counts):  ossi_tier_counts_by_year_notUR_vs_URg0.tsv
#   - PNG/PDF:       ossi_tier_stacked_pct_by_year_notUR_vs_URg0.{png,pdf}

def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce").fillna(0) > 0
    low = s.astype(str).str.strip().str.lower()
    true_set  = {"1","true","yes","y"}
    false_set = {"","0","false","no","n","na","nan","none"}
    out = pd.Series(False, index=s.index)
    out = out.where(~low.isin(true_set), True)
    out = out.where(~low.isin(false_set), False)
    # anything else non-empty -> True
    out = out.where(~(~low.isin(true_set | false_set) & (low != "")), True)
    return out

# Ensure required columns exist
missing_cols = [c for c in ["under_request","genuine","ossi_tier", YEAR_COL] if c not in df.columns]
if missing_cols:
    raise SystemExit(f"Missing required columns for this plot: {missing_cols}")

# Prepare group labels
d = df[[YEAR_COL, "ossi_tier", "under_request", "genuine"]].copy()
d[YEAR_COL] = pd.to_numeric(d[YEAR_COL], errors="coerce")
d = d.dropna(subset=[YEAR_COL, "ossi_tier"]).copy()
d["year"] = d[YEAR_COL].astype(int)

ur = _to_bool(d["under_request"])
gen = _to_bool(d["genuine"])

d["group"] = pd.NA
d.loc[~ur, "group"] = "not_UR"
d.loc[ur & (~gen), "group"] = "UR_g0"
d = d.dropna(subset=["group"]).copy()

# Normalize tiers to expected set
tiers = ["Gold", "Silver", "Bronze", "None"]
d["ossi_tier"] = d["ossi_tier"].fillna("None")
d.loc[~d["ossi_tier"].isin(tiers), "ossi_tier"] = "None"

# Counts by (year, group, tier)
ct = d.groupby(["year", "group", "ossi_tier"]).size().unstack(fill_value=0)
for t in tiers:
    if t not in ct.columns:
        ct[t] = 0
ct = ct[tiers]  # consistent order

# Percentages within each (year, group)
tot = ct.sum(axis=1).rename("n_total")
pct = (ct.T / tot).T * 100.0
pct = pct.reset_index().sort_values(["year", "group"])
ct  = ct.reset_index().sort_values(["year", "group"])
pct["n_total"] = tot.reset_index(drop=True)

# Save TSVs
pct_tsv = OUTDIR / "ossi_tier_stacked_pct_by_year_notUR_vs_URg0.tsv"
ct_tsv  = OUTDIR / "ossi_tier_counts_by_year_notUR_vs_URg0.tsv"
pct.to_csv(pct_tsv, sep="\t", index=False)
ct.to_csv(ct_tsv,  sep="\t", index=False)
print(f"Wrote {pct_tsv}")
print(f"Wrote {ct_tsv}")

# ---------- Plotting ----------
import numpy as np
import matplotlib.patches as mpatches

years_present = sorted(pct["year"].unique().astype(int))
groups = ["not_UR", "UR_g0"]  # fixed order on x within each year

# x positions: for each year two bars with horizontal offset
x_base = np.arange(1, len(years_present) + 1)

# Wider bars + spacing (adjust if needed)
bar_gap   = 0.30     # distance between the two bars of a year
bar_width = 0.24     # actual bar width (wider than before)

pos = {
    "not_UR": x_base - bar_gap/2,
    "UR_g0":  x_base + bar_gap/2,
}

# Build per-tier arrays aligned to (years_present, groups)
tiers = ["Gold", "Silver", "Bronze", "None"]
vals = {g: {t: np.zeros(len(years_present)) for t in tiers} for g in groups}
for i, y in enumerate(years_present):
    for g in groups:
        row = pct[(pct["year"] == y) & (pct["group"] == g)]
        if row.empty:
            continue
        for t in tiers:
            vals[g][t][i] = float(row.iloc[0].get(t, 0.0))

# Color-blind friendly tier palette
palette = {
    "Gold":   "#E69F00",
    "Silver": "#7F7F7F",
    "Bronze": "#8C510A",
    "None":   "#BDBDBD",
}

fig_w = max(14, 0.55 * len(years_present))
fig, ax = plt.subplots(figsize=(fig_w, 6))

# Draw stacked bars for each group within each year
for g in groups:
    bottom = np.zeros(len(years_present))
    # style per group
    if g == "UR_g0":
        edge_kws = dict(edgecolor="black", linewidth=0.8)   # <<< black borders
    else:
        edge_kws = dict(edgecolor="none")                   # <<< plain

    for t in tiers:
        ax.bar(
            pos[g],
            vals[g][t],
            width=bar_width,
            bottom=bottom,
            color=palette[t],
            **edge_kws,
            label=t if g == groups[0] else None  # tier legend only once
        )
        bottom += vals[g][t]

# X ticks: years at midpoints
ax.set_xticks(x_base)
ax.set_xticklabels([str(y) for y in years_present], rotation=90, ha="center")

ax.set_xlim(0.5, len(years_present) + 0.5)
ax.set_ylim(0, 100)
ax.set_xlabel("Year", labelpad=12)
ax.set_ylabel("% of papers")
ax.set_title("OSSI tiers — stacked % by year, split: not_UR vs UR & genuine=0")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)

# Tier legend (left) + group-style key (right)
tier_handles = [mpatches.Patch(color=palette[t], label=t) for t in tiers]
group_handles = [
    mpatches.Patch(facecolor="#CCCCCC", edgecolor="none", label="not_UR"),
    mpatches.Patch(facecolor="#CCCCCC", edgecolor="black", linewidth=0.8, label="UR & g=0"),
]

leg1 = ax.legend(handles=tier_handles, loc="upper left", frameon=False, title="Tier")
ax.add_artist(leg1)
ax.legend(handles=group_handles, loc="upper right", frameon=False, title="Group (bar style)")

png = OUTDIR / "ossi_tier_stacked_pct_by_year_notUR_vs_URg0.png"
pdf = OUTDIR / "ossi_tier_stacked_pct_by_year_notUR_vs_URg0.pdf"
fig.tight_layout()
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)
print(f"Wrote {png} and {pdf}")



# %%
# Plot M — Two-panel layout: stacked % by YEAR for (left) not_UR and (right) UR & genuine=0
# Produces:
#   - TSV: ossi_tier_stacked_pct_by_year_notUR_vs_URg0_panels.tsv
#   - PNG/PDF: ossi_tier_stacked_pct_by_year_notUR_vs_URg0_panels_row.{png,pdf}

def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce").fillna(0) > 0
    low = s.astype(str).str.strip().str.lower()
    true_set  = {"1","true","yes","y"}
    false_set = {"","0","false","no","n","na","nan","none"}
    out = pd.Series(False, index=s.index)
    out = out.where(~low.isin(true_set), True)
    out = out.where(~low.isin(false_set), False)
    out = out.where(~(~low.isin(true_set | false_set) & (low != "")), True)
    return out

req_cols = [YEAR_COL, "ossi_tier", "under_request", "genuine"]
missing = [c for c in req_cols if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

d = df[[YEAR_COL, "ossi_tier", "under_request", "genuine"]].copy()
d[YEAR_COL] = pd.to_numeric(d[YEAR_COL], errors="coerce")
d = d.dropna(subset=[YEAR_COL, "ossi_tier"]).copy()
d["year"] = d[YEAR_COL].astype(int)

ur  = _to_bool(d["under_request"])
gen = _to_bool(d["genuine"])

# Define groups
d["group"] = pd.NA
d.loc[~ur, "group"] = "not_UR"
d.loc[ur & (~gen), "group"] = "UR_g0"
d = d.dropna(subset=["group"]).copy()

# Normalize tiers
tiers = ["Gold", "Silver", "Bronze", "None"]
d["ossi_tier"] = d["ossi_tier"].fillna("None")
d.loc[~d["ossi_tier"].isin(tiers), "ossi_tier"] = "None"

# Counts and percentages
ct = d.groupby(["year", "group", "ossi_tier"]).size().unstack(fill_value=0)
for t in tiers:
    if t not in ct.columns:
        ct[t] = 0
ct = ct[tiers]
tot = ct.sum(axis=1).rename("n_total")
pct = (ct.T / tot).T * 100.0

# Tidy table for export
pct_tidy = pct.reset_index().melt(id_vars=["year", "group"], value_vars=tiers,
                                  var_name="tier", value_name="pct")
pct_tsv = OUTDIR / "ossi_tier_stacked_pct_by_year_notUR_vs_URg0_panels.tsv"
pct_tidy.sort_values(["group", "year", "tier"]).to_csv(pct_tsv, sep="\t", index=False)
print(f"Wrote {pct_tsv}")

# -------- Plotting: two panels in one row --------
import numpy as np
import matplotlib.pyplot as plt

years = sorted(pct.index.get_level_values("year").unique().astype(int))
x = np.arange(1, len(years) + 1)

# Color-blind friendly palette
palette = {"Gold":"#E69F00", "Silver":"#7F7F7F", "Bronze":"#8C510A", "None":"#BDBDBD"}

fig_w = max(16, 0.6 * len(years))
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_w, 6), sharey=True)

group_order = ["not_UR", "UR_g0"]
titles = {"not_UR":"not_UR", "UR_g0":"UR & genuine=0"}

bar_width = 0.5  # wider bars since groups are split into separate panels

for ax, g in zip(axes, group_order):
    # Build stacked values for this group
    vals = {t: np.zeros(len(years)) for t in tiers}
    sub = pct.xs(g, level="group", drop_level=False)
    for i, y in enumerate(years):
        if (y, g) in sub.index:
            row = sub.loc[(y, g)]
            for t in tiers:
                vals[t][i] = float(row.get(t, 0.0))
    bottom = np.zeros(len(years))
    for t in tiers:
        ax.bar(x, vals[t], width=bar_width, bottom=bottom, color=palette[t], edgecolor="none")
        bottom += vals[t]

    ax.set_title(titles[g])
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=90, ha="center")
    ax.set_xlim(0.5, len(years) + 0.5)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

axes[0].set_ylabel("% of papers")
fig.suptitle("OSSI tiers — stacked % by year (left: not_UR, right: UR & genuine=0)", y=0.98)
axes[1].set_xlabel("Year", labelpad=12)
axes[0].set_xlabel("Year", labelpad=12)

# Tier legend once, above right
handles = [plt.Rectangle((0,0), 1, 1, color=palette[t]) for t in tiers]
fig.legend(handles, tiers, loc="upper right", ncol=4, frameon=False)

fig.tight_layout(rect=[0.04, 0.05, 0.98, 0.92])

png = OUTDIR / "ossi_tier_stacked_pct_by_year_notUR_vs_URg0_panels_row.png"
pdf = OUTDIR / "ossi_tier_stacked_pct_by_year_notUR_vs_URg0_panels_row.pdf"
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)
print(f"Wrote {png} and {pdf}")







# %%
# ITS (pooled with journal fixed effects) for ossi_score and each indicator (LPM).
# Outputs per outcome: TSV of coefficients + forest-style plot (level & slope change).
# Interrupted time series (pooled + journal fixed effects)
#
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BREAK_YEAR = 2016  # breakpoint at end-2016: post = year >= 2017
its_outdir = Path(OUTDIR) / "ITS"
its_outdir.mkdir(parents=True, exist_ok=True)

def to_bool01(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(int)
    low = s.astype(str).str.strip().str.lower()
    true_set = {"1","true","yes","y"}
    return low.isin(true_set).astype(int)

# Prepare modeling frame
d0 = df.copy()
d0["year"] = pd.to_numeric(d0[YEAR_COL], errors="coerce").astype("Int64")
d0 = d0.dropna(subset=["year"]).copy()
d0["year"] = d0["year"].astype(int)

# Time variables for segmented regression
d0["t"] = d0["year"] - d0["year"].min()
d0["post"] = (d0["year"] >= (BREAK_YEAR + 1)).astype(int)
d0["t_post"] = (d0["year"] - BREAK_YEAR) * d0["post"]

# Ensure numeric OSSI
d0["ossi_score"] = pd.to_numeric(d0.get("ossi_score", np.nan), errors="coerce")

# Build outcomes list
OUTCOMES = ["ossi_score"] + [c for c in INDICATOR_COLS if c in d0.columns]
for c in INDICATOR_COLS:
    if c in d0.columns:
        d0[c] = to_bool01(d0[c])

def fit_its_fe(outcome: str):
    # journal fixed effects via C(journal); cluster SEs by journal
    if "journal" not in d0.columns:
        raise SystemExit("Need a journal column for FE/clustered SEs.")
    f = f"{outcome} ~ t + post + t_post + C(journal)"
    m = smf.ols(f, data=d0).fit(cov_type="cluster", cov_kwds={"groups": d0["journal"]})
    ci = m.conf_int().rename(columns={0:"ci_lo",1:"ci_hi"})
    est = pd.DataFrame({"term": m.params.index, "coef": m.params.values,
                        "se": m.bse.values, "p": m.pvalues.values}).merge(ci, left_on="term", right_index=True)
    est["outcome"] = outcome
    return m, est

all_tables = []
for y in OUTCOMES:
    m, est = fit_its_fe(y)
    # keep the interpretable ITS terms
    keep = est[est["term"].isin(["t","post","t_post"])].copy()
    keep.to_csv(its_outdir / f"ITS_FE_{y}.tsv", sep="\t", index=False)
    all_tables.append(keep.assign(outcome=y))

its_table = pd.concat(all_tables, ignore_index=True)
its_table.to_csv(its_outdir / "ITS_FE_all_outcomes.tsv", sep="\t", index=False)
print(f"Wrote {its_outdir/'ITS_FE_all_outcomes.tsv'}")

# Forest-like panel: one subplot per outcome, show post (level change) and t_post (slope change)
fig, axes = plt.subplots(nrows=len(OUTCOMES), ncols=1, figsize=(8, 2.4*len(OUTCOMES)), sharex=True)
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

for ax, y in zip(axes, OUTCOMES):
    sub = its_table[its_table["outcome"]==y].set_index("term").reindex(["post","t_post"])
    ylabels = ["Level change (post)", "Slope change (t_post)"]
    x = sub["coef"].values
    lo = sub["ci_lo"].values
    hi = sub["ci_hi"].values
    ypos = np.arange(len(x), 0, -1)
    ax.hlines(ypos, lo, hi)
    ax.plot(x, ypos, "o")
    ax.axvline(0, linestyle=":", alpha=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels)
    ax.set_title(y)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

axes[-1].set_xlabel("Coefficient (95% CI)")
fig.tight_layout()
png = its_outdir / "ITS_FE_forest_all_outcomes.png"
pdf = its_outdir / "ITS_FE_forest_all_outcomes.pdf"
fig.savefig(png, dpi=300); fig.savefig(pdf); plt.close(fig)
print(f"Wrote {png} and {pdf}")





# %%
# ITS per journal for ossi_score only (level & slope change per journal).
# Outputs: per-journal TSV + forest plot of slope changes.
# 
# ITS per journal (who changed most?)
per_outdir = Path(OUTDIR) / "ITS_per_journal"
per_outdir.mkdir(parents=True, exist_ok=True)

JCOL = "journal"
if JCOL not in d0.columns:
    raise SystemExit("No 'journal' column.")

res = []
for jname, dJ in d0.dropna(subset=["ossi_score"]).groupby(JCOL):
    if dJ["ossi_score"].notna().sum() < 30:
        continue  # skip tiny samples
    m = smf.ols("ossi_score ~ t + post + t_post", data=dJ).fit(cov_type="HC1")
    ci = m.conf_int().rename(columns={0:"ci_lo",1:"ci_hi"})
    tab = pd.DataFrame({"term": m.params.index, "coef": m.params.values,
                        "se": m.bse.values, "p": m.pvalues.values}).merge(ci, left_on="term", right_index=True)
    row = tab.set_index("term").reindex(["post","t_post"])
    res.append({
        "journal": jname,
        "level_change": row.loc["post","coef"],
        "level_ci_lo": row.loc["post","ci_lo"],
        "level_ci_hi": row.loc["post","ci_hi"],
        "slope_change": row.loc["t_post","coef"],
        "slope_ci_lo": row.loc["t_post","ci_lo"],
        "slope_ci_hi": row.loc["t_post","ci_hi"],
        "n": len(dJ)
    })

its_j = pd.DataFrame(res).sort_values("slope_change", ascending=False)
its_j.to_csv(per_outdir / "ITS_per_journal_ossi_score.tsv", sep="\t", index=False)
print(f"Wrote {per_outdir/'ITS_per_journal_ossi_score.tsv'}")

# Forest of slope changes by journal (top/bottom 15 to keep it readable)
show_n = min(15, len(its_j))
top = its_j.head(show_n)
bot = its_j.tail(show_n)
plot_df = pd.concat([top, bot])
ypos = np.arange(len(plot_df), 0, -1)

fig, ax = plt.subplots(figsize=(9, 0.35*len(plot_df)+2))
ax.hlines(ypos, plot_df["slope_ci_lo"], plot_df["slope_ci_hi"])
ax.plot(plot_df["slope_change"], ypos, "o")
ax.axvline(0, linestyle=":", alpha=0.6)
ax.set_yticks(ypos)
ax.set_yticklabels(plot_df["journal"])
ax.set_xlabel("Slope change after 2016 (95% CI)")
ax.set_title("ITS slope change in ossi_score by journal (top/bottom)")
ax.grid(True, axis="x", linestyle=":", alpha=0.5)
fig.tight_layout()
png = per_outdir / "ITS_per_journal_ossi_score_forest.png"
pdf = per_outdir / "ITS_per_journal_ossi_score_forest.pdf"
fig.savefig(png, dpi=300); fig.savefig(pdf); plt.close(fig)
print(f"Wrote {png} and {pdf}")




# %%
# DiD with journal & year fixed effects, clustered SEs by journal.
# Outcome: ossi_score and each indicator (LPM for binaries).
# 
# Difference-in-Differences (treated vs control journals)
did_outdir = Path(OUTDIR) / "DiD"
did_outdir.mkdir(parents=True, exist_ok=True)

# EDIT THIS LIST for  "treated" STAR/strong-policy journals (case-insensitive match)
TREATED_JOURNALS = {
    "nature", "cell", "nature communications", "nature genetics", "science (new york, n.y.)"
}

d1 = d0.copy()
d1["journal_lc"] = d1.get("journal","").astype(str).str.lower()
d1["treated"] = d1["journal_lc"].isin(TREATED_JOURNALS).astype(int)

# Define post period; drop the transition year if needed
d1 = d1[(d1["year"] >= YEAR_MIN) & (d1["year"] <= YEAR_MAX)].copy()
d1["post"] = (d1["year"] >= (BREAK_YEAR + 1)).astype(int)

def fit_did(outcome: str):
    # y ~ treated*post + FE(journal) + FE(year), clustered by journal
    f = f"{outcome} ~ treated*post + C(journal) + C(year)"
    m = smf.ols(f, data=d1).fit(cov_type="cluster", cov_kwds={"groups": d1["journal"]})
    est = m.params.rename("coef").to_frame()
    ci  = m.conf_int().rename(columns={0:"ci_lo",1:"ci_hi"})
    p   = m.pvalues.rename("p").to_frame()
    tab = est.join(ci).join(p).reset_index().rename(columns={"index":"term"})
    return m, tab

did_tables = []
for y in OUTCOMES:
    m, tab = fit_did(y)
    tab.to_csv(did_outdir / f"DiD_{y}.tsv", sep="\t", index=False)
    # Pull the DiD effect term
    eff = tab[tab["term"] == "treated:post"].copy()
    eff["outcome"] = y
    did_tables.append(eff)

did_summary = pd.concat(did_tables, ignore_index=True)
did_summary.to_csv(did_outdir / "DiD_summary_treated_post.tsv", sep="\t", index=False)
print(f"Wrote {did_outdir/'DiD_summary_treated_post.tsv'}")

# Forest of DiD effects across outcomes
fig, ax = plt.subplots(figsize=(8, 2 + 0.5*len(did_summary)))
ypos = np.arange(len(did_summary), 0, -1)
ax.hlines(ypos, did_summary["ci_lo"], did_summary["ci_hi"])
ax.plot(did_summary["coef"], ypos, "o")
ax.axvline(0, linestyle=":", alpha=0.6)
ax.set_yticks(ypos)
ax.set_yticklabels(did_summary["outcome"])
ax.set_xlabel("DiD effect (treated × post), 95% CI")
ax.set_title("Difference-in-differences across outcomes")
ax.grid(True, axis="x", linestyle=":", alpha=0.5)
fig.tight_layout()
png = did_outdir / "DiD_forest.png"
pdf = did_outdir / "DiD_forest.pdf"
fig.savefig(png, dpi=300); fig.savefig(pdf); plt.close(fig)
print(f"Wrote {png} and {pdf}")




# %%
# Post-2016 slopes of ossi_score by journal (simple trend), ranked + lollipop plot.
# Post-2016 slope rankings (lollipop)
slope_outdir = Path(OUTDIR) / "SlopeRank"
slope_outdir.mkdir(parents=True, exist_ok=True)

post = d0[d0["year"] >= (BREAK_YEAR + 1)].dropna(subset=["ossi_score"]).copy()
JCOL = "journal"
rows = []
for jname, dJ in post.groupby(JCOL):
    if dJ["ossi_score"].notna().sum() < 10:
        continue
    m = smf.ols("ossi_score ~ year", data=dJ).fit(cov_type="HC1")
    ci = m.conf_int().rename(columns={0:"ci_lo",1:"ci_hi"})
    slope = m.params["year"]; lo = ci.loc["year","ci_lo"]; hi = ci.loc["year","ci_hi"]
    rows.append({"journal": jname, "slope": slope, "ci_lo": lo, "ci_hi": hi, "n": len(dJ)})

slopes = pd.DataFrame(rows).sort_values("slope", ascending=False)
slopes.to_csv(slope_outdir / "post2016_slope_by_journal.tsv", sep="\t", index=False)
print(f"Wrote {slope_outdir/'post2016_slope_by_journal.tsv'}")

# Lollipop
N = min(22, len(slopes))  # show top 22 by default (matches the journal set)
plot_df = slopes.head(N)
ypos = np.arange(N, 0, -1)

fig, ax = plt.subplots(figsize=(9, 0.4*N + 2))
ax.hlines(ypos, 0, plot_df["slope"])
ax.plot(plot_df["slope"], ypos, "o")
ax.axvline(0, linestyle=":", alpha=0.6)
ax.set_yticks(ypos)
ax.set_yticklabels(plot_df["journal"])
ax.set_xlabel("Post-2016 slope of ossi_score (95% CI not shown here)")
ax.set_title("Journals ranked by post-2016 improvement in OSSI score")
ax.grid(True, axis="x", linestyle=":", alpha=0.5)
fig.tight_layout()
png = slope_outdir / "post2016_slope_by_journal_lollipop.png"
pdf = slope_outdir / "post2016_slope_by_journal_lollipop.pdf"
fig.savefig(png, dpi=300); fig.savefig(pdf); plt.close(fig)
print(f"Wrote {png} and {pdf}")


# %%
# === G0 — UR summary (2010–2015): pie + bar, whole corpus =====================
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- config -------------------------------------------------------------------
#IN_TSV = "meta_openaccess_enriched.tsv"   # change if needed
#OUTDIR = Path("3.analyses/figures_ossi/summary")

IN_TSV = Path("/Users/benoit/work/under_request/2.data/meta_under_request_tagged_open_science.tsv")
OUTDIR = Path("/Users/benoit/work/under_request/3.analyses/figures_ossi/summary")




# %% === G0 — Whole-corpus UR summary (pie + stacked bar; 100%) =================
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- config -------------------------------------------------------------------
IN_TSV = Path("/Users/benoit/work/under_request/2.data/meta_under_request_tagged_open_science.tsv")
OUTDIR = Path("/Users/benoit/work/under_request/3.analyses/figures_ossi/summary")
OUTDIR.mkdir(parents=True, exist_ok=True)

# --- load & coerce ------------------------------------------------------------
df = pd.read_csv(IN_TSV, sep="\t", dtype=str)

required = {"under_request", "genuine"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {IN_TSV}: {', '.join(sorted(missing))}")

# Coerce to numeric (treat NA as 0)
df["under_request"] = pd.to_numeric(df["under_request"], errors="coerce").fillna(0).astype(int)
df["genuine"] = pd.to_numeric(df["genuine"], errors="coerce").fillna(0).astype(int)

# --- category mapping (mutually exclusive) ------------------------------------
def _ur_cat(row):
    if row["under_request"] == 1 and row["genuine"] == 1:
        return "UR & genuine==1"
    elif row["under_request"] == 1 and row["genuine"] == 0:
        return "UR & genuine==0"
    return "not_UR"

df["UR_category"] = df.apply(_ur_cat, axis=1)

order = ["not_UR", "UR & genuine==0", "UR & genuine==1"]
counts = df["UR_category"].value_counts().reindex(order, fill_value=0)
total = int(counts.sum())

if total == 0:
    raise ValueError("No rows found in the corpus to summarize (total == 0).")

fractions = counts / total
summary_df = (
    pd.DataFrame({"category": counts.index, "count": counts.values, "fraction": fractions.values})
)
summary_df.to_csv(OUTDIR / "ur_summary_all_counts.tsv", sep="\t", index=False)

# Build display labels with n=
labels_with_n = [f"{cat} (n={int(cnt)})" for cat, cnt in zip(counts.index, counts.values)]

# --- PIE chart ----------------------------------------------------------------
fig_p, ax_p = plt.subplots(figsize=(5.5, 5.5))

def _fmt_pct(p):
    # p is a percentage float (0–100)
    n = int(round(p * total / 100.0))
    return f"{p:.1f}%\n(n={n})"

ax_p.pie(
    counts.values,
    labels=labels_with_n,
    autopct=_fmt_pct,
    startangle=90
)
ax_p.axis("equal")
ax_p.set_title("Whole-corpus UR summary (pie)")

fig_p.tight_layout()
fig_p.savefig(OUTDIR / "ur_summary_all_pie.png", dpi=300)
fig_p.savefig(OUTDIR / "ur_summary_all_pie.pdf")
plt.close(fig_p)

# --- STACKED BAR (single bar; sums to 100%) -----------------------------------
fig_b, ax_b = plt.subplots(figsize=(6.5, 4.5))

# Heights as percentages
heights = (fractions.values * 100.0)

# Draw stacked bar at x=0
bottom = 0.0
bars = []
for h in heights:
    bar = ax_b.bar([0], [h], bottom=bottom)
    bars.append(bar)
    bottom += h

# X axis setup
ax_b.set_xlim(-0.6, 0.6)
ax_b.set_xticks([0])
ax_b.set_xticklabels(["All papers"])
ax_b.set_ylabel("Percent of papers")
ax_b.set_title("Whole-corpus UR summary (stacked bar)")

# Legend with n= labels
ax_b.legend([b[0] for b in bars], labels_with_n, loc="upper right", frameon=True)

# Annotate percentages on each segment
bottom = 0.0
for i, h in enumerate(heights):
    if h < 2.0:  # skip tiny labels to avoid clutter
        bottom += h
        continue
    ax_b.text(
        0, bottom + h/2.0, f"{h:.1f}%",
        ha="center", va="center"
    )
    bottom += h

# Y range and grid
ax_b.set_ylim(0, 100)
ax_b.yaxis.grid(True, linestyle=":", linewidth=0.7)

fig_b.tight_layout()
fig_b.savefig(OUTDIR / "ur_summary_all_stacked_bar.png", dpi=300)
fig_b.savefig(OUTDIR / "ur_summary_all_stacked_bar.pdf")
plt.close(fig_b)

print(f"[G0] Saved: {OUTDIR.resolve()}/ur_summary_all_counts.tsv, *_pie.(png|pdf), *_stacked_bar.(png|pdf)")






# %%
# CLI compatibility: if run this file as a script, running the whole file does all outputs.
if __name__ == "__main__":
    pass




