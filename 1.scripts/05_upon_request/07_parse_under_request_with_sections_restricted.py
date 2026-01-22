#!/usr/bin/env python3
# --------------------------------------------------
# 07_parse_under_request_with_sections_restricted.py
# --------------------------------------------------
# Parse PMC JATS XML for "data available upon request" variants.
# - Processes ALL rows (no ok_analysis filter)
# - Namespace-agnostic (handles JATS namespaces)
# - Uses sentence-aware extraction to store the whole sentence
# - Preserves all original columns from meta_openaccess_enriched2.csv
# - Adds/overwrites: under_request (0/1), matching (exact), phrase (full sentence), section, restricted
# - On XML parse error: matching="xml_error", under_request=0
# --------------------------------------------------

import os
import re
import html
import pandas as pd
import xml.etree.ElementTree as ET

# ------------ CONFIG ------------
META_FILE = "2.data/meta_openaccess_enriched2.csv"
META_SEP = ";"             # CHANGE if needed: "\t" for TSV, ";" for semicolon CSV
XML_DIR = "3.xml"
OUT_FILE = "2.data/meta_under_request.tsv"

FLUSH_EVERY = 1000         # periodic writes to avoid data loss

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ------------ REGEX ------------
# Exact/common phrasings (strict)
EXACT_PATTERNS = [
    r"\bdata (?:is|are)? available upon request\b",
    r"\bavailable (?:upon|by|under) request\b",
    r"\bavailable (?:on|upon|under) reasonable request\b",
    r"\bdatasets? (?:is|are)? available upon request\b",
    r"\bprovided (?:upon|on) request\b",
    r"\bavailable from the corresponding author (?:on|upon) request\b",
    r"\bavailable from the author (?:on|upon) request\b",
    r"\bcan be obtained (?:upon|on) request\b",
    r"\bon reasonable request\b",
    r"\bupon reasonable request\b",
    r"\bunder reasonable request\b",
]

# Flexible phrasings: allow up to ~3 words between key tokens
FLEX_PATTERNS = [
    r"\bavailable\b(?:\W+\w+){0,3}?\b(upon|on|under)\b(?:\W+\w+){0,3}?\brequest\b",
    r"\bprovided\b(?:\W+\w+){0,3}?\b(upon|on)\b(?:\W+\w+){0,3}?\brequest\b",
    r"\b(obtain(?:ed)?|access(?:ed)?)\b(?:\W+\w+){0,3}?\b(upon|on)\b(?:\W+\w+){0,3}?\brequest\b",
]

RESTRICTED_PATTERNS = [
    r"due to (?:privacy|ethical|confidentiality|legal|regulation)s?",
    r"restricted access",
    r"cannot be shared",
    r"not publicly available",
    r"available in aggregate form only",
    r"available upon request (?:and approval|to qualified researchers)",
]

EXACT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in EXACT_PATTERNS]
FLEX_PATTERNS  = [re.compile(p, re.IGNORECASE) for p in FLEX_PATTERNS]
RESTRICTED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in RESTRICTED_PATTERNS]

# ------------ HELPERS ------------
def tag_name(elem):
    """Return local tag name without namespace."""
    t = elem.tag
    if isinstance(t, str) and t.startswith("{"):
        return t.split("}", 1)[1]
    return t

def extract_text(elem) -> str:
    """Concatenate all text (including descendants) into a single string, lightly normalized."""
    raw = " ".join(elem.itertext())
    text = html.unescape(raw)
    # optional: remove pure numeric citation blocks like [1], [1,2], [12–15]
    text = re.sub(r"\[(?:\d+|[\d,\s–-]+)\]", "", text)
    # normalize whitespace only (avoid destructive cleaning)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sentence_split(text: str):
    """
    Split text into sentences in a light-weight manner.
    Avoids heavy dependencies; decent for scientific prose.
    """
    if not text:
        return []
    # Split on end punctuation followed by whitespace and an uppercase or '('
    # This keeps abbreviations like "e.g." more often intact.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\(])", text)
    # If splitting produced nothing useful, fallback to entire text
    if not parts:
        return [text]
    return parts

def sentence_window(text: str, start_idx: int) -> str:
    """Fallback char-window if sentence splitting fails to capture context."""
    left = text.rfind(".", 0, start_idx)
    if left == -1:
        left = 0
    else:
        left = left + 1
    right_candidates = [text.find(sym, start_idx) for sym in [".", "!", "?"]]
    right_candidates = [x for x in right_candidates if x != -1]
    right = min(right_candidates) if right_candidates else len(text)
    snippet = text[left:right].strip()
    if len(snippet) < 20:
        lo = max(0, start_idx - 200)
        hi = min(len(text), start_idx + 200)
        snippet = text[lo:hi].strip()
    return snippet

def detect_phrase_in_text(text: str):
    """
    Return (under_request, match_exact, match_sentence, restricted) for a text block.
    - under_request: 1 if any exact or flexible pattern matches; else 0
    - match_exact: exact regex substring (group 0)
    - match_sentence: full sentence containing the match (or window fallback)
    - restricted: 1 if restricted pattern appears in the sentence; else 0
    """
    if not text:
        return 0, "", "", 0

    sents = sentence_split(text)
    if not sents:
        sents = [text]

    # First pass: exact patterns
    for sent in sents:
        for pat in EXACT_PATTERNS:
            m = pat.search(sent)
            if m:
                match_exact = m.group(0).strip()
                match_sentence = sent.strip() if sent.strip() else sentence_window(text, m.start())
                restricted = int(any(r.search(match_sentence) for r in RESTRICTED_PATTERNS))
                return 1, match_exact, match_sentence, restricted

    # Second pass: flexible patterns
    for sent in sents:
        for pat in FLEX_PATTERNS:
            m = pat.search(sent)
            if m:
                match_exact = m.group(0).strip()
                match_sentence = sent.strip() if sent.strip() else sentence_window(text, m.start())
                restricted = int(any(r.search(match_sentence) for r in RESTRICTED_PATTERNS))
                return 1, match_exact, match_sentence, restricted

    return 0, "", "", 0

def iter_candidate_blocks(root):
    """
    Yield (label, element) for likely locations:
      - sec (label from sec-type or title)
      - abstract
      - ack / acknowledgments
      - fn (footnotes)
      - supplementary-material
    """
    # Sections
    for sec in root.iter():
        if tag_name(sec) == "sec":
            sec_type = sec.get("sec-type", "") or ""
            title = None
            for child in sec:
                if tag_name(child) == "title":
                    title = extract_text(child)
                    if title:
                        break
            label = sec_type or (title or "sec")
            yield (label, sec)

    # Abstracts
    for abs_el in root.iter():
        if tag_name(abs_el) == "abstract":
            yield ("abstract", abs_el)

    # Acknowledgments
    for ack in root.iter():
        if tag_name(ack) in ("ack", "acknowledgments"):
            yield ("acknowledgments", ack)

    # Footnotes
    for fn in root.iter():
        if tag_name(fn) == "fn":
            yield ("footnote", fn)

    # Supplementary
    for sup in root.iter():
        if tag_name(sup) == "supplementary-material":
            yield ("supplementary-material", sup)

# ------------ MAIN ------------
print(f"Loading corpus from {META_FILE}")
meta = pd.read_csv(META_FILE, sep=META_SEP)
print(f"Loaded {len(meta)} total entries")

# Ensure output columns exist
for c in ["under_request", "matching", "phrase", "section", "restricted"]:
    if c not in meta.columns:
        meta[c] = ""

total = len(meta)
processed = 0

for idx, row in meta.iterrows():
    # Defaults for every row
    under_request = 0
    matching = ""
    phrase = ""
    section = ""
    restricted = 0

    pmcid_raw = str(row.get("pmcid", ""))
    pmcid = pmcid_raw.replace("PMC", "")
    xml_path = os.path.join(XML_DIR, f"{pmcid}.xml")

    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            hit_found = False
            # Scan likely blocks first
            for label, elem in iter_candidate_blocks(root):
                txt = extract_text(elem)
                u, m_exact, m_sent, r = detect_phrase_in_text(txt)
                if u:
                    under_request, matching, phrase, section, restricted = u, m_exact, m_sent, label, r
                    hit_found = True
                    break

            # Fallback: scan entire article if no hit in blocks
            if not hit_found:
                whole = extract_text(root)
                u, m_exact, m_sent, r = detect_phrase_in_text(whole)
                if u:
                    under_request, matching, phrase, section, restricted = u, m_exact, m_sent, "whole-article", r

        except ET.ParseError:
            # XML parse error: mark explicitly
            matching = "xml_error"
            under_request = 0
            phrase = ""
            section = ""
            restricted = 0
    else:
        # No XML file found: still produce a deterministic row
        under_request = 0

    # Guarantee 0/1 and write back
    meta.loc[idx, "under_request"] = int(under_request)
    meta.loc[idx, "matching"] = matching
    meta.loc[idx, "phrase"] = phrase
    meta.loc[idx, "section"] = section
    meta.loc[idx, "restricted"] = int(restricted)

    processed += 1
    if processed % FLUSH_EVERY == 0:
        meta.to_csv(OUT_FILE, sep="\t", index=False)
        print(f"Processed {processed} / {total}")

# Final write
meta.to_csv(OUT_FILE, sep="\t", index=False)
print(f"Done. Parsed {total} entries; full table written to {OUT_FILE}")
