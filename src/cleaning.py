"""Deterministic, unfitted cleaning functions. Source values remain auditable."""
import html
import re
import unicodedata

NEGATIONS = {"no", "not", "nor", "never", "without"}
TFIDF_STOPWORDS = frozenset("a an the and or of to in on at by for from with as is are was were be been being it its this that these those we our us they their them he she his her you your".split())
BOUNDARY = re.compile(
    r"^[ \t]*item[ \t]+(?:7\s*a[ \t]*[.\-:]?[ \t\r\n]*quantitative\s+and\s+qualitative|8[ \t]*[.\-:]?[ \t\r\n]*financial\s+statements)",
    re.I | re.M,
)

def normalize_identifier(value, width=None):
    """Allow digit identifiers and spreadsheet-style .0, reject other strings."""
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in {"", "nan", "none", "<na>", "nat"}:
        return None
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    if not re.fullmatch(r"\d+", s):
        return None
    s = s.lstrip("0") or "0"
    if s == "0" or (width and len(s) > width):
        return None
    return s.zfill(width) if width else s

def normalize_accession(value):
    s = re.sub(r"-", "", str(value).strip())
    return s if re.fullmatch(r"\d{18}", s) else None

def tokenize_tfidf(text):
    """Optional TF-IDF-only view; never feed this stripped view to encoders."""
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*", text.lower())
    return [t for t in tokens if t not in TFIDF_STOPWORDS or t in NEGATIONS]

def clean_text(raw, min_chars=1000):
    normalized = unicodedata.normalize("NFC", html.unescape(raw))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", "").replace("\u200b", "").replace("\ufeff", "")
    # Remove only recognized tags. Preserve < and > in financial comparisons.
    normalized = re.sub(r"</?(?:p|div|span|br|font|table|tr|td|th|html|body)\b[^>]*>", " ", normalized, flags=re.I)
    original_normalized_chars = len(normalized)
    matches = list(BOUNDARY.finditer(normalized))
    cut = next((m for m in matches if m.start() >= 2500), None)
    cut_chars = 0
    boundary_excerpt = ""
    if cut:
        boundary_excerpt = normalized[max(0, cut.start()-120):cut.start()+240]
        cut_chars = len(normalized)-cut.start()
        normalized = normalized[:cut.start()]
    lines=[]
    removed_noise_lines=0
    for line in normalized.splitlines():
        if re.fullmatch(r"\s*(?:table\s+of\s+contents|page\s+\d+(?:\s+of\s+\d+)?)\s*", line, flags=re.I):
            removed_noise_lines += 1
            continue
        lines.append(line)
    # Keep layout blocks; they are not claimed to be linguistic paragraphs.
    blocks=[]
    seen_long=set()
    duplicate_blocks=0
    for block in re.split(r"\n[ \t]*\n+", "\n".join(lines)):
        block=" ".join(block.split())
        if not block:
            continue
        if len(block)>=300:
            if block in seen_long:
                duplicate_blocks+=1
                continue
            seen_long.add(block)
        blocks.append(block)
    text="\n\n".join(blocks)
    issues=[]
    if len(text)<min_chars:
        issues.append("short_text_review")
    if "\ufffd" in text:
        issues.append("encoding_replacement_review")
    if any(m.start()<2500 for m in matches):
        issues.append("early_boundary_heading_review")
    if cut and cut.start()/max(original_normalized_chars,1)<0.1:
        issues.append("large_boundary_trim_review")
    if len(text)/max(len(raw),1)<0.2:
        issues.append("large_cleaning_change_review")
    return text, {
        "raw_characters":len(raw), "clean_characters":len(text),
        "layout_block_count":len(blocks), "removed_noise_lines":removed_noise_lines,
        "removed_duplicate_long_blocks":duplicate_blocks,
        "boundary_trim_characters":cut_chars,
        "boundary_excerpt":boundary_excerpt,
        "quality_flags":"|".join(issues),
    }
