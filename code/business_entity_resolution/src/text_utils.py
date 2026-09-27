"""
Text normalization utilities for business names and addresses.

Design notes
------------
- We normalize aggressively for NAME (legal-suffix / abbreviation noise is the
  dominant noise source per the problem statement) and moderately for
  ADDRESS (abbreviation expansion + punctuation/whitespace cleanup), while
  keeping numeric tokens (house/PIN numbers) intact since they carry signal.
- Nothing here hard-codes country values -- these are generic string ops
  that apply the same way regardless of which country label appears.
"""
import re
import unicodedata
import regex as _uregex

# --- Legal-suffix / common business-name abbreviation expansions ---------
NAME_REPLACEMENTS = {
    r"\bcorp\b": "corporation",
    r"\bco\b": "company",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bp\.l\.c\b": "public limited company",
    r"\bplc\b": "public limited company",
    r"\binc\b": "incorporated",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\bmfg\b": "manufacturing",
    r"\bassoc\b": "associates",
    r"\bintl\b": "international",
    r"\bind\b": "industries",
    r"\bgrp\b": "group",
    r"\bbros\b": "brothers",
    r"&": " and ",
}

# --- Common address abbreviation expansions --------------------------------
ADDR_REPLACEMENTS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bdr\b": "drive",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bnr\b": "near",
    r"\bopp\b": "opposite",
    r"\bsec\b": "sector",
    r"\bcolny\b": "colony",
    r"\bsoc\b": "society",
    r"\bindl\b": "industrial",
    r"\best\b": "estate",
    r"&": " and ",
}

STOPWORDS = {"the", "a", "an", "of", "and", "for", "in", "at", "to"}


def _strip_diacritics(text: str) -> str:
    """Fold transliteration/accent variants to plain ASCII where possible."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _base_clean(text: str) -> str:
    """
    Lowercase + strip punctuation while PRESERVING non-Latin scripts intact.

    IMPORTANT BUG FIX: a plain `re.sub(r"[^\\w\\s&]", ...)` looks safe but is
    NOT Unicode-complete in this environment -- stdlib `\\w` fails to include
    Unicode "Mc" (spacing combining mark) characters, which is exactly the
    category Devanagari (and other Indic-script) vowel signs/matras belong
    to. On real data (India-labeled records are frequently in Devanagari,
    not just romanized), that older approach silently deleted every matra,
    turning "राम मार्केटिंग" into "र म म र क ट ग" -- consonant skeletons with
    no vowel information, which destroys any hope of matching. We instead
    filter by explicit Unicode category (keep Letter/Mark/Number + whitespace
    + '&', via the `regex` package's \\p{} property classes) so combining
    vowel signs in any script survive.
    """
    if text is None:
        return ""
    text = str(text)
    text = _strip_diacritics(text)
    text = text.lower()
    text = text.replace("_", " ")
    text = _uregex.sub(r"[^\p{L}\p{M}\p{N}\s&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(text: str) -> str:
    text = _base_clean(text)
    for pattern, repl in NAME_REPLACEMENTS.items():
        text = re.sub(pattern, repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_address(text: str) -> str:
    text = _base_clean(text)
    for pattern, repl in ADDR_REPLACEMENTS.items():
        text = re.sub(pattern, repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_series_name(s):
    """Vectorized version of normalize_name for a pandas Series (fast at millions-of-rows scale)."""
    s = s.fillna("").astype(str)
    s = s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    s = s.str.lower().str.replace("_", " ", regex=False)
    s = s.str.replace(r"[^\w\s&]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    for pattern, repl in NAME_REPLACEMENTS.items():
        s = s.str.replace(pattern, repl, regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def normalize_series_address(s):
    """Vectorized version of normalize_address for a pandas Series."""
    s = s.fillna("").astype(str)
    s = s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    s = s.str.lower().str.replace("_", " ", regex=False)
    s = s.str.replace(r"[^\w\s&]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    for pattern, repl in ADDR_REPLACEMENTS.items():
        s = s.str.replace(pattern, repl, regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def tokenize(text: str, drop_stopwords: bool = True, min_len: int = 1):
    if not text:
        return []
    toks = text.split(" ")
    toks = [t for t in toks if len(t) >= min_len]
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def char_ngrams(text: str, n: int = 3):
    text = text.replace(" ", "")
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0
