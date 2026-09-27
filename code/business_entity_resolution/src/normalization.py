"""Text normalization for business names and addresses.

Design goals (driven by the dataset blockers document + EDA):
  * Source 1 is always Latin script; Source 2/3 contain ~5-11% native-script
    names (Devanagari, Kannada, Tamil, ...). We transliterate EVERYTHING to a
    common Latin representation with `unidecode` so cross-script pairs become
    comparable. (unidecode is used only for text normalization; the ML *model*
    is a scikit-learn classifier, satisfying the MIT/Apache <=8B rule.)
  * We never collapse everything into one string. We keep several
    representations (token list, order-invariant core, squeezed form, char
    n-gram source) so different similarity signals can be computed and so
    over-aggressive normalization does not merge unrelated businesses.
  * No external data / no network. Pure local string ops.
"""
import re
import unicodedata

try:
    from unidecode import unidecode as _unidecode
except Exception:  # pragma: no cover - fallback if lib missing
    def _unidecode(s):
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

# Missing-value sentinels (blocker #7: literal NULL + empty fields).
NULL_TOKENS = {"", "null", "none", "nan", "na", "n/a", "-", "--", "..."}

# Legal-suffix tokens across the observed geographies (US / India / France).
# Removed from the "core" name so suffix variation (blocker #5) does not block.
LEGAL_SUFFIXES = {
    # english / generic
    "private", "pvt", "pvtltd", "limited", "ltd", "ltda", "llp", "llc", "lc",
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
    "plc", "pte", "pty", "gmbh", "ag", "bv", "nv", "srl", "sl", "sa", "sas",
    "sasu", "sarl", "eurl", "sci", "scp", "snc", "selarl", "sccv", "group",
    "groupe", "holding", "holdings", "enterprise", "enterprises", "ventures",
    "trading", "international", "intl", "global", "associates", "assoc",
    "and", "the", "of",
}

# Common address abbreviations -> canonical (blocker #9). Kept small & safe.
ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bldg": "building", "hwy": "highway", "ln": "lane",
    "dr": "drive", "ct": "court", "sq": "square", "apt": "apartment",
    "fl": "floor", "flr": "floor", "no": "number", "opp": "opposite",
    "near": "near", "cross": "cross", "main": "main", "sec": "sector",
    "ph": "phase", "nr": "near", "hno": "number", "hn": "number",
    "gali": "gali", "marg": "marg", "nagar": "nagar", "colony": "colony",
    "r": "rue", "bd": "boulevard", "imp": "impasse", "all": "allee",
}

_WORD_RE = re.compile(r"[a-z0-9]+")
_REPEAT_RE = re.compile(r"(.)\1+")
_DOMAIN_RE = re.compile(r"\.(com|net|org|in|co|io|biz|fr|us)\b")


def _pre(s):
    """NFKC-normalize, transliterate to Latin, lowercase. Returns '' for nulls."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = _unidecode(s)
    s = s.lower()
    if s.strip() in NULL_TOKENS:
        return ""
    return s


def squeeze(s):
    """Collapse runs of the same char: 'limittedd' -> 'limited', 'raam'->'ram'.

    This aligns noisy phonetic transliterations of Indic scripts with their
    English counterparts (blocker #12). Applied as an *extra* representation,
    never the only one.
    """
    return _REPEAT_RE.sub(r"\1", s)


def name_tokens(raw, drop_suffix=True):
    """Significant name tokens: transliterated, lowercased, suffix-stripped."""
    s = _pre(raw)
    if not s:
        return []
    s = s.replace("&", " and ")
    s = _DOMAIN_RE.sub(" ", s)
    toks = _WORD_RE.findall(s)
    if drop_suffix:
        core = [t for t in toks if t not in LEGAL_SUFFIXES]
        # keep the raw tokens if suffix-stripping removed everything
        toks = core if core else toks
    return toks


def name_reprs(raw):
    """Return a dict of several name representations for one raw string."""
    toks = name_tokens(raw)
    toks_sorted = sorted(set(toks))
    join = "".join(toks)
    sqz = squeeze(join)
    return {
        "tokens": toks,
        "tokset": frozenset(toks),
        "core": " ".join(toks_sorted),      # order-invariant
        "join": join,                        # concatenated, order-preserving
        "sqz": sqz,                          # squeezed (translit-robust)
        "sorted_join": "".join(toks_sorted),
    }


# Postal codes: US 5-digit ZIP, India 6-digit PIN, France 5-digit. Generic
# 5-6 digit token (open-set friendly; not hard-coded to a country).
_POSTAL_RE = re.compile(r"\b(\d{5,6})\b")
_NUM_RE = re.compile(r"\d+")


def addr_tokens(raw):
    """Significant address tokens with light abbreviation expansion."""
    s = _pre(raw)
    if not s:
        return []
    s = s.replace("&", " and ")
    toks = _WORD_RE.findall(s)
    return [ADDR_ABBREV.get(t, t) for t in toks]


def addr_reprs(raw):
    """Several address representations + extracted components."""
    s = _pre(raw)
    toks = addr_tokens(raw)
    tokset = frozenset(toks)
    postals = frozenset(_POSTAL_RE.findall(s))
    nums = frozenset(n for n in _NUM_RE.findall(s) if len(n) < 5)
    words = frozenset(t for t in toks if t.isalpha() and len(t) >= 3)
    return {
        "tokens": toks,
        "tokset": tokset,
        "core": " ".join(sorted(tokset)),
        "join": "".join(toks),
        "postals": postals,
        "nums": nums,
        "words": words,
        "is_null": len(toks) == 0,
    }
