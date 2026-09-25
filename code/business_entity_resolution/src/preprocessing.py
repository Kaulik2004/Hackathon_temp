"""Memory-efficient, Unicode-safe text normalization for business records.

Design constraints (see Documentation_template.md / README.md for rationale):

* Source 2/3 contain non-Latin-script business names (Devanagari, Kannada, ...)
  for India records. Any byte-range regex such as ``[^a-z0-9\\s]`` silently
  blanks those rows. Every regex here is Unicode-aware (Python 3 ``str``
  patterns default to Unicode ``\\w``/``\\W`` semantics -- we never opt out of
  that), and normalization never transliterates or ASCII-folds text.
* ``country`` is treated as an open string set: it is normalized for
  whitespace only and is never mapped, lower-cased against a fixed table, or
  compared against a hardcoded whitelist anywhere in this module.
* All TSV reads use ``dtype=str, keep_default_na=False, na_filter=False`` so
  an empty address field is the empty string ``""``, never ``NaN`` (NaN would
  silently become the string ``"nan"`` under careless casting and would break
  every regex/string method downstream).
* Large source files (up to ~500MB) are streamed in chunks and written to
  Parquet via ``pyarrow.parquet.ParquetWriter`` so peak memory is bounded by
  ``chunksize`` regardless of file size, and downstream stages (blocking,
  feature engineering) read the typed, pre-normalized Parquet instead of
  re-parsing the raw TSV.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterator, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DELIM = "\t"

# ---------------------------------------------------------------------------
# Normalization tables
# ---------------------------------------------------------------------------

# ASCII-only legal-suffix normalization. Matched on whitespace/punctuation
# tokens consisting solely of ASCII letters/periods, so non-Latin scripts
# never enter this table and pass through untouched.
_LEGAL_SUFFIX_MAP = {
    "pvt": "private",
    "pvt.": "private",
    "ltd": "limited",
    "ltd.": "limited",
    "llc": "llc",
    "l.l.c.": "llc",
    "l.l.c": "llc",
    "corp": "corporation",
    "corp.": "corporation",
    "inc": "incorporated",
    "inc.": "incorporated",
    "co": "company",
    "co.": "company",
    "llp": "llp",
    "l.l.p.": "llp",
    "sarl": "sarl",
    "s.a.r.l.": "sarl",
}

# Address component abbreviations, canonicalized toward the shorter form
# (matches common data-entry conventions and keeps token-overlap blocking
# keys tighter).
_ADDR_ABBR_MAP = {
    "street": "st",
    "st.": "st",
    "road": "rd",
    "rd.": "rd",
    "avenue": "ave",
    "ave.": "ave",
    "boulevard": "blvd",
    "blvd.": "blvd",
    "drive": "dr",
    "dr.": "dr",
    "lane": "ln",
    "ln.": "ln",
    "court": "ct",
    "ct.": "ct",
    "apartment": "apt",
    "apt.": "apt",
}

_DBA_PATTERN = re.compile(r"(?i)\bdba\b[:\s]*")
_FORMERLY_PATTERN = re.compile(r"(?i)\bformerly\b[:\s]*")
_AMP_PATTERN = re.compile(r"\s*&\s*")
_DECORATIVE_EDGE_PATTERN = re.compile(r"^[\W_]+|[\W_]+$", flags=re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+", flags=re.UNICODE)
# Word tokenization deliberately does NOT use a `\w+` findall: Python's `\w`
# class excludes Unicode combining marks (category Mn/Mc), which Devanagari/
# Kannada vowel signs (matras) and conjunct-forming marks rely on. A `\w+`
# scan would terminate the run at every matra and shatter one word into a
# sequence of single-consonant tokens (silently -- no error, no blanking --
# but it destroys token-level signal for these scripts). Instead we split on
# an explicit set of separator characters (whitespace + common punctuation),
# which works correctly for every script since it never inspects individual
# character categories -- any character that isn't an explicit separator,
# combining marks included, stays part of its token.
_TOKEN_SEPARATOR_PATTERN = re.compile(r"[\s.,;:!?\"'`|()\[\]{}<>]+", flags=re.UNICODE)
_ASCII_TOKEN_PATTERN = re.compile(r"^[A-Za-z.]+$")
_LEADING_NUMBER_PATTERN = re.compile(r"^\s*(\d[\w-]*)")
_POSTAL_CODE_PATTERN = re.compile(r"\b\d{4,6}\b")


# ---------------------------------------------------------------------------
# Core text normalization
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    """Unicode-safe base normalization: NFKC -> casefold -> collapse whitespace.

    ``casefold`` (not ``lower``) is the Unicode-correct case-normalization
    function; it is a no-op on scripts with no case concept (Devanagari,
    Kannada, CJK, ...), so non-Latin text is left untouched rather than
    corrupted. Nothing here transliterates or drops characters outside the
    ASCII range.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = _AMP_PATTERN.sub(" and ", s)
    s = _DECORATIVE_EDGE_PATTERN.sub("", s)
    s = _WHITESPACE_PATTERN.sub(" ", s).strip()
    return s


_TRAILING_PUNCT_PATTERN = re.compile(r"[^\w]+$", flags=re.UNICODE)
_LEADING_PUNCT_PATTERN = re.compile(r"^[^\w]+", flags=re.UNICODE)


def _split_edge_punct(tok: str) -> tuple[str, str, str]:
    """Split a token into (leading_punct, core, trailing_punct), e.g.
    "street," -> ("", "street", ","). Table lookups match on ``core`` only,
    so trailing commas (ubiquitous in comma-separated address components)
    don't block a match, while the punctuation is preserved in the output.
    """
    lead_m = _LEADING_PUNCT_PATTERN.match(tok)
    lead = lead_m.group(0) if lead_m else ""
    rest = tok[len(lead) :]
    trail_m = _TRAILING_PUNCT_PATTERN.search(rest)
    trail = trail_m.group(0) if trail_m else ""
    core = rest[: len(rest) - len(trail)] if trail else rest
    return lead, core, trail


def _apply_token_map(tokens: list[str], table: dict[str, str], strip: bool = False) -> list[str]:
    out = []
    for tok in tokens:
        lead, core, trail = _split_edge_punct(tok)
        key = core.lower()
        if _ASCII_TOKEN_PATTERN.match(key) and key in table:
            if strip:
                continue
            out.append(f"{lead}{table[key]}{trail}")
        else:
            out.append(tok)
    return out


def _apply_legal_suffix_map(tokens: list[str], strip: bool) -> list[str]:
    return _apply_token_map(tokens, _LEGAL_SUFFIX_MAP, strip=strip)


def normalize_business_name(raw: str) -> dict:
    """Return normalized name variants used for blocking keys and features.

    Keys:
        name_norm             -- base-normalized full name (suffixes kept,
                                  but canonicalized, e.g. "pvt" -> "private")
        name_suffix_stripped  -- name_norm with legal-suffix tokens removed
        name_primary          -- text before a DBA/"formerly" marker (or the
                                  full name if no marker is present)
        name_alias            -- text after a DBA/"formerly" marker, or ""
    """
    if not raw:
        return {
            "name_norm": "",
            "name_suffix_stripped": "",
            "name_primary": "",
            "name_alias": "",
        }

    primary_raw, alias_raw = raw, ""
    m = _DBA_PATTERN.search(raw) or _FORMERLY_PATTERN.search(raw)
    if m:
        primary_raw = raw[: m.start()].strip()
        alias_raw = raw[m.end() :].strip()
        if not primary_raw:
            # e.g. "DBA: Foo" with nothing before the marker -- treat the
            # whole string as primary rather than losing it.
            primary_raw, alias_raw = raw, ""

    base = normalize_text(raw)
    tokens = base.split(" ") if base else []
    name_norm = " ".join(_apply_legal_suffix_map(tokens, strip=False))
    name_suffix_stripped = " ".join(_apply_legal_suffix_map(tokens, strip=True))

    return {
        "name_norm": name_norm,
        "name_suffix_stripped": name_suffix_stripped or name_norm,
        "name_primary": normalize_text(primary_raw),
        "name_alias": normalize_text(alias_raw),
    }


def normalize_address(raw: str) -> dict:
    """Return normalized address text plus extracted house-number/postal code.

    Extraction is deliberately country-agnostic (no US/India-specific
    formats hardcoded): a leading numeric token is treated as a house
    number, and any standalone 4-6 digit run is treated as a candidate
    postal code. These are sparse, high-precision signals fed to
    feature_engineering.py, not blocking keys on their own merit alone.
    """
    if not raw:
        return {"addr_norm": "", "addr_house_number": "", "addr_postal_code": ""}

    base = normalize_text(raw)
    tokens = base.split(" ") if base else []
    normalized_tokens = _apply_token_map(tokens, _ADDR_ABBR_MAP)
    addr_norm = " ".join(normalized_tokens)

    house_match = _LEADING_NUMBER_PATTERN.match(raw.strip())
    house_number = house_match.group(1) if house_match else ""

    postal_matches = _POSTAL_CODE_PATTERN.findall(raw)
    postal_code = postal_matches[-1] if postal_matches else ""

    return {
        "addr_norm": addr_norm,
        "addr_house_number": house_number,
        "addr_postal_code": postal_code,
    }


def tokenize(s: str) -> list[str]:
    """Unicode-safe word tokenization (never a byte-range character class;
    never a `\\w+` scan, which would shatter Devanagari/Kannada text -- see
    the comment on ``_TOKEN_SEPARATOR_PATTERN`` above)."""
    if not s:
        return []
    return [p for p in _TOKEN_SEPARATOR_PATTERN.split(s) if p]


# ---------------------------------------------------------------------------
# Chunked I/O
# ---------------------------------------------------------------------------

_READ_KWARGS = dict(
    sep=DELIM,
    dtype=str,
    keep_default_na=False,
    na_filter=False,
    encoding="utf-8",
)


def load_source_chunked(path: Path, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    """Yield chunks of a raw source TSV with safe dtypes.

    ``dtype=str`` is mandatory: entity_id / business_name / business_address
    must never be numeric-inferred (a purely numeric-looking business name
    would otherwise be mangled). ``keep_default_na=False`` + ``na_filter=False``
    keep an empty address as ``""`` rather than ``NaN``.
    """
    for chunk in pd.read_csv(path, chunksize=chunksize, **_READ_KWARGS):
        yield chunk


def read_source_full(path: Path) -> pd.DataFrame:
    """Read a whole source TSV at once (safe dtypes). Use only for files that
    are known to fit comfortably in memory (e.g. id-only reads)."""
    return pd.read_csv(path, **_READ_KWARGS)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized columns to a raw source dataframe.

    Adds: name_norm, name_suffix_stripped, name_primary, name_alias,
    addr_norm, addr_house_number, addr_postal_code. Raw columns
    (entity_id, business_name, business_address, country) are preserved
    untouched for audit; ``country`` is never modified.
    """
    name_parts = df["business_name"].map(normalize_business_name)
    addr_parts = df["business_address"].map(normalize_address)

    name_df = pd.DataFrame(list(name_parts), index=df.index)
    addr_df = pd.DataFrame(list(addr_parts), index=df.index)

    out = pd.concat([df, name_df, addr_df], axis=1)
    return out


def preprocess_source_file(
    src_path: Path, out_path: Path, chunksize: int = 200_000
) -> Path:
    """Stream-normalize one raw source TSV into a Parquet file.

    Runs in O(chunksize) memory regardless of input file size.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    n_rows = 0
    try:
        for chunk in load_source_chunked(src_path, chunksize=chunksize):
            norm = normalize_dataframe(chunk)
            table = pa.Table.from_pandas(norm, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
            n_rows += len(norm)
            logger.info("preprocessed %s: %d rows so far", src_path.name, n_rows)
    finally:
        if writer is not None:
            writer.close()
    logger.info("wrote %s (%d rows)", out_path, n_rows)
    return out_path


def preprocess_all(
    dataset_dir: Path,
    artifacts_dir: Path,
    split: Literal["train", "test"],
    chunksize: int = 200_000,
    force: bool = False,
) -> dict[str, Path]:
    """Preprocess every source file for a split (train or test).

    Returns a mapping like {"source1": Path(...), "source2": Path(...), ...}
    (plus "ground_truth" for the train split, copied through with safe dtypes
    for reuse in model.py without re-parsing the raw TSV).
    """
    split_dir = dataset_dir / split
    out_dir = artifacts_dir / "preprocessed" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}
    for source_name in ("source1", "source2", "source3"):
        src_path = split_dir / f"{split}_{source_name}.tsv"
        out_path = out_dir / f"{source_name}.parquet"
        if out_path.exists() and not force:
            logger.info("reusing cached %s", out_path)
        else:
            preprocess_source_file(src_path, out_path, chunksize=chunksize)
        results[source_name] = out_path

    if split == "train":
        gt_src = split_dir / "train_ground_truth.tsv"
        gt_out = out_dir / "ground_truth.parquet"
        if not gt_out.exists() or force:
            gt = read_source_full(gt_src)
            pq.write_table(pa.Table.from_pandas(gt, preserve_index=False), str(gt_out))
        results["ground_truth"] = gt_out

    return results


def load_preprocessed(path: Path) -> pd.DataFrame:
    """Load a preprocessed Parquet artifact back into a DataFrame.

    Use only for artifacts known to fit comfortably in memory whole (e.g.
    Source 1, or ground_truth). For the large Source 2/3 artifacts (multiple
    GB as Parquet, 5M+ rows), prefer ``load_preprocessed_filtered`` below --
    a single ``pq.read_table(...).to_pandas()`` call on one of those needs a
    large contiguous allocation for the whole table (Arrow buffers and the
    resulting pandas block manager coexist in memory during the conversion),
    which is exactly what raises ``pyarrow.lib.ArrowMemoryError`` on an 8GB
    machine once anything else is already resident.
    """
    return pq.read_table(str(path)).to_pandas()


def load_preprocessed_filtered(
    path: Path,
    countries: set[str] | None = None,
    entity_ids: set[str] | None = None,
    batch_size: int = 200_000,
) -> pd.DataFrame:
    """Stream a large preprocessed Parquet artifact in row-group batches,
    keeping only rows matching ``countries`` and/or ``entity_ids``, without
    ever materializing the full table in memory at once.

    Peak memory is bounded by roughly one batch plus the accumulated filtered
    result, instead of the whole file. Pass both filters as ``None`` to
    return everything -- still batched, so even an unfiltered read avoids
    the single large Arrow-to-pandas conversion that a plain
    ``pq.read_table(path).to_pandas()`` call requires.
    """
    pf = pq.ParquetFile(str(path))
    kept_chunks: list[pd.DataFrame] = []
    schema_chunk: pd.DataFrame | None = None

    for batch in pf.iter_batches(batch_size=batch_size):
        chunk = batch.to_pandas()
        if schema_chunk is None:
            schema_chunk = chunk.iloc[0:0]
        if countries is not None:
            chunk = chunk[chunk["country"].isin(countries)]
        if entity_ids is not None:
            chunk = chunk[chunk["entity_id"].isin(entity_ids)]
        if len(chunk):
            kept_chunks.append(chunk)

    if not kept_chunks:
        return schema_chunk if schema_chunk is not None else pd.DataFrame()
    return pd.concat(kept_chunks, ignore_index=True)
