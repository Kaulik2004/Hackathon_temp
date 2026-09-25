"""Vectorized pairwise similarity feature extraction for candidate pairs.

All string-similarity features use ``rapidfuzz`` (C++-backed), never a
Python-loop ``python-Levenshtein`` call per pair -- at candidate-pair counts
in the hundreds of thousands to millions, a pure-Python loop would dominate
runtime. Pairs are processed in chunks (``chunk_size``) to bound peak memory
when joining the (potentially very large) candidate-pairs frame against the
preprocessed lookup tables.

Empty-string handling is explicit throughout: ``rapidfuzz.fuzz.ratio("", "")``
returns 0, not 100 -- an empty-vs-empty comparison is "nothing to compare",
not "identical", so every feature that could be computed on an empty field
is paired with an explicit "*_present_both" flag the model can use to tell
"fields disagree" apart from "no data on one/both sides".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from . import preprocessing

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "name_exact_match",
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaro_winkler",
    "name_token_jaccard",
    "name_char_ngram_jaccard",
    "name_len_ratio",
    "name_first_token_match",
    "name_used_alias",
    "addr_token_jaccard",
    "addr_token_sort_ratio",
    "addr_partial_ratio",
    "addr_house_number_match",
    "addr_house_number_present_both",
    "addr_postal_code_match",
    "addr_postal_code_present_both",
    "addr_len_ratio",
    "addr_missing_a",
    "addr_missing_b",
    "country_match",
    "tfidf_name_score",
    "tfidf_addr_score",
    "embedding_score",
    "embedding_available",
    "matched_exact_key",
    "blocking_channel_count",
    "candidate_source_is_s3",
    "rank_within_s1_by_name_score",
    "n_candidates_for_s1",
    "score_gap_to_next_best",
]


def _char_ngram_set(s: str, n: int = 3) -> set[str]:
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _best_name_scores(name_a: str, alias_a: str, name_b: str, alias_b: str) -> dict:
    """Compute name-similarity metrics across primary/alias name combinations
    on both sides, keeping whichever pairing scores highest.

    DBA/"formerly" rewrites can appear on either side of a pair -- Source 1
    is the cleaner reference source, but nothing prevents a noisy Source
    2/3 record from being the one carrying the "formerly X" alias (as
    observed in the real data, e.g. "Onyxveo formerly Espinoza & Davis
    Industrials LLC"). Checking only one side's alias would silently miss
    exactly the DBA case this feature exists to catch.
    """
    a_candidates = [(name_a, False)]
    if alias_a and alias_a != name_a:
        a_candidates.append((alias_a, True))
    b_candidates = [(name_b, False)]
    if alias_b and alias_b != name_b:
        b_candidates.append((alias_b, True))

    best = None
    for cand_a, used_alias_a in a_candidates:
        for cand_b, used_alias_b in b_candidates:
            scores = {
                "name_exact_match": float(cand_a == cand_b and bool(cand_b)),
                "name_levenshtein_ratio": fuzz.ratio(cand_a, cand_b) / 100.0,
                "name_token_sort_ratio": fuzz.token_sort_ratio(cand_a, cand_b) / 100.0,
                "name_token_set_ratio": fuzz.token_set_ratio(cand_a, cand_b) / 100.0,
                "name_partial_ratio": fuzz.partial_ratio(cand_a, cand_b) / 100.0,
                "name_jaro_winkler": JaroWinkler.normalized_similarity(cand_a, cand_b),
                "name_used_alias": float(used_alias_a or used_alias_b),
            }
            rank_key = scores["name_token_sort_ratio"] + scores["name_levenshtein_ratio"]
            if best is None or rank_key > best[0]:
                best = (rank_key, scores)
    return best[1]


def _compute_row_features(row: pd.Series) -> dict:
    name_a, name_b = row["name_norm_s1"], row["name_norm_t"]
    alias_a = row.get("name_alias_s1", "")
    alias_b = row.get("name_alias_t", "")

    name_scores = _best_name_scores(name_a, alias_a, name_b, alias_b)

    tokens_a = set(preprocessing.tokenize(name_a))
    tokens_b = set(preprocessing.tokenize(name_b))
    ngrams_a = _char_ngram_set(name_a)
    ngrams_b = _char_ngram_set(name_b)

    toks_a_list = preprocessing.tokenize(name_a)
    toks_b_list = preprocessing.tokenize(name_b)
    first_token_match = bool(toks_a_list and toks_b_list and toks_a_list[0] == toks_b_list[0])

    addr_a, addr_b = row["addr_norm_s1"], row["addr_norm_t"]
    addr_tokens_a = set(preprocessing.tokenize(addr_a))
    addr_tokens_b = set(preprocessing.tokenize(addr_b))

    house_a, house_b = row["addr_house_number_s1"], row["addr_house_number_t"]
    postal_a, postal_b = row["addr_postal_code_s1"], row["addr_postal_code_t"]

    out = {
        **name_scores,
        "name_token_jaccard": _jaccard(tokens_a, tokens_b),
        "name_char_ngram_jaccard": _jaccard(ngrams_a, ngrams_b),
        "name_len_ratio": _len_ratio(name_a, name_b),
        "name_first_token_match": float(first_token_match),
        "addr_token_jaccard": _jaccard(addr_tokens_a, addr_tokens_b),
        "addr_token_sort_ratio": fuzz.token_sort_ratio(addr_a, addr_b) / 100.0,
        "addr_partial_ratio": fuzz.partial_ratio(addr_a, addr_b) / 100.0 if addr_a and addr_b else 0.0,
        "addr_house_number_match": float(bool(house_a) and house_a == house_b),
        "addr_house_number_present_both": float(bool(house_a) and bool(house_b)),
        "addr_postal_code_match": float(bool(postal_a) and postal_a == postal_b),
        "addr_postal_code_present_both": float(bool(postal_a) and bool(postal_b)),
        "addr_len_ratio": _len_ratio(addr_a, addr_b),
        "addr_missing_a": float(not addr_a),
        "addr_missing_b": float(not addr_b),
        "country_match": float(row["country_s1"] == row["country_t"]),
    }
    return out


def compute_pairwise_features(
    pairs_df: pd.DataFrame,
    s1_lookup: pd.DataFrame,
    target_lookup: pd.DataFrame,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Join blocking pairs against preprocessed lookups and compute features.

    ``pairs_df`` columns: source1_entity_id, candidate_entity_id,
    candidate_source, plus blocking_* score columns (from blocking.py).
    ``s1_lookup``/``target_lookup``: preprocessed frames indexed by
    entity_id, containing name_norm/name_alias/addr_norm/
    addr_house_number/addr_postal_code/country.
    """
    if pairs_df.empty:
        return pairs_df.assign(**{col: pd.Series(dtype="float32") for col in FEATURE_COLUMNS})

    s1_cols = ["entity_id", "name_norm", "name_alias", "addr_norm", "addr_house_number", "addr_postal_code", "country"]
    t_cols = ["entity_id", "name_norm", "name_alias", "addr_norm", "addr_house_number", "addr_postal_code", "country"]

    s1_idx = s1_lookup[s1_cols].set_index("entity_id")
    t_idx = target_lookup[t_cols].set_index("entity_id")

    feature_chunks = []
    n = len(pairs_df)
    for start in range(0, n, chunk_size):
        chunk = pairs_df.iloc[start : start + chunk_size]
        joined = chunk.join(s1_idx.add_suffix("_s1"), on="source1_entity_id")
        joined = joined.join(t_idx.add_suffix("_t"), on="candidate_entity_id")
        joined = joined.fillna(
            {
                "name_norm_s1": "",
                "name_alias_s1": "",
                "addr_norm_s1": "",
                "addr_house_number_s1": "",
                "addr_postal_code_s1": "",
                "country_s1": "",
                "name_norm_t": "",
                "name_alias_t": "",
                "addr_norm_t": "",
                "addr_house_number_t": "",
                "addr_postal_code_t": "",
                "country_t": "",
            }
        )
        computed = joined.apply(_compute_row_features, axis=1, result_type="expand")
        feature_chunks.append(pd.concat([chunk.reset_index(drop=True), computed.reset_index(drop=True)], axis=1))
        logger.info("computed features for %d/%d pairs", min(start + chunk_size, n), n)

    features = pd.concat(feature_chunks, ignore_index=True)

    features["embedding_available"] = features["embedding_score"].notna().astype("float32")
    features["embedding_score"] = features["embedding_score"].fillna(0.0)
    features["tfidf_name_score"] = features["tfidf_name_score"].fillna(0.0)
    features["tfidf_addr_score"] = features["tfidf_addr_score"].fillna(0.0)
    features["matched_exact_key"] = features["matched_exact_key"].fillna(False).astype("float32")
    features["blocking_channel_count"] = features["blocking_channel_count"].fillna(0).astype("float32")
    features["candidate_source_is_s3"] = (features["candidate_source"] == "S3").astype("float32")

    return features


def add_per_s1_aggregate_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-S1-entity competitive-context features via vectorized groupby.

    - rank_within_s1_by_name_score: 1 = best candidate for this S1 entity
      by combined name score.
    - n_candidates_for_s1: candidate-set size for this S1 entity (proxy for
      name genericity / ambiguity).
    - score_gap_to_next_best: this candidate's combined score minus the
      runner-up's, within the same S1 group (large gap = confident pick,
      small gap = ambiguous).
    """
    if features_df.empty:
        features_df["rank_within_s1_by_name_score"] = pd.Series(dtype="float32")
        features_df["n_candidates_for_s1"] = pd.Series(dtype="float32")
        features_df["score_gap_to_next_best"] = pd.Series(dtype="float32")
        return features_df

    out = features_df.copy()
    combined_score = out["name_token_sort_ratio"] + out["name_levenshtein_ratio"] + out["tfidf_name_score"]
    out["_combined_score"] = combined_score

    grouped = out.groupby("source1_entity_id")["_combined_score"]
    out["rank_within_s1_by_name_score"] = grouped.rank(method="first", ascending=False).astype("float32")
    out["n_candidates_for_s1"] = grouped.transform("size").astype("float32")

    def _gap(s: pd.Series) -> pd.Series:
        sorted_vals = np.sort(s.to_numpy())[::-1]
        best = sorted_vals[0]
        second = sorted_vals[1] if len(sorted_vals) > 1 else sorted_vals[0]
        gap = best - second
        return pd.Series(np.where(s == best, gap, 0.0), index=s.index)

    out["score_gap_to_next_best"] = grouped.transform(lambda s: _gap(s)).astype("float32")
    out = out.drop(columns=["_combined_score"])
    return out


def build_feature_matrix(
    pairs_df: pd.DataFrame,
    s1_lookup: pd.DataFrame,
    target_lookup: pd.DataFrame,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Convenience wrapper: pairwise features + per-S1 aggregate features."""
    features = compute_pairwise_features(pairs_df, s1_lookup, target_lookup, chunk_size=chunk_size)
    features = add_per_s1_aggregate_features(features)
    return features
