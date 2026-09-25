"""Multi-channel, memory-safe candidate generation (blocking) for entity resolution.

Why blocking is mandatory, not an optimization
-----------------------------------------------
Source 1 has ~2.2M (train) / ~1.7M (test) rows; the pooled Source 2 + Source 3
target space has ~10.3M rows. A dense S1 x target similarity matrix would need
on the order of 2x10^13 cells -- impossible to allocate on any single machine.
This module never constructs such a matrix. Every comparison is restricted to
a bounded candidate set produced by an inverted-index / sparse-matrix-multiply
mechanism, exactly as required by the problem's scale.

Country partitioning and the open-set constraint
--------------------------------------------------
``partition_by_country`` groups every frame by whatever string values are
actually present in the data. It never compares against a literal "US",
"India", or "France" -- the partition keys come entirely from
``df["country"].unique()``. A country that has Source-1 rows but zero
Source-2/Source-3 rows (which should not happen in practice, but must not
silently crash or drop entities) still gets a partition entry with empty
target frames, so those Source-1 entities correctly fall through to "zero
candidates" rather than vanishing from the pipeline. This is the entire
mechanism that makes the France test country (absent from training data)
"just work": it is nothing more than a value that appears in
``country.unique()`` at runtime and is routed through the identical code path
as every other country.

Candidate channels (unioned per Source-1 entity)
--------------------------------------------------
1. Exact/near-exact blocking keys (cheap hash-join): suffix-stripped name,
   sorted-token name (catches word-order transpositions), house-number +
   postal-code key.
2. TF-IDF character n-gram (name) and word-token (address) sparse blocking:
   a ``TfidfVectorizer`` is fit on the target pool, Source-1 rows are
   *transformed* (not re-fit) through the same vectorizer so token ids align,
   and candidates come from a **sparse x sparse matrix multiply**
   (``Q @ T.T``), never densified. ``max_df`` performs document-frequency
   pruning so generic tokens (common short n-grams, filler words) don't
   explode postings lists, per the "standard practice in record linkage at
   scale" requirement. Because both matrices are L2-normalized (the sklearn
   default for TF-IDF), nonzero entries of the product are exactly the
   cosine similarities -- reused downstream as a feature, not thrown away.
3. Optional multilingual sentence-embedding ANN (FAISS), lazily imported and
   only used when explicitly requested or a GPU is available. This is the
   only channel able to retrieve true matches whose business name changes
   script entirely between sources (a real, observed pattern for India
   records, e.g. a Latin-script Source-1 name matching a Devanagari-script
   Source-2 name with zero shared characters or word tokens).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from . import preprocessing

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BlockingConfig:
    max_df_ratio_name: float = 0.02
    max_df_ratio_addr: float = 0.05
    ngram_range_name: tuple[int, int] = (3, 5)
    max_candidates_per_channel: int = 200
    max_candidates_per_s1: int = 400
    s1_chunk_size: int = 50_000
    use_embeddings: bool = False
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_topk: int = 20
    embedding_batch_size: int = 256


# ---------------------------------------------------------------------------
# Stage 0: country partitioning (open-set safe)
# ---------------------------------------------------------------------------

def partition_by_country(
    s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame
) -> dict[str, dict[str, pd.DataFrame]]:
    """Partition all three frames by the country values present in S1.

    Iterates over ``s1["country"].unique()`` (S1-driven) so every Source-1
    country gets an entry, even one absent from S2/S3 (empty target slices
    rather than a KeyError or a silently skipped country). No country value
    is ever hardcoded or whitelisted here.
    """
    partitions: dict[str, dict[str, pd.DataFrame]] = {}
    s2_by_country = {k: v for k, v in s2.groupby("country", sort=False)}
    s3_by_country = {k: v for k, v in s3.groupby("country", sort=False)}

    empty_s2 = s2.iloc[0:0]
    empty_s3 = s3.iloc[0:0]

    for country, s1_slice in s1.groupby("country", sort=False):
        partitions[country] = {
            "s1": s1_slice,
            "s2": s2_by_country.get(country, empty_s2),
            "s3": s3_by_country.get(country, empty_s3),
        }
        if country not in s2_by_country and country not in s3_by_country:
            logger.warning(
                "country %r has %d S1 rows but zero S2/S3 rows; those "
                "entities will get zero blocking candidates",
                country,
                len(s1_slice),
            )
    return partitions


# ---------------------------------------------------------------------------
# Stage 1: exact / near-exact blocking keys
# ---------------------------------------------------------------------------

def build_exact_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add cheap exact-match blocking key columns to a preprocessed frame."""
    out = df.copy()
    out["key_name_stripped"] = out["name_suffix_stripped"]

    def sorted_tokens(s: str) -> str:
        toks = sorted(preprocessing.tokenize(s))
        return " ".join(toks)

    out["key_name_sorted"] = out["name_norm"].map(sorted_tokens)
    out["key_name_first3_sorted"] = out["name_norm"].map(
        lambda s: " ".join(sorted(preprocessing.tokenize(s)[:3]))
    )
    has_house = out["addr_house_number"].astype(bool) & out["addr_postal_code"].astype(bool)
    out["key_addr_house_postal"] = np.where(
        has_house,
        out["addr_house_number"] + "|" + out["addr_postal_code"],
        "",
    )
    return out


def _exact_key_candidates(
    s1_keyed: pd.DataFrame, target_keyed: pd.DataFrame, key_col: str
) -> pd.DataFrame:
    """Hash-join on a single key column, dropping empty keys (which would
    otherwise match every empty-keyed row against every other)."""
    left = s1_keyed[s1_keyed[key_col] != ""][["entity_id", key_col]]
    right = target_keyed[target_keyed[key_col] != ""][["entity_id", key_col]]
    if left.empty or right.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    merged = left.merge(right, on=key_col, suffixes=("_s1", "_t"))
    return merged.rename(
        columns={"entity_id_s1": "source1_entity_id", "entity_id_t": "candidate_entity_id"}
    )[["source1_entity_id", "candidate_entity_id"]]


def generate_exact_key_pairs(s1_keyed: pd.DataFrame, target_keyed: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "key_name_stripped",
        "key_name_sorted",
        "key_name_first3_sorted",
        "key_addr_house_postal",
    ]
    frames = [_exact_key_candidates(s1_keyed, target_keyed, k) for k in key_cols]
    if not frames:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    combined = pd.concat(frames, ignore_index=True)
    combined["matched_exact_key"] = True
    return combined.drop_duplicates(["source1_entity_id", "candidate_entity_id"])


# ---------------------------------------------------------------------------
# Stage 2: TF-IDF sparse blocking (never densified)
# ---------------------------------------------------------------------------

def fit_tfidf_channel(
    texts: pd.Series, analyzer: str, ngram_range: tuple[int, int], max_df: float
) -> TfidfVectorizer:
    # Guard against a degenerate (empty / all-stopword) corpus.
    non_empty = texts[texts.str.len() > 0]
    if non_empty.empty:
        non_empty = pd.Series([" "])

    # A max_df *ratio* can round below min_df=1 document count on a small
    # partition (a small country partition, a small subsample distractor
    # pool, ...), which scikit-learn rejects outright. DF pruning is a
    # large-corpus concern; on a tiny corpus every token is informative, so
    # fall back to no pruning (max_df=1.0) rather than crash.
    effective_max_df = max_df
    if int(max_df * len(non_empty)) < 1:
        effective_max_df = 1.0

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_df=effective_max_df,
        min_df=1,
        sublinear_tf=True,
    )
    vectorizer.fit(non_empty)
    return vectorizer


def sparse_topk_candidates(
    Q: sp.csr_matrix,
    T: sp.csr_matrix,
    s1_ids: np.ndarray,
    target_ids: np.ndarray,
    k: int,
    chunk_size: int = 50_000,
    score_col: str = "score",
) -> pd.DataFrame:
    """Top-k candidates per Source-1 row via chunked sparse matmul.

    Never densifies the full Q @ T.T product. Processes Q in row-chunks so
    peak memory is bounded by (chunk_size x n_target_nnz) rather than the
    full partition, and only the already-small top-k slice of each chunk's
    result is ever touched with per-row indexing.
    """
    n_s1 = Q.shape[0]
    if n_s1 == 0 or T.shape[0] == 0:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", score_col])

    T_csc = T.T.tocsr()  # T.T as CSR for the matmul's right-hand side access pattern
    rows_s1: list[str] = []
    rows_cand: list[str] = []
    rows_score: list[float] = []

    for start in range(0, n_s1, chunk_size):
        end = min(start + chunk_size, n_s1)
        Q_chunk = Q[start:end]
        sim_chunk = Q_chunk @ T_csc  # sparse x sparse -> sparse CSR, never dense
        sim_chunk = sim_chunk.tocsr()

        for local_i in range(sim_chunk.shape[0]):
            row = sim_chunk.getrow(local_i)
            if row.nnz == 0:
                continue
            if row.nnz > k:
                top_local = np.argpartition(row.data, -k)[-k:]
            else:
                top_local = np.arange(row.nnz)
            cols = row.indices[top_local]
            vals = row.data[top_local]
            s1_id = s1_ids[start + local_i]
            rows_s1.extend([s1_id] * len(cols))
            rows_cand.extend(target_ids[cols].tolist())
            rows_score.extend(vals.tolist())

    return pd.DataFrame(
        {"source1_entity_id": rows_s1, "candidate_entity_id": rows_cand, score_col: rows_score}
    )


def _tfidf_channel_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    text_col: str,
    analyzer: str,
    ngram_range: tuple[int, int],
    max_df: float,
    k: int,
    chunk_size: int,
    score_col: str,
) -> pd.DataFrame:
    if s1_df.empty or target_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", score_col])

    vectorizer = fit_tfidf_channel(target_df[text_col], analyzer, ngram_range, max_df)
    T = vectorizer.transform(target_df[text_col])
    Q = vectorizer.transform(s1_df[text_col])

    return sparse_topk_candidates(
        Q,
        T,
        s1_df["entity_id"].to_numpy(),
        target_df["entity_id"].to_numpy(),
        k=k,
        chunk_size=chunk_size,
        score_col=score_col,
    )


# ---------------------------------------------------------------------------
# Stage 3: optional multilingual embedding ANN channel
# ---------------------------------------------------------------------------

def _embeddings_available() -> bool:
    try:
        import torch  # noqa: F401
        import sentence_transformers  # noqa: F401
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


def gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def build_embedding_index(target_df: pd.DataFrame, model, batch_size: int = 256):
    """Encode target names and build a FAISS inner-product index.

    Lazily imports faiss. Vectors are L2-normalized so inner product ==
    cosine similarity. Uses a flat index for partitions that comfortably fit
    in RAM and an IVF index for larger ones (relevant on the full-scale EC2
    run, not the local subsample).
    """
    import faiss

    texts = target_df["name_norm"].tolist()
    embeddings = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    ).astype("float32")

    dim = embeddings.shape[1]
    n = embeddings.shape[0]
    if n < 500_000:
        index = faiss.IndexFlatIP(dim)
    else:
        nlist = max(1, int(n**0.5))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
    index.add(embeddings)
    return index


def query_embedding_candidates(
    s1_df: pd.DataFrame, model, index, target_ids: np.ndarray, k: int = 20, batch_size: int = 256
) -> pd.DataFrame:
    if s1_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "embedding_score"])

    texts = s1_df["name_norm"].tolist()
    queries = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    ).astype("float32")

    scores, idx = index.search(queries, k)
    s1_ids = s1_df["entity_id"].to_numpy()

    rows_s1, rows_cand, rows_score = [], [], []
    for i in range(len(s1_ids)):
        for j in range(idx.shape[1]):
            col = idx[i, j]
            if col < 0:
                continue
            rows_s1.append(s1_ids[i])
            rows_cand.append(target_ids[col])
            rows_score.append(float(scores[i, j]))

    return pd.DataFrame(
        {"source1_entity_id": rows_s1, "candidate_entity_id": rows_cand, "embedding_score": rows_score}
    )


def _load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


# ---------------------------------------------------------------------------
# Orchestration: union of all channels, per country partition
# ---------------------------------------------------------------------------

def _union_candidates(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "tfidf_name_score",
                "tfidf_addr_score",
                "embedding_score",
                "matched_exact_key",
            ]
        )
    combined = pd.concat(frames, ignore_index=True, sort=False)
    agg = {
        "tfidf_name_score": "max",
        "tfidf_addr_score": "max",
        "embedding_score": "max",
        "matched_exact_key": "max",
    }
    present_agg = {k: v for k, v in agg.items() if k in combined.columns}
    grouped = combined.groupby(
        ["source1_entity_id", "candidate_entity_id"], as_index=False
    ).agg(present_agg)
    for col in ("tfidf_name_score", "tfidf_addr_score", "embedding_score"):
        if col not in grouped.columns:
            grouped[col] = np.nan
    if "matched_exact_key" not in grouped.columns:
        grouped["matched_exact_key"] = False
    else:
        grouped["matched_exact_key"] = np.where(
            grouped["matched_exact_key"].isna(), False, grouped["matched_exact_key"]
        ).astype(bool)

    channel_presence = pd.DataFrame(index=grouped.index)
    for col in ("tfidf_name_score", "tfidf_addr_score", "embedding_score"):
        channel_presence[col] = grouped[col].notna()
    channel_presence["matched_exact_key"] = grouped["matched_exact_key"].astype(bool)
    grouped["blocking_channel_count"] = channel_presence.sum(axis=1)
    return grouped


def _cap_candidates_per_s1(pairs: pd.DataFrame, max_per_s1: int) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    pairs = pairs.copy()
    combined_score = pairs[["tfidf_name_score", "tfidf_addr_score", "embedding_score"]].fillna(0).max(axis=1)
    combined_score = combined_score + pairs["matched_exact_key"].astype(float) * 1e-3
    pairs["_rank_score"] = combined_score
    pairs["_rank"] = pairs.groupby("source1_entity_id")["_rank_score"].rank(
        method="first", ascending=False
    )
    kept = pairs[pairs["_rank"] <= max_per_s1].drop(columns=["_rank_score", "_rank"])
    return kept


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    cfg: Optional[BlockingConfig] = None,
) -> pd.DataFrame:
    """Generate the unioned candidate set for every Source-1 entity.

    ``s1_df``/``s2_df``/``s3_df`` must already be preprocessed (see
    preprocessing.normalize_dataframe). Returns a long-format frame:
    source1_entity_id, candidate_entity_id, candidate_source ('S2'/'S3'),
    tfidf_name_score, tfidf_addr_score, embedding_score, matched_exact_key,
    blocking_channel_count. An S1 entity with zero candidates simply has no
    rows here -- callers must left-join against the full S1 id list before
    writing any output file.
    """
    cfg = cfg or BlockingConfig()

    embedding_model = None
    use_embeddings = cfg.use_embeddings or gpu_available()
    if use_embeddings:
        if _embeddings_available():
            logger.info("loading embedding model %s", cfg.embedding_model_name)
            embedding_model = _load_embedding_model(cfg.embedding_model_name)
        else:
            logger.warning(
                "embedding channel requested but torch/sentence-transformers/faiss "
                "are not installed; continuing without it"
            )
            use_embeddings = False

    partitions = partition_by_country(s1_df, s2_df, s3_df)
    all_results = []

    for country, parts in partitions.items():
        s1_p, s2_p, s3_p = parts["s1"], parts["s2"], parts["s3"]
        logger.info(
            "blocking partition country=%r: s1=%d s2=%d s3=%d",
            country,
            len(s1_p),
            len(s2_p),
            len(s3_p),
        )
        if s1_p.empty:
            continue

        s1_keyed = build_exact_keys(s1_p)

        for source_label, target_p in (("S2", s2_p), ("S3", s3_p)):
            if target_p.empty:
                continue
            target_keyed = build_exact_keys(target_p)

            exact_pairs = generate_exact_key_pairs(s1_keyed, target_keyed)

            name_pairs = _tfidf_channel_candidates(
                s1_p,
                target_p,
                text_col="name_norm",
                analyzer="char_wb",
                ngram_range=cfg.ngram_range_name,
                max_df=cfg.max_df_ratio_name,
                k=cfg.max_candidates_per_channel,
                chunk_size=cfg.s1_chunk_size,
                score_col="tfidf_name_score",
            )
            addr_pairs = _tfidf_channel_candidates(
                s1_p,
                target_p,
                text_col="addr_norm",
                analyzer="word",
                ngram_range=(1, 1),
                max_df=cfg.max_df_ratio_addr,
                k=cfg.max_candidates_per_channel,
                chunk_size=cfg.s1_chunk_size,
                score_col="tfidf_addr_score",
            )

            channel_frames = [exact_pairs, name_pairs, addr_pairs]

            if use_embeddings and embedding_model is not None:
                index = build_embedding_index(
                    target_p, embedding_model, batch_size=cfg.embedding_batch_size
                )
                emb_pairs = query_embedding_candidates(
                    s1_p,
                    embedding_model,
                    index,
                    target_p["entity_id"].to_numpy(),
                    k=cfg.embedding_topk,
                    batch_size=cfg.embedding_batch_size,
                )
                channel_frames.append(emb_pairs)

            unioned = _union_candidates(channel_frames)
            if unioned.empty:
                continue
            unioned["candidate_source"] = source_label
            all_results.append(unioned)

    if not all_results:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
                "tfidf_name_score",
                "tfidf_addr_score",
                "embedding_score",
                "matched_exact_key",
                "blocking_channel_count",
            ]
        )

    result = pd.concat(all_results, ignore_index=True)
    result = _cap_candidates_per_s1(result, cfg.max_candidates_per_s1)
    return result
