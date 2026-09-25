"""Matching model: LightGBM/XGBoost classifier + F0.5-targeted threshold tuning.

Key design decisions
---------------------
* Labels come **only** from blocking's own candidate pairs joined against
  ground truth -- never by injecting ground truth into candidate generation.
  Doing the latter would make blocking recall unmeasurable and would let the
  model train on an unrealistically clean candidate distribution.
* The train/holdout split happens at the **Source-1 entity level** (stratified
  by country), never at the pair level. Splitting pairs directly would leak
  "this entity has N other confirmed matches" signal across the split and
  inflate the validation score.
* The competition metric -- per-S1-entity macro F_0.5, with singletons
  scoring 1.0 when left empty and 0.0 when anything is predicted for them --
  is implemented once (``per_entity_f_beta``) and used for *both* threshold
  tuning and final reporting. This guarantees "what was optimized" and "what
  is reported" are provably the same function, instead of tuning against a
  pair-level proxy (e.g. sklearn's flattened ``fbeta_score``) and reporting
  something else.
* Singletons get no special-cased branch: an S1 entity with every candidate
  scoring below the tuned threshold naturally produces an empty predicted
  set, which is exactly correct for both true singletons and (unavoidably)
  entities whose true match blocking failed to retrieve.
"""

from __future__ import annotations

import dataclasses
import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_id_list(s: str) -> set[str]:
    if not s:
        return set()
    return {x for x in s.split(",") if x}


def build_training_labels(candidate_pairs: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Add a binary ``label`` column: 1 if the candidate is a true match."""
    gt_map: dict[str, set[str]] = {
        row.source1_entity_id: parse_id_list(row.matched_entity_ids)
        for row in ground_truth.itertuples(index=False)
    }
    out = candidate_pairs.copy()
    out["label"] = [
        1 if cid in gt_map.get(s1, ()) else 0
        for s1, cid in zip(out["source1_entity_id"], out["candidate_entity_id"])
    ]
    return out


def compute_blocking_recall(candidate_pairs: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Fraction of ground-truth positive pairs present in candidate_pairs.

    This is the hard recall ceiling for everything downstream: no amount of
    threshold tuning can recover a true match that blocking never proposed
    as a candidate. Must be logged and inspected before trusting any
    trained-model result.
    """
    candidate_set = set(zip(candidate_pairs["source1_entity_id"], candidate_pairs["candidate_entity_id"]))
    total_gt_pairs = 0
    found = 0
    for row in ground_truth.itertuples(index=False):
        for cid in parse_id_list(row.matched_entity_ids):
            total_gt_pairs += 1
            if (row.source1_entity_id, cid) in candidate_set:
                found += 1
    if total_gt_pairs == 0:
        return 1.0
    return found / total_gt_pairs


def make_holdout_split(
    s1_df: pd.DataFrame, holdout_frac: float = 0.15, seed: int = 42
) -> tuple[set[str], set[str]]:
    """Split Source-1 entity ids into train/holdout, stratified by country.

    Splitting at the entity level (not the pair level) is required to avoid
    leaking one entity's confirmed matches across the split.
    """
    rng = np.random.default_rng(seed)
    train_ids: set[str] = set()
    holdout_ids: set[str] = set()
    for _, group in s1_df.groupby("country", sort=False):
        ids = group["entity_id"].to_numpy()
        rng.shuffle(ids)
        n_holdout = max(1, int(len(ids) * holdout_frac)) if len(ids) > 1 else 0
        holdout_ids.update(ids[:n_holdout])
        train_ids.update(ids[n_holdout:])
    return train_ids, holdout_ids


def train_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    model_type: Literal["lgbm", "xgb"] = "lgbm",
    scale_pos_weight: float | None = None,
    params: dict | None = None,
):
    """Train a gradient-boosted classifier on pairwise features.

    A standard binary log-loss objective is used (F_0.5 is not directly
    usable as a boosting objective in LightGBM/XGBoost without a custom
    objective, which is unnecessary complexity here); the precision-heavy
    requirement of the competition metric is instead handled entirely at
    the threshold-selection stage below, which is the correct place to
    control a post-hoc-binarized metric like F_0.5.
    """
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    if scale_pos_weight is None:
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    logger.info(
        "training %s: n_train=%d (pos=%d neg=%d, scale_pos_weight=%.2f) n_val=%d",
        model_type,
        len(y_train),
        n_pos,
        n_neg,
        scale_pos_weight,
        len(y_val),
    )

    if model_type == "lgbm":
        import lightgbm as lgb

        default_params = dict(
            objective="binary",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        )
        default_params.update(params or {})
        model = lgb.LGBMClassifier(**default_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
    elif model_type == "xgb":
        import xgboost as xgb

        default_params = dict(
            objective="binary:logistic",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="aucpr",
            early_stopping_rounds=30,
        )
        default_params.update(params or {})
        model = xgb.XGBClassifier(**default_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        raise ValueError(f"unknown model_type: {model_type!r}")

    return model


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def per_entity_f_beta(
    pred_map: dict[str, set[str]],
    gt_map: dict[str, set[str]],
    all_s1_ids: set[str],
    beta: float = 0.5,
) -> float:
    """The exact competition metric: macro-averaged per-S1-entity F_beta.

    A true singleton (empty ground-truth set) scores 1.0 if predicted empty,
    0.0 if anything is predicted. Every id in ``all_s1_ids`` contributes to
    the average, including ids with zero predicted candidates (predicted
    empty set) -- an entity blocking failed to retrieve a match for still
    counts, and correctly scores 0 if it had true matches.
    """
    beta_sq = beta * beta
    scores = []
    for s1 in all_s1_ids:
        pred = pred_map.get(s1, set())
        gt = gt_map.get(s1, set())
        if not gt and not pred:
            scores.append(1.0)
            continue
        if not pred:
            scores.append(0.0)
            continue
        if not gt:
            scores.append(0.0)
            continue
        tp = len(pred & gt)
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / len(pred)
        recall = tp / len(gt)
        denom = beta_sq * precision + recall
        f = (1 + beta_sq) * precision * recall / denom if denom > 0 else 0.0
        scores.append(f)
    return float(np.mean(scores)) if scores else 1.0


def tune_threshold_for_entity_f_beta(
    pairs_df_with_prob: pd.DataFrame,
    ground_truth: pd.DataFrame,
    all_s1_ids: set[str],
    beta: float = 0.5,
    grid: np.ndarray | None = None,
    prob_col: str = "prob",
) -> tuple[float, float]:
    """Grid-search the probability threshold maximizing the real per-entity
    macro F_beta on the given (holdout) set. Returns (best_threshold, best_score).

    This directly optimizes the competition metric, not a pair-level proxy.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    gt_map = {
        row.source1_entity_id: parse_id_list(row.matched_entity_ids)
        for row in ground_truth.itertuples(index=False)
    }

    best_threshold, best_score = grid[0], -1.0
    for threshold in grid:
        kept = pairs_df_with_prob[pairs_df_with_prob[prob_col] >= threshold]
        pred_map: dict[str, set[str]] = {}
        for s1, cid in zip(kept["source1_entity_id"], kept["candidate_entity_id"]):
            pred_map.setdefault(s1, set()).add(cid)
        score = per_entity_f_beta(pred_map, gt_map, all_s1_ids, beta=beta)
        logger.info("threshold=%.3f macro_F%.1f=%.4f", threshold, beta, score)
        if score > best_score:
            best_threshold, best_score = threshold, score

    return float(best_threshold), float(best_score)


def apply_threshold(pairs_df_with_prob: pd.DataFrame, threshold: float, prob_col: str = "prob") -> pd.DataFrame:
    return pairs_df_with_prob[pairs_df_with_prob[prob_col] >= threshold].copy()


@dataclasses.dataclass
class ModelArtifact:
    model: object
    threshold: float
    model_type: str
    feature_columns: list[str]


def save_model(artifact: ModelArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info("saved model artifact to %s (threshold=%.3f)", path, artifact.threshold)


def load_model(path: Path) -> ModelArtifact:
    with open(path, "rb") as f:
        return pickle.load(f)
