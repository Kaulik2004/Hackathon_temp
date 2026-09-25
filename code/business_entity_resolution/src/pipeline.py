"""End-to-end CLI orchestrator: sync -> preprocess -> block -> featurize ->
train/tune -> test-inference -> write outputs -> (optional) upload.

Usage: run as a module from ``code/business_entity_resolution/`` (so the
package's relative imports resolve), with the process's *working directory*
set to ``student_resource/`` (so the default ``dataset/``/``output/`` paths
line up with ``utils/validate_submission.py``'s own defaults). Concretely,
from ``student_resource/``:

    python -m src.pipeline --mode local --dataset-dir dataset --output-dir output ^
        --artifacts-dir code/business_entity_resolution/artifacts
    (cwd must contain code/business_entity_resolution/src/ on PYTHONPATH --
     the wrapper script code/business_entity_resolution/run_pipeline.py
     below sets this up automatically)

Or simply:

    python code/business_entity_resolution/run_pipeline.py --mode local --sample-size 20000

See README.md for the full command reference.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import blocking, feature_engineering, model, preprocessing

logger = logging.getLogger("business_entity_resolution.pipeline")


# ---------------------------------------------------------------------------
# Sampling (local subsample validation path)
# ---------------------------------------------------------------------------

def stratified_sample_s1(s1_df: pd.DataFrame, sample_size_per_country: int, seed: int) -> pd.DataFrame:
    """Stratified-by-country sample of Source-1 rows."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in s1_df.groupby("country", sort=False):
        n = min(sample_size_per_country, len(group))
        idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
        parts.append(group.loc[idx])
    return pd.concat(parts, ignore_index=True)


def restrict_target_pool_for_sample(
    sampled_s1: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    ground_truth: pd.DataFrame | None,
    max_pool_size: int = 400_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a target pool for a sampled S1 set: guaranteed ground-truth
    matches (if available) + a bounded random distractor pool per country,
    so the subsample is neither trivially easy (pool = only true matches)
    nor requires scanning the full multi-million-row target pool.
    """
    must_include: set[str] = set()
    if ground_truth is not None:
        # Vectorized isin() filter, not a Python-level loop over every
        # ground-truth row (2.2M+ rows on the full training set) -- the
        # loop form took ~15 minutes to isolate a few hundred sampled S1
        # entities' matches and was pure overhead.
        sampled_ids = set(sampled_s1["entity_id"])
        relevant_gt = ground_truth[ground_truth["source1_entity_id"].isin(sampled_ids)]
        for ids_str in relevant_gt["matched_entity_ids"]:
            must_include.update(model.parse_id_list(ids_str))

    rng = np.random.default_rng(seed)
    countries = set(sampled_s1["country"].unique())

    def _restrict(target_df: pd.DataFrame) -> pd.DataFrame:
        in_country = target_df[target_df["country"].isin(countries)]
        must = in_country[in_country["entity_id"].isin(must_include)]
        remaining_budget = max(0, max_pool_size - len(must))
        rest = in_country[~in_country["entity_id"].isin(must_include)]
        n_extra = min(remaining_budget, len(rest))
        if n_extra > 0:
            extra_idx = rng.choice(rest.index.to_numpy(), size=n_extra, replace=False)
            extra = rest.loc[extra_idx]
        else:
            extra = rest.iloc[0:0]
        return pd.concat([must, extra], ignore_index=True)

    return _restrict(s2_df), _restrict(s3_df)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _pairs_to_id_map(pairs_df: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if pairs_df.empty:
        return out
    for s1, group in pairs_df.groupby("source1_entity_id")["candidate_entity_id"]:
        ids = list(dict.fromkeys(group.tolist()))  # de-dup, preserve order
        out[s1] = ids
    return out


def write_id_list_tsv(pairs_df: pd.DataFrame, all_s1_ids: pd.Series, out_path: Path, id_col_name: str) -> None:
    id_map = _pairs_to_id_map(pairs_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{id_col_name}\n")
        for s1 in all_s1_ids:
            ids = id_map.get(s1, [])
            f.write(f"{s1}\t{','.join(ids)}\n")
    logger.info("wrote %s (%d rows)", out_path, len(all_s1_ids))


def write_candidate_pairs_tsv(pairs_df: pd.DataFrame, all_s1_ids: pd.Series, out_path: Path) -> None:
    write_id_list_tsv(pairs_df, all_s1_ids, out_path, "candidate_entity_ids")


def write_matching_results_tsv(matched_df: pd.DataFrame, all_s1_ids: pd.Series, out_path: Path) -> None:
    write_id_list_tsv(matched_df, all_s1_ids, out_path, "matched_entity_ids")


def run_internal_validation_checks(matching_path: Path, candidate_path: Path, test_s1_ids: set[str]) -> None:
    """Fast-fail sanity net, run before/independent of the external
    utils/validate_submission.py gate. Raises on any violation."""

    def _load(path: Path) -> dict[str, set[str]]:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
        out = {}
        for row in df.itertuples(index=False):
            ids = set(x for x in row[1].split(",") if x)
            out[row[0]] = ids
        return out, df

    matched_map, matched_df = _load(matching_path)
    candidate_map, candidate_df = _load(candidate_path)

    errors = []
    if len(matched_df) != matched_df.iloc[:, 0].nunique():
        errors.append("duplicate source1_entity_id rows in matching_results.tsv")
    if set(matched_df.iloc[:, 0]) != test_s1_ids:
        missing = test_s1_ids - set(matched_df.iloc[:, 0])
        extra = set(matched_df.iloc[:, 0]) - test_s1_ids
        if missing:
            errors.append(f"matching_results.tsv missing {len(missing)} required S1 id(s)")
        if extra:
            errors.append(f"matching_results.tsv has {len(extra)} unexpected S1 id(s)")

    for s1, ids in matched_map.items():
        if any(cid.startswith("S1-") for cid in ids):
            errors.append(f"self-match (S1- id) found in matching_results.tsv for {s1}")
        if any(not cid.startswith(("S2-", "S3-")) for cid in ids):
            errors.append(f"non-S2/S3 id found in matching_results.tsv for {s1}")
        if not ids <= candidate_map.get(s1, set()):
            errors.append(f"matched ids for {s1} are not a subset of candidate_pairs.tsv")

    if errors:
        raise RuntimeError("internal validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    logger.info("internal validation checks passed (%d S1 rows)", len(matched_df))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PipelineResult:
    blocking_recall: float | None
    holdout_f_beta: float | None
    threshold: float | None
    n_test_rows: int
    output_dir: Path


def _log_stage_time(name: str, start: float) -> None:
    logger.info("[timing] %s took %.1fs", name, time.perf_counter() - start)


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    project_root = Path.cwd()
    dataset_dir = project_root / args.dataset_dir
    output_dir = project_root / args.output_dir
    artifacts_dir = project_root / args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "aws":
        from . import aws_s3_connector

        if not args.s3_bucket:
            raise ValueError("--s3-bucket is required when --mode aws")
        syncer = aws_s3_connector.S3DatasetSync(args.s3_bucket, region_name=args.region)
        syncer.sync_dataset(dataset_dir)

    # ---- Preprocess ----
    t0 = time.perf_counter()
    train_paths = preprocessing.preprocess_all(dataset_dir, artifacts_dir, split="train", force=args.force_preprocess)
    test_paths = preprocessing.preprocess_all(dataset_dir, artifacts_dir, split="test", force=args.force_preprocess)
    _log_stage_time("preprocessing", t0)

    blocking_recall = None
    holdout_f_beta = None
    threshold = None
    artifact = None

    if not args.skip_train:
        # Train-side frames are only loaded when actually training -- a
        # --skip-train resume run (e.g. after a trained model.pkl already
        # exists) must never pay the cost of loading train_source2/3
        # (multi-GB as Parquet) just to throw them away unused. This was a
        # real contributor to an ArrowMemoryError seen in practice: those
        # frames stayed resident through the test-inference stage even when
        # they were never touched again.
        train_s1 = preprocessing.load_preprocessed(train_paths["source1"])
        ground_truth = preprocessing.load_preprocessed(train_paths["ground_truth"])

        # ---- Optional subsample for a fast local validation run ----
        if args.sample_size:
            logger.info("subsampling train: %d S1 entities per country", args.sample_size)
            train_s1_use = stratified_sample_s1(train_s1, args.sample_size, args.sample_seed)
            sample_countries = set(train_s1_use["country"].unique())
            # Batched (never-fully-materialized) load, pre-filtered to the
            # countries actually needed by this sample.
            train_s2_full = preprocessing.load_preprocessed_filtered(
                train_paths["source2"], countries=sample_countries
            )
            train_s3_full = preprocessing.load_preprocessed_filtered(
                train_paths["source3"], countries=sample_countries
            )
            train_s2_use, train_s3_use = restrict_target_pool_for_sample(
                train_s1_use, train_s2_full, train_s3_full, ground_truth, seed=args.sample_seed
            )
            del train_s2_full, train_s3_full
            logger.info(
                "sampled universe: s1=%d s2=%d s3=%d", len(train_s1_use), len(train_s2_use), len(train_s3_use)
            )
        else:
            train_s1_use = train_s1
            train_s2_use = preprocessing.load_preprocessed_filtered(train_paths["source2"])
            train_s3_use = preprocessing.load_preprocessed_filtered(train_paths["source3"])

        # ---- Blocking ----
        cfg = blocking.BlockingConfig(use_embeddings=args.use_embeddings)
        t0 = time.perf_counter()
        train_pairs = blocking.generate_candidate_pairs(train_s1_use, train_s2_use, train_s3_use, cfg)
        _log_stage_time("blocking (train)", t0)

        gt_for_sample = ground_truth[ground_truth["source1_entity_id"].isin(set(train_s1_use["entity_id"]))]
        blocking_recall = model.compute_blocking_recall(train_pairs, gt_for_sample)
        logger.info("blocking recall (train): %.4f", blocking_recall)

        # ---- Features + labels ----
        t0 = time.perf_counter()
        target_lookup = pd.concat([train_s2_use, train_s3_use], ignore_index=True)
        train_features = feature_engineering.build_feature_matrix(train_pairs, train_s1_use, target_lookup)
        train_features = model.build_training_labels(train_features, gt_for_sample)
        _log_stage_time("feature engineering (train)", t0)

        # ---- Split, train, tune threshold ----
        train_ids, holdout_ids = model.make_holdout_split(train_s1_use, args.holdout_frac, args.sample_seed)
        train_mask = train_features["source1_entity_id"].isin(train_ids)
        holdout_mask = train_features["source1_entity_id"].isin(holdout_ids)

        feature_cols = [c for c in feature_engineering.FEATURE_COLUMNS if c in train_features.columns]
        X_train = train_features.loc[train_mask, feature_cols]
        y_train = train_features.loc[train_mask, "label"].to_numpy()
        X_val = train_features.loc[holdout_mask, feature_cols]
        y_val = train_features.loc[holdout_mask, "label"].to_numpy()

        t0 = time.perf_counter()
        clf = model.train_classifier(X_train, y_train, X_val, y_val, model_type=args.model)
        _log_stage_time("model training", t0)

        holdout_pairs = train_features.loc[holdout_mask].copy()
        holdout_pairs["prob"] = model.predict_proba(clf, X_val)
        holdout_s1_ids = set(train_s1_use.loc[train_s1_use["entity_id"].isin(holdout_ids), "entity_id"])
        gt_holdout = ground_truth[ground_truth["source1_entity_id"].isin(holdout_s1_ids)]

        threshold, holdout_f_beta = model.tune_threshold_for_entity_f_beta(
            holdout_pairs, gt_holdout, holdout_s1_ids, beta=0.5
        )
        logger.info("tuned threshold=%.3f -> holdout macro F0.5=%.4f", threshold, holdout_f_beta)

        precision, recall = _precision_recall_at_threshold(holdout_pairs, gt_holdout, threshold)
        singleton_acc = _singleton_accuracy(holdout_pairs, gt_holdout, holdout_s1_ids, threshold)
        logger.info(
            "holdout diagnostics: precision=%.4f recall=%.4f singleton_accuracy=%.4f",
            precision,
            recall,
            singleton_acc,
        )

        artifact = model.ModelArtifact(
            model=clf, threshold=threshold, model_type=args.model, feature_columns=feature_cols
        )
        model.save_model(artifact, artifacts_dir / "model.pkl")

        # Free every train-side object before moving to test inference --
        # they are never touched again, and holding them resident is exactly
        # what starved the test-side Parquet load in practice.
        del (
            train_s1, train_s1_use, train_s2_use, train_s3_use, ground_truth,
            train_pairs, gt_for_sample, target_lookup, train_features,
            X_train, y_train, X_val, y_val, holdout_pairs, gt_holdout, clf,
        )
        gc.collect()
    else:
        artifact = model.load_model(artifacts_dir / "model.pkl")

    # ---- Test inference ----
    # test_source2/3 are large (multi-GB as Parquet); load_preprocessed_filtered
    # streams the Parquet->pandas conversion in batches (never materializing
    # the whole table at once) and, when sampling, filters to only the
    # countries the sampled S1 entities actually need.
    test_s1 = preprocessing.load_preprocessed(test_paths["source1"])

    if args.sample_size:
        logger.info("subsampling test: %d S1 entities per country (plumbing check only)", args.sample_size)
        test_s1_use = stratified_sample_s1(test_s1, args.sample_size, args.sample_seed)
        sample_countries = set(test_s1_use["country"].unique())
        test_s2_full = preprocessing.load_preprocessed_filtered(
            test_paths["source2"], countries=sample_countries
        )
        test_s3_full = preprocessing.load_preprocessed_filtered(
            test_paths["source3"], countries=sample_countries
        )
        test_s2_use, test_s3_use = restrict_target_pool_for_sample(
            test_s1_use, test_s2_full, test_s3_full, None, seed=args.sample_seed
        )
        del test_s2_full, test_s3_full
        gc.collect()
    else:
        test_s1_use = test_s1
        test_s2_use = preprocessing.load_preprocessed_filtered(test_paths["source2"])
        test_s3_use = preprocessing.load_preprocessed_filtered(test_paths["source3"])

    cfg = blocking.BlockingConfig(use_embeddings=args.use_embeddings)
    t0 = time.perf_counter()
    test_pairs = blocking.generate_candidate_pairs(test_s1_use, test_s2_use, test_s3_use, cfg)
    _log_stage_time("blocking (test)", t0)

    target_lookup_test = pd.concat([test_s2_use, test_s3_use], ignore_index=True)
    t0 = time.perf_counter()
    test_features = feature_engineering.build_feature_matrix(test_pairs, test_s1_use, target_lookup_test)
    _log_stage_time("feature engineering (test)", t0)

    feature_cols = artifact.feature_columns
    for col in feature_cols:
        if col not in test_features.columns:
            test_features[col] = 0.0
    X_test = test_features[feature_cols]
    test_features["prob"] = model.predict_proba(artifact.model, X_test) if len(X_test) else np.array([])

    matched = model.apply_threshold(test_features, artifact.threshold)

    # ---- Write outputs (left-join against full S1 id list) ----
    all_test_s1_ids = test_s1_use["entity_id"]
    write_candidate_pairs_tsv(test_pairs, all_test_s1_ids, output_dir / "candidate_pairs.tsv")
    write_matching_results_tsv(matched, all_test_s1_ids, output_dir / "matching_results.tsv")

    run_internal_validation_checks(
        output_dir / "matching_results.tsv",
        output_dir / "candidate_pairs.tsv",
        set(all_test_s1_ids),
    )

    if args.mode == "aws":
        from . import aws_s3_connector

        syncer = aws_s3_connector.S3DatasetSync(args.s3_bucket, region_name=args.region)
        syncer.upload_outputs(output_dir)

    result = PipelineResult(
        blocking_recall=blocking_recall,
        holdout_f_beta=holdout_f_beta,
        threshold=threshold,
        n_test_rows=len(all_test_s1_ids),
        output_dir=output_dir,
    )
    _print_summary(result, args)
    return result


def _precision_recall_at_threshold(pairs_with_prob: pd.DataFrame, ground_truth: pd.DataFrame, threshold: float):
    gt_map = {
        row.source1_entity_id: model.parse_id_list(row.matched_entity_ids)
        for row in ground_truth.itertuples(index=False)
    }
    kept = pairs_with_prob[pairs_with_prob["prob"] >= threshold]
    tp = fp = fn = 0
    pred_map: dict[str, set[str]] = {}
    for s1, cid in zip(kept["source1_entity_id"], kept["candidate_entity_id"]):
        pred_map.setdefault(s1, set()).add(cid)
    for s1, gt_ids in gt_map.items():
        pred_ids = pred_map.get(s1, set())
        tp += len(pred_ids & gt_ids)
        fp += len(pred_ids - gt_ids)
        fn += len(gt_ids - pred_ids)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _singleton_accuracy(pairs_with_prob, ground_truth, s1_ids, threshold) -> float:
    gt_map = {
        row.source1_entity_id: model.parse_id_list(row.matched_entity_ids)
        for row in ground_truth.itertuples(index=False)
    }
    true_singletons = {s1 for s1 in s1_ids if not gt_map.get(s1)}
    if not true_singletons:
        return float("nan")
    kept = pairs_with_prob[pairs_with_prob["prob"] >= threshold]
    predicted_nonempty = set(kept["source1_entity_id"])
    correct = sum(1 for s1 in true_singletons if s1 not in predicted_nonempty)
    return correct / len(true_singletons)


def _print_summary(result: PipelineResult, args: argparse.Namespace) -> None:
    print("\n" + "=" * 70)
    print("Pipeline run summary")
    print("=" * 70)
    if result.blocking_recall is not None:
        print(f"  Blocking recall (train):       {result.blocking_recall:.4f}")
    if result.holdout_f_beta is not None:
        print(f"  Holdout macro F0.5:             {result.holdout_f_beta:.4f}")
    if result.threshold is not None:
        print(f"  Tuned decision threshold:       {result.threshold:.3f}")
    print(f"  Test S1 rows written:           {result.n_test_rows}")
    print(f"  Output directory:               {result.output_dir}")
    if args.sample_size:
        print(
            f"  NOTE: this was a --sample-size {args.sample_size} subsample run. "
            "utils/validate_submission.py will correctly report missing S1 rows "
            "when pointed at the full test_source1.tsv -- that is expected, not "
            "a pipeline bug. Run without --sample-size for a full submission."
        )
    print(
        "\nNext step: python utils/validate_submission.py --matching "
        f"{result.output_dir / 'matching_results.tsv'} --candidate "
        f"{result.output_dir / 'candidate_pairs.tsv'} --test-dir dataset/test"
    )
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    p.add_argument("--mode", choices=["local", "aws"], default="local")
    p.add_argument("--s3-bucket", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--dataset-dir", default="dataset")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--artifacts-dir", default="code/business_entity_resolution/artifacts")
    p.add_argument("--sample-size", type=int, default=None, help="S1 entities per country for a fast subsample run")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--use-embeddings", action="store_true")
    p.add_argument("--model", choices=["lgbm", "xgb"], default="lgbm")
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--skip-train", action="store_true", help="reuse a saved model artifact, run test inference only")
    p.add_argument("--force-preprocess", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    log_dir = Path.cwd() / args.artifacts_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "pipeline.log")],
    )

    try:
        run_pipeline(args)
    except Exception:
        logger.exception("pipeline run failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
