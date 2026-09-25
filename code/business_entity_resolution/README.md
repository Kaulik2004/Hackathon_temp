# Business Entity Resolution — Pipeline README

Team: **Hackers** — Amazon ML Challenge 2026

## Overview

Resolves Source-1 (reference) business records against noisy Source-2/Source-3
records at scale (~10.3M pooled target records). The pipeline never
constructs a dense similarity matrix; it uses a multi-channel blocking stage
(exact keys + TF-IDF character-n-gram sparse matmul + optional multilingual
embedding ANN) to produce a bounded candidate set per Source-1 entity, then a
LightGBM/XGBoost classifier scores each candidate pair, and a single
probability threshold — tuned directly against the competition's real
per-entity macro F₀.₅ metric — decides the final matches.

## 1. Environment setup

```powershell
# from the project root (one level above student_resource/)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r student_resource\code\business_entity_resolution\requirements.txt
```

To skip the optional embedding channel entirely (recommended for a first,
fast CPU-only run), comment out or remove the last three lines of
`requirements.txt` (`sentence-transformers`, `torch`, `faiss-cpu`) before
installing — the default blocking path works standalone without them.

## 2. Data layout expected

```
student_resource/
├── dataset/
│   ├── train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
│   └── test/{test_source1,test_source2,test_source3}.tsv
├── output/                (created by the pipeline)
├── utils/validate_submission.py
└── code/business_entity_resolution/   (this package)
```

In `--mode aws`, the dataset is downloaded automatically from
`s3://<bucket>/dataset/` if not already present locally — you don't need to
pre-populate `dataset/` yourself in that case.

## 3. Quick start (local, fast, no embeddings)

From `student_resource/`:

```powershell
python code\business_entity_resolution\run_pipeline.py --mode local
```

This runs the **full** train → validate → test pipeline against the entire
dataset (2.2M/1.7M+ rows). On a small machine this can take a long time and
use significant memory — see §4 for a fast subsample validation run instead,
and §6 for the recommended full-scale AWS run.

## 4. Local subsample validation run

Proves the pipeline is correct end-to-end (schema, no leakage, sane blocking
recall, correctly computed F₀.₅) without processing the full dataset:

```powershell
python code\business_entity_resolution\run_pipeline.py --mode local --sample-size 20000
```

This samples ~20,000 Source-1 entities per country from `train_source1.tsv`,
builds a bounded (ground-truth-guaranteed + distractor) target pool, runs
blocking → feature engineering → training → threshold tuning on a
Source-1-level holdout split, and prints:

- **Blocking recall** — fraction of true training matches present in the
  candidate set (the hard recall ceiling for everything downstream)
- **Holdout macro F₀.₅** — computed with the exact competition formula
- **Precision / recall** at the tuned threshold, and **singleton accuracy**

It then also runs the identical test-inference code path against a
`test_source1.tsv` subsample, purely to prove `matching_results.tsv` /
`candidate_pairs.tsv` are written correctly. Note:
`utils/validate_submission.py` will correctly report "missing S1 rows" if
pointed at the full test set after a `--sample-size` run — that's expected
(the run only covered a subsample), not a pipeline bug. Run without
`--sample-size` for a submission-ready output.

## 5. Enabling the embedding blocking channel

```powershell
python code\business_entity_resolution\run_pipeline.py --mode local --use-embeddings
```

Downloads `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(Apache-2.0, ~118M parameters, ~470MB) on first use. This channel is what
retrieves true matches where the business name changes script entirely
between sources (e.g. a Latin-script Source-1 name vs. a Devanagari-script
Source-2 name for the same India business) — token-overlap blocking alone
cannot find these. It's optional because it's slow on CPU; a GPU is
auto-detected and used automatically if present, but is not required.

## 6. Full-scale AWS run

```powershell
python code\business_entity_resolution\run_pipeline.py --mode aws --s3-bucket <bucket-name> --region us-east-1
```

- Downloads `dataset/` from `s3://<bucket>/dataset/` once (skipped if already
  present locally), runs the full pipeline against local disk, then uploads
  `output/matching_results.tsv` and `output/candidate_pairs.tsv` to
  `s3://<bucket>/output/`.
- **Credentials**: never hardcoded. Set via environment variables
  (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`) or,
  preferably, attach an IAM role to the EC2 instance with `s3:GetObject` +
  `s3:ListBucket` on `dataset/*` and `s3:PutObject` on `output/*` — boto3's
  default credential chain resolves the role automatically with no code
  changes.
- **Instance sizing**: `c6i.2xlarge` (8 vCPU / 16GB) or larger for the
  default (no-embeddings) path; the embedding channel runs on CPU without a
  GPU but is faster with one — `g4dn.xlarge` if you want to enable
  `--use-embeddings` for the full run.
- Add `--use-embeddings` to enable the cross-script embedding channel on the
  full run.

## 7. Validating output

From `student_resource/`, after any run:

```powershell
python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test
```

Prints `PASS` (exit 0) when safe to submit, or a numbered list of issues to
fix. The pipeline also runs its own internal fast-fail checks (subset
property, no self-matches, no duplicate ids/rows, full S1 coverage) before
writing files, but this external validator is the authoritative final gate.

## 8. Module reference (`src/`)

| Module | Responsibility |
| --- | --- |
| `preprocessing.py` | Unicode-safe text normalization (NFKC + casefold, never transliteration/byte-range stripping), legal-suffix/DBA/address normalization, chunked TSV → Parquet streaming |
| `blocking.py` | Country partitioning (open-set safe), exact-key hash-join channel, TF-IDF sparse-matmul channel (never densified), optional multilingual-embedding ANN channel — all unioned per Source-1 entity |
| `feature_engineering.py` | Vectorized `rapidfuzz` pairwise similarity features (name + address), per-S1 aggregate/competitive-context features |
| `model.py` | LightGBM/XGBoost training, S1-entity-level holdout split, threshold tuning against the real per-entity macro F₀.₅ |
| `aws_s3_connector.py` | S3 dataset sync (download-once) + output upload, IAM-role-first credential resolution |
| `pipeline.py` | CLI orchestration of the full flow, output writing, internal validation checks |

## 9. Reproducing from scratch

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r student_resource\code\business_entity_resolution\requirements.txt
cd student_resource
python code\business_entity_resolution\run_pipeline.py --mode local --sample-size 20000   # fast validation
python code\business_entity_resolution\run_pipeline.py --mode local                       # full local run (large machine) or use --mode aws
python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test
```

## 10. Known limitations

- Blocking partitions strictly by country; a true match with an inconsistent
  country label across sources would be missed. No cross-country fallback is
  implemented (documented trade-off for tractability at this scale).
- A single global probability threshold is used rather than a per-entity
  adaptive cutoff (e.g. via `score_gap_to_next_best`) — kept simple and
  auditable; flagged as a future-work extension.
- The embedding channel is optional and CPU-slow without a GPU; the default
  fast path relies on address-based token overlap as a partial fallback for
  cross-script name pairs.
