"""Business Entity Resolution pipeline package.

Modules:
    aws_s3_connector    -- S3 dataset sync / output upload (AWS mode only)
    preprocessing       -- Unicode-safe text normalization, chunked TSV -> Parquet
    blocking            -- multi-channel candidate generation (exact keys, TF-IDF
                            sparse blocking, optional multilingual embedding ANN)
    feature_engineering -- vectorized pairwise similarity features
    model               -- LightGBM/XGBoost classifier + F0.5 threshold tuning
    pipeline            -- CLI entry point orchestrating the end-to-end run
"""

__version__ = "1.0.0"
