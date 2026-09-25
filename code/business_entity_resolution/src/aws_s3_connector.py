"""AWS S3 integration: sync the dataset down once, upload outputs once.

Adapted (not copied) from a reference boto3 pattern of a process-wide
singleton client plus a thin service wrapper. Only two operations are
implemented, matching what this pipeline actually needs:

1. ``S3DatasetSync.sync_dataset`` -- if ``dataset/`` is not already present
   locally, download it once from ``s3://<bucket>/dataset/`` to local disk.
   Every downstream pipeline stage then reads local paths exclusively.
   S3 is object storage, not a POSIX filesystem: reading TSVs directly via
   ``s3://...`` per chunk/per worker would turn every read into a
   network-bound GET and dominate runtime, so this module deliberately
   never does that -- it syncs once, then gets out of the way.
2. ``S3DatasetSync.upload_outputs`` -- upload the two result TSVs back to
   ``s3://<bucket>/output/`` after the run completes.

Credentials are **never** read or hardcoded in this file. ``boto3.client("s3")``
/ ``boto3.resource("s3")`` are called with no explicit key arguments, so
boto3's own default credential chain resolves them in order: an IAM role
attached to the EC2 instance first, then the ``AWS_ACCESS_KEY_ID`` /
``AWS_SECRET_ACCESS_KEY`` / ``AWS_DEFAULT_REGION`` environment variables, then
the shared credentials file. This matches the recommended pattern of
preferring an instance IAM role over embedding keys anywhere in code.
"""

from __future__ import annotations

import logging
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

_EXPECTED_TRAIN_FILES = (
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv",
)
_EXPECTED_TEST_FILES = ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv")


class S3ClientSingleton:
    """Lazily-constructed, process-wide boto3 client.

    Credential resolution is left entirely to boto3's default chain (IAM
    role -> env vars -> shared config file); this class never passes
    explicit access keys.
    """

    _client = None

    @classmethod
    def get_client(cls, region_name: str | None = None):
        if cls._client is None:
            try:
                cls._client = boto3.client("s3", region_name=region_name)
            except (BotoCoreError, NoCredentialsError) as exc:
                raise RuntimeError(
                    "AWS credentials not found. Attach an IAM role to this "
                    "instance, or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY "
                    "/ AWS_DEFAULT_REGION."
                ) from exc
        return cls._client


class S3DatasetSync:
    """Dataset download-if-absent + output upload, nothing else."""

    def __init__(self, bucket: str, region_name: str | None = None):
        self.bucket = bucket
        self.client = S3ClientSingleton.get_client(region_name)

    def dataset_present_locally(self, local_dataset_dir: Path) -> bool:
        """Cheap, local-only presence check (no S3 call)."""
        train_dir = local_dataset_dir / "train"
        test_dir = local_dataset_dir / "test"
        for fname in _EXPECTED_TRAIN_FILES:
            p = train_dir / fname
            if not p.is_file() or p.stat().st_size == 0:
                return False
        for fname in _EXPECTED_TEST_FILES:
            p = test_dir / fname
            if not p.is_file() or p.stat().st_size == 0:
                return False
        return True

    def sync_dataset(self, local_dataset_dir: Path, s3_prefix: str = "dataset/") -> None:
        """Download the full dataset from S3 exactly once, if not already local.

        Downloads happen via a paginated ``list_objects_v2`` + one
        ``download_file`` call per object, mirroring the S3 key structure
        onto ``local_dataset_dir``. After this call returns, the pipeline
        never touches S3 again for dataset reads.
        """
        if self.dataset_present_locally(local_dataset_dir):
            logger.info("dataset already present locally at %s; skipping S3 sync", local_dataset_dir)
            return

        logger.info("dataset not found locally; syncing from s3://%s/%s", self.bucket, s3_prefix)
        local_dataset_dir.mkdir(parents=True, exist_ok=True)

        paginator = self.client.get_paginator("list_objects_v2")
        n_downloaded = 0
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith("/"):
                        continue
                    relative = key[len(s3_prefix) :] if key.startswith(s3_prefix) else key
                    dest = local_dataset_dir / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    logger.info("downloading s3://%s/%s -> %s", self.bucket, key, dest)
                    self.client.download_file(self.bucket, key, str(dest))
                    n_downloaded += 1
        except (BotoCoreError, ClientError):
            logger.exception("dataset sync from S3 failed after %d file(s)", n_downloaded)
            raise

        logger.info("dataset sync complete: %d file(s) downloaded", n_downloaded)

    def upload_outputs(self, local_output_dir: Path, s3_prefix: str = "output/") -> None:
        """Upload matching_results.tsv and candidate_pairs.tsv to S3.

        Failures are logged and re-raised (never silently swallowed): a
        failed upload must be visible to the operator running the job.
        """
        for fname in ("matching_results.tsv", "candidate_pairs.tsv"):
            local_path = local_output_dir / fname
            if not local_path.is_file():
                logger.warning("expected output file %s not found; skipping upload", local_path)
                continue
            key = f"{s3_prefix}{fname}"
            try:
                logger.info("uploading %s -> s3://%s/%s", local_path, self.bucket, key)
                self.client.upload_file(str(local_path), self.bucket, key)
            except (BotoCoreError, ClientError):
                logger.exception("failed to upload %s to s3://%s/%s", local_path, self.bucket, key)
                raise
        logger.info("output upload complete")
