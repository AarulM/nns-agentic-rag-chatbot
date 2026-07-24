"""
Create (or repair) the S3 bucket that stages multimodal uploads.

Deliberately NOT part of the CDK stack: this bucket holds the extraction
cache, and keeping it outside CloudFormation means `teardown_everything.py`
leaves it alone. Rebuilding the stack therefore does not throw away work
you have already paid Bedrock Data Automation to do.

Idempotent — safe to re-run. Every step is get-or-create, so running it
after a partial failure finishes the job.

    python setup_upload_bucket.py

Lifecycle policy, and why these numbers:

  uploads/        1 day    Staged originals — the raw file bytes, PII and
                           all. file_ingest deletes each one the instant BDA
                           finishes reading it (success or failure), so in
                           the normal path they do not linger at all; this
                           rule is only a backstop for a process that dies
                           mid-job. Kept short because the content is
                           unredactable raw PII, not for storage reasons.
  bda-output/     1 day    Raw BDA result JSON — the extracted text *before*
                           PII redaction. Also proactively deleted the moment
                           it is parsed into memory; 1 day is the backstop.
  extract-cache/  kept     Content-hash cache of extracted text, already
                           PII-redacted (see pii_redaction.py). This is the
                           money saver: a cache hit turns a re-upload into
                           one S3 GET instead of a fresh extraction. Small
                           JSON, so it costs almost nothing to retain.

Why 1 day and not 7: uploads/ and bda-output/ hold unredacted content, and
the redaction work (pii_redaction.py) is wasted if the pre-redaction source
is left sitting in S3. The proactive deletes above do the real work; the
short expiry only bounds how long an orphan from a crashed job can survive.

Note on storage classes: there is deliberately no Standard-IA transition
here. IA bills a 128 KB minimum per object and requires 30 days in Standard
first — so for the cache (objects well under 128 KB) it would *raise* the
bill, and for uploads/ (deleted at 7 days) a 30-day transition can never
fire. Expiration is the correct lever for this bucket, not tiering.
"""
import sys

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, "agents")
from aws_config import REGION, UPLOAD_BUCKET  # noqa: E402

LIFECYCLE_RULES = [
    {
        # Unredacted raw file bytes. file_ingest deletes these right after
        # extraction; 1 day only bounds orphans from a crashed job.
        "ID": "expire-staged-uploads",
        "Filter": {"Prefix": "uploads/"},
        "Status": "Enabled",
        "Expiration": {"Days": 1},
    },
    {
        # Pre-redaction extracted text. Also deleted right after parsing;
        # 1 day is the backstop.
        "ID": "expire-bda-output",
        "Filter": {"Prefix": "bda-output/"},
        "Status": "Enabled",
        "Expiration": {"Days": 1},
    },
    {
        # A failed multipart upload of a large video leaves parts behind
        # that are invisible in the console but still billed.
        "ID": "abort-incomplete-multipart",
        "Filter": {"Prefix": ""},
        "Status": "Enabled",
        "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 3},
    },
]


def step(message: str) -> None:
    print(f"\n=== {message} ===")


def main() -> int:
    if not REGION:
        print("ERROR: AWS_REGION is not set. Fill it in .env first.")
        return 1
    if not UPLOAD_BUCKET:
        print(
            "ERROR: UPLOAD_BUCKET is not set in .env.\n"
            "Pick a globally-unique name (e.g. nns-multimodal-<account-id>) "
            "and set it there, then re-run this script."
        )
        return 1

    s3 = boto3.client("s3", region_name=REGION)

    step(f"Bucket {UPLOAD_BUCKET} ({REGION})")
    try:
        s3.head_bucket(Bucket=UPLOAD_BUCKET)
        print("Already exists — leaving contents alone.")
    except ClientError as error:
        if error.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
            print(f"ERROR: cannot access bucket: {error}")
            return 1
        # us-east-1 is the one region that rejects an explicit
        # LocationConstraint, so it needs the argument omitted entirely.
        kwargs = {"Bucket": UPLOAD_BUCKET}
        if REGION != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": REGION}
        s3.create_bucket(**kwargs)
        print("Created.")

    step("Blocking all public access")
    s3.put_public_access_block(
        Bucket=UPLOAD_BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    print("Done.")

    step("Enabling default encryption")
    s3.put_bucket_encryption(
        Bucket=UPLOAD_BUCKET,
        ServerSideEncryptionConfiguration={
            "Rules": [
                {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
            ]
        },
    )
    print("Done.")

    step("Applying lifecycle rules")
    s3.put_bucket_lifecycle_configuration(
        Bucket=UPLOAD_BUCKET, LifecycleConfiguration={"Rules": LIFECYCLE_RULES}
    )
    for rule in LIFECYCLE_RULES:
        prefix = rule["Filter"].get("Prefix") or "(whole bucket)"
        if "Expiration" in rule:
            print(f"  {prefix:<16} expire after {rule['Expiration']['Days']} days")
        else:
            days = rule["AbortIncompleteMultipartUpload"]["DaysAfterInitiation"]
            print(f"  {prefix:<16} abort incomplete uploads after {days} days")

    print(f"\nUPLOAD_BUCKET={UPLOAD_BUCKET} is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
