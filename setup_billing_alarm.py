"""
Create a low-threshold AWS Budget so an accidentally-left-running resource
surfaces as an email automatically, instead of relying on remembering to run
teardown and eyeball the console every session.

Why a Budget and not a CloudWatch billing alarm: a CloudWatch alarm on
`AWS/Billing EstimatedCharges` needs "Receive Billing Alerts" enabled in
account preferences AND an SNS topic whose email subscription you must click
to confirm — three manual steps that are easy to get half-done. AWS Budgets
emails the address directly, no confirmation click, and the first two budgets
are free. For "tell me before this costs more than a few dollars", Budgets is
the lower-friction, more reliable choice.

Idempotent — safe to re-run. If the budget already exists it is deleted and
recreated so the thresholds always match this file.

    BILLING_ALERT_EMAIL=you@example.com python setup_billing_alarm.py

The email is read from the environment (or .env), never hardcoded — this repo
is public, so a real address must never live in a committed file.

Thresholds: a monthly COST budget (default $10, override with
BILLING_BUDGET_LIMIT). You are emailed when actual spend crosses 50% and 90%
of it, and when the month is *forecast* to exceed 100%. The 50% actual alert
is the one that matters here: the KB's OpenSearch collection runs ~$1/hr, so
a stack left up overnight trips it well before the bill gets large.
"""
import os
import sys

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, "agents")
from aws_config import REGION, IS_GOVCLOUD  # noqa: E402

BUDGET_NAME = "nns-chatbot-monthly-guardrail"


def step(message: str) -> None:
    print(f"\n=== {message} ===")


def main() -> int:
    if IS_GOVCLOUD:
        # AWS Budgets is not offered in GovCloud partitions. There, use a
        # CloudWatch alarm on AWS/Billing EstimatedCharges in us-gov-west-1
        # instead — out of scope for this commercial-account helper.
        print(
            "AWS Budgets is not available in GovCloud. Set up a CloudWatch "
            "billing alarm on AWS/Billing EstimatedCharges instead."
        )
        return 1

    email = os.environ.get("BILLING_ALERT_EMAIL", "").strip()
    if not email:
        print(
            "ERROR: BILLING_ALERT_EMAIL is not set.\n"
            "Set it in .env (gitignored) or pass it inline:\n"
            "  BILLING_ALERT_EMAIL=you@example.com python setup_billing_alarm.py"
        )
        return 1

    try:
        limit = float(os.environ.get("BILLING_BUDGET_LIMIT", "10"))
    except ValueError:
        print("ERROR: BILLING_BUDGET_LIMIT must be a number (dollars).")
        return 1

    account_id = boto3.client("sts", region_name=REGION or None).get_caller_identity()[
        "Account"
    ]
    # Budgets is a global service homed in us-east-1 for the commercial
    # partition, regardless of the app's working region.
    budgets = boto3.client("budgets", region_name="us-east-1")

    # Email me at these points. Actual@50% is the early-warning that catches
    # a stack left running; forecast@100% catches a slow leak before EOM.
    def _notification(ntype: str, threshold: float) -> dict:
        return {
            "Notification": {
                "NotificationType": ntype,
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": threshold,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": [{"SubscriptionType": "EMAIL", "Address": email}],
        }

    notifications = [
        _notification("ACTUAL", 50.0),
        _notification("ACTUAL", 90.0),
        _notification("FORECASTED", 100.0),
    ]

    budget = {
        "BudgetName": BUDGET_NAME,
        "BudgetLimit": {"Amount": str(limit), "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST",
    }

    step(f"Budget '{BUDGET_NAME}' (account {account_id})")
    try:
        budgets.describe_budget(AccountId=account_id, BudgetName=BUDGET_NAME)
        print("Already exists — recreating so thresholds match this script.")
        budgets.delete_budget(AccountId=account_id, BudgetName=BUDGET_NAME)
    except ClientError as error:
        if error.response["Error"]["Code"] != "NotFoundException":
            print(f"ERROR: could not check for an existing budget: {error}")
            return 1

    try:
        budgets.create_budget(
            AccountId=account_id,
            Budget=budget,
            NotificationsWithSubscribers=notifications,
        )
    except ClientError as error:
        print(f"ERROR: could not create the budget: {error}")
        return 1

    print(f"Created ${limit:.0f}/month cost budget.")
    print(f"Alerts email {email} at:")
    print(f"  - actual spend > 50% (${limit * 0.5:.2f})")
    print(f"  - actual spend > 90% (${limit * 0.9:.2f})")
    print(f"  - forecast to exceed 100% (${limit:.2f})")
    print("\nBudget emails need no confirmation click — you're done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
