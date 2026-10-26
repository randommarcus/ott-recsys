"""Module 9: store the read-only Zilliz settings in ONE SSM SecureString parameter (/ott/zilliz) for the Lambda function. YOU run this.

Reads ZILLIZ_URI and ZILLIZ_TOKEN_READONLY from .env inside Python (the values never appear on a command line, in shell history, or in any output),
writes {"uri": ..., "token": ...} with the AWS-managed key, and prints only `stored: yes/no` plus an error TYPE on failure. It refuses to store the
full-access ZILLIZ_TOKEN by mistake: if the read-only token is blank or equal to ZILLIZ_TOKEN it stops without calling AWS.

    python scripts/put_ssm_secrets.py --region eu-central-1
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ("eu-central-1", "ap-south-1")
PARAM_NAME = "/ott/zilliz"


def load_env():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")


def main(argv=None, session_factory=None, env=None, env_loader=load_env) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--region", required=True, choices=REGIONS)
    ap.add_argument("--profile", default="ott-dev")
    args = ap.parse_args(argv)
    if env is None:
        env_loader()
        env = os.environ
    uri, token, full = env.get("ZILLIZ_URI", ""), env.get("ZILLIZ_TOKEN_READONLY", ""), env.get("ZILLIZ_TOKEN", "")
    if not uri or not token:
        print("stored: no (ZILLIZ_URI and ZILLIZ_TOKEN_READONLY must both be set in .env)")
        return 1
    if token == full:
        print("stored: no (ZILLIZ_TOKEN_READONLY is identical to the full-access ZILLIZ_TOKEN; create a Read-Only key in the Zilliz console)")
        return 1
    value = json.dumps({"uri": uri, "token": token})
    if len(value) > 4000:
        print("stored: no (the settings are too large for a standard SSM parameter)")
        return 1
    try:
        if session_factory is None:
            import boto3
            session_factory = lambda profile, region: boto3.Session(profile_name=profile, region_name=region)
        ssm = session_factory(args.profile, args.region).client("ssm", region_name=args.region)
        ssm.put_parameter(Name=PARAM_NAME, Value=value, Type="SecureString", Overwrite=True)
    except Exception as exc:
        code = str((getattr(exc, "response", None) or {}).get("Error", {}).get("Code", "")) or type(exc).__name__
        print(f"stored: no (error: {code})")
        return 1
    print(f"stored: yes ({PARAM_NAME} in {args.region})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
