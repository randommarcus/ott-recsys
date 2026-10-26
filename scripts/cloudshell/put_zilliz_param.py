"""AWS CloudShell, one standalone file (standard library + boto3). Stores the READ-ONLY Zilliz settings in ONE SSM SecureString parameter, /ott/zilliz.

You are asked for the cluster URI and the read-only key with HIDDEN prompts (nothing you type is echoed or stored in the shell history). The key is asked
twice to catch typos. Nothing is printed except `stored: yes/no` and, on failure, an error code. Uses boto3's default credential chain (the console identity).

    python3 put_zilliz_param.py --region eu-central-1

Use the key you created with the Zilliz console role Read-Only, NOT your full-access key: this script cannot tell the difference, so double-check in Zilliz."""
import argparse
import getpass
import json
import sys

REGIONS = ("eu-central-1", "ap-south-1")
PARAM_NAME = "/ott/zilliz"


def main(argv=None, session_factory=None, ask=getpass.getpass) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--region", required=True, choices=REGIONS)
    args = ap.parse_args(argv)
    uri = ask("Zilliz cluster URI (hidden): ").strip()
    key = ask("Zilliz READ-ONLY key (hidden): ").strip()
    again = ask("Type the READ-ONLY key again (hidden): ").strip()
    if not uri or not key:
        print("stored: no (both values are required)")
        return 1
    if key != again:
        print("stored: no (the two keys did not match)")
        return 1
    if not uri.startswith("https://"):
        print("stored: no (the URI must start with https://)")
        return 1
    value = json.dumps({"uri": uri, "token": key})
    if len(value) > 4000:
        print("stored: no (too large for a standard SSM parameter)")
        return 1
    try:
        if session_factory is None:
            import boto3
            session_factory = lambda region: boto3.Session(region_name=region)
        session_factory(args.region).client("ssm", region_name=args.region).put_parameter(Name=PARAM_NAME, Value=value, Type="SecureString", Overwrite=True)
    except Exception as exc:
        code = str((getattr(exc, "response", None) or {}).get("Error", {}).get("Code", "")) or type(exc).__name__
        print(f"stored: no (error: {code})")
        return 1
    print(f"stored: yes ({PARAM_NAME} in {args.region})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
