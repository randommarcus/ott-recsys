"""Module 9: find and delete every `ott-*` resource this project can have created. DRY RUN BY DEFAULT: it only reports what exists.

    python scripts/aws_teardown.py                      # report, both regions
    python scripts/aws_teardown.py --region eu-central-1
    python scripts/aws_teardown.py --confirm DELETE-OTT # delete, then re-check and print 'remaining: N'

Looks only at exact names (it never lists accounts; the policy's names-only list calls are for a manual leftovers check): function ott-recsys-serve and ott-permtest (and their Function URLs), log groups
/aws/lambda/ott-recsys-serve and /aws/lambda/ott-permtest, DynamoDB tables ott-users, ott-bandit, ott-permtest, SSM parameters /ott/zilliz and /ott/permtest.
NOT touched: the IAM role, the ott-dev user and its key, the budget alert, and the Zilliz collections (see the manual checklist in reports/module9_plan.md).
Prints names and exists / absent / unknown(code) only: no account id, ARN or request id."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ("eu-central-1", "ap-south-1")
FUNCTIONS = ("ott-recsys-serve", "ott-permtest")
LOG_GROUPS = ("/aws/lambda/ott-recsys-serve", "/aws/lambda/ott-permtest")
TABLES = ("ott-users", "ott-bandit", "ott-permtest")
PARAMS = ("/ott/zilliz", "/ott/permtest")
CONFIRM_WORD = "DELETE-OTT"


def code_of(exc) -> str:
    return str((getattr(exc, "response", None) or {}).get("Error", {}).get("Code", "")) or type(exc).__name__


def probe(fn) -> str:
    """exists / absent / unknown(code)."""
    try:
        fn()
        return "exists"
    except Exception as exc:
        c = code_of(exc)
        return "absent" if "NotFound" in c or c == "ParameterNotFound" else f"unknown({c})"


class Teardown:
    def __init__(self, region, session):
        self.region = region
        mk = lambda n: session.client(n, region_name=region)
        self.lam, self.logs, self.ddb, self.ssm = mk("lambda"), mk("logs"), mk("dynamodb"), mk("ssm")

    def inventory(self) -> list:
        items = []
        for f in FUNCTIONS:
            items.append(("lambda function", f, probe(lambda f=f: self.lam.get_function(FunctionName=f))))
            items.append(("function URL", f, probe(lambda f=f: self.lam.get_function_url_config(FunctionName=f))))
        items += [("log group", g, probe(lambda g=g: self.logs.describe_log_streams(logGroupName=g, limit=1))) for g in LOG_GROUPS]
        items += [("dynamodb table", t, probe(lambda t=t: self.ddb.describe_table(TableName=t))) for t in TABLES]
        items += [("ssm parameter", p, probe(lambda p=p: self.ssm.get_parameter(Name=p))) for p in PARAMS]  # no decryption: only existence
        return items

    def delete(self, kind, name) -> str:
        fn = {"function URL": lambda: self.lam.delete_function_url_config(FunctionName=name),
              "lambda function": lambda: self.lam.delete_function(FunctionName=name),
              "log group": lambda: self.logs.delete_log_group(logGroupName=name),
              "dynamodb table": lambda: self.ddb.delete_table(TableName=name),
              "ssm parameter": lambda: self.ssm.delete_parameter(Name=name)}[kind]
        try:
            fn()
            return "deleted"
        except Exception as exc:
            c = code_of(exc)
            return "already gone" if "NotFound" in c or c == "ParameterNotFound" else f"FAILED({c})"


ORDER = {"function URL": 0, "lambda function": 1, "log group": 2, "dynamodb table": 3, "ssm parameter": 4}  # URL before function, tables before nothing else depends on them


def main(argv=None, session_factory=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--region", choices=REGIONS + ("all",), default="all")
    ap.add_argument("--profile", default="ott-dev")
    ap.add_argument("--confirm", default=None, help=f"type {CONFIRM_WORD} to delete; without it nothing is changed")
    args = ap.parse_args(argv)
    if args.confirm is not None and args.confirm != CONFIRM_WORD:
        print(f"refused: --confirm must be exactly {CONFIRM_WORD}")
        return 2
    delete = args.confirm == CONFIRM_WORD
    if session_factory is None:
        import boto3
        session_factory = lambda profile, region: boto3.Session(profile_name=profile, region_name=region)
    regions = REGIONS if args.region == "all" else (args.region,)
    remaining_total = 0
    print("DELETE MODE" if delete else "DRY RUN (nothing is changed; add --confirm DELETE-OTT to delete)")
    for region in regions:
        try:
            td = Teardown(region, session_factory(args.profile, region))
        except Exception as exc:
            print(f"{region}: could not open the profile {args.profile!r}: {type(exc).__name__}")
            return 2
        print(f"== {region} ==")
        inv = td.inventory()
        for kind, name, state in inv:
            print(f"  {kind:<16s} {name:<34s} {state}")
        if delete:
            for kind, name, state in sorted(inv, key=lambda x: ORDER[x[0]]):
                if state == "exists" or state.startswith("unknown"):  # an 'unknown' may still be deletable; try, and report
                    print(f"  delete {kind:<16s} {name:<34s} {td.delete(kind, name)}")
            inv = td.inventory()
            print("  re-check:")
            for kind, name, state in inv:
                if state != "absent":
                    print(f"  {kind:<16s} {name:<34s} {state}")
        left = sum(1 for _, _, s in inv if s != "absent")
        remaining_total += left if delete else sum(1 for _, _, s in inv if s == "exists")
        print(f"  {'remaining' if delete else 'existing'}: {left if delete else sum(1 for _, _, s in inv if s == 'exists')}")
    if delete:
        print(f"remaining: {remaining_total}")
        return 0 if remaining_total == 0 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
