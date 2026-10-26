"""Module 9, checkpoint 2: throwaway permission test for ONE region. YOU run this, with your ott-dev profile; Claude does not.

Creates, uses and deletes tiny `ott-permtest*` resources (a log group, one SSM parameter holding a dummy value, one DynamoDB table, one 128 MB
Lambda function with a Function URL) to learn which actions your account allows in this region, including whether a service control policy blocks any.
Prints ONLY a label and ALLOWED / DENIED / ERROR(code) per action: no account id, ARN, request id, URL or error message. Cleanup always runs.

    python scripts/aws_permtest.py --region eu-central-1
    python scripts/aws_permtest.py --region ap-south-1

Needs the policy in infra/iam/ott-dev-policy.json on the profile, the role ott-recsys-lambda-role, and a profile created by `aws configure --profile ott-dev`.
If a run is interrupted, delete leftovers with scripts/aws_teardown.py. Expected cost: $0.00 (seconds of tiny resources).

AWS CloudShell: this file is standalone (standard library + boto3 only). Upload just this file and run

    python3 aws_permtest.py --region eu-central-1 --session-credentials

--session-credentials uses boto3's default credential chain (in CloudShell: the console identity's temporary credentials) and no named profile. The
"EC2 must be denied" check is skipped in that mode, because that deny belongs to the ott-dev policy, not to the console role. Results go to ./logs/."""
import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REGIONS = ("eu-central-1", "ap-south-1")
ROLE_NAME = "ott-recsys-lambda-role"
FN, TABLE, LOG_GROUP, PARAM = "ott-permtest", "ott-permtest", "/aws/lambda/ott-permtest", "/ott/permtest"
DUMMY_VALUE = "not-a-secret-permission-test"


def classify(exc) -> str:
    """ALLOWED/DENIED words only. The error text is inspected (never printed) to tell a service control policy from a plain policy denial."""
    err = (getattr(exc, "response", None) or {}).get("Error", {})
    code, text = str(err.get("Code", "")), str(err.get("Message", "")).lower()
    if "AccessDenied" in code or "UnauthorizedOperation" in code or code in ("AuthorizationError", "AccessDeniedException"):
        if "service control policy" in text:
            return "DENIED (service control policy)"
        if "explicit deny" in text:
            return "DENIED (explicit deny)"
        return "DENIED"
    return f"ERROR({code or type(exc).__name__})"


def signed_post(url: str, region: str, credentials) -> int:
    """POST {} to the Function URL signed with SigV4 (service 'lambda'). Returns the HTTP status only."""
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    req = AWSRequest(method="POST", url=url, data=b"{}", headers={"Content-Type": "application/json"})
    SigV4Auth(credentials, "lambda", region).add_auth(req)
    r = urllib.request.Request(url, data=b"{}", headers=dict(req.headers.items()), method="POST")
    try:
        with urllib.request.urlopen(r, timeout=20) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def hello_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("hello.py", "def handler(event, context):\n    return {'statusCode': 200, 'body': 'ok'}\n")
    return buf.getvalue()


class Run:
    def __init__(self, region: str, session, signed_call=signed_post, sleep=time.sleep, session_credentials: bool = False, log_dir="logs"):
        self.region, self.s, self.signed_call, self.sleep = region, session, signed_call, sleep
        self.session_credentials, self.log_dir = session_credentials, Path(log_dir)
        self.results, self.cleanups, self.state = [], [], {}

    def client(self, name):
        try:
            from botocore.config import Config
            cfg = Config(retries={"max_attempts": 3}, connect_timeout=5, read_timeout=30)
            return self.s.client(name, region_name=self.region, config=cfg)
        except ImportError:
            return self.s.client(name, region_name=self.region)

    def step(self, label, fn, needs=None, cleanup=None, expect_denied=False):
        if needs and not self.state.get(needs):
            status = f"SKIPPED (needs {needs})"
        else:
            try:
                out = fn()
                status = "ALLOWED"
                if isinstance(out, str) and out.startswith(("DENIED", "ERROR")):
                    status = out
                if expect_denied:
                    status = "ALLOWED (UNEXPECTED: this should be denied)" if status == "ALLOWED" else status + " (as designed)"
                elif status == "ALLOWED" and cleanup:
                    self.cleanups.append((label, cleanup))
            except Exception as exc:  # classify() reads the error code; nothing is printed except the words
                status = classify(exc)
                if expect_denied and status.startswith("DENIED"):
                    status += " (as designed)"
        self.state[label] = status.startswith("ALLOWED")
        self.results.append((label, status))
        print(f"  {label:<56s} {status}", flush=True)
        return status

    def run_cleanup(self):
        ok = 0
        for label, fn in reversed(self.cleanups):
            try:
                fn()
                res = "REMOVED"
            except Exception as exc:
                res = "ALREADY GONE" if "NotFound" in str((getattr(exc, "response", None) or {}).get("Error", {}).get("Code", "")) else "CLEANUP FAILED " + classify(exc)
            ok += res in ("REMOVED", "ALREADY GONE")
            print(f"  cleanup: {label:<47s} {res}", flush=True)
        return ok, len(self.cleanups)

    # ------------------------------------------------------------------ the steps
    def go(self) -> int:
        print(f"Permission test in {self.region} (profile resources only: ott-permtest*; nothing sensitive is printed)", flush=True)
        sts, logs, ssm, ddb, lam = (self.client(x) for x in ("sts", "logs", "ssm", "dynamodb", "lambda"))
        ec2 = None if self.session_credentials else self.client("ec2")
        acct = {}

        def identity():
            acct["id"] = sts.get_caller_identity()["Account"]  # kept in memory to build the role ARN; never printed
        try:
            self.step("sts get_caller_identity", identity)
            self.step("logs create_log_group", lambda: logs.create_log_group(logGroupName=LOG_GROUP), cleanup=lambda: logs.delete_log_group(logGroupName=LOG_GROUP))
            self.step("logs put_retention_policy (7 days)", lambda: logs.put_retention_policy(logGroupName=LOG_GROUP, retentionInDays=7), needs="logs create_log_group")
            self.step("ssm put_parameter (SecureString, dummy value)",
                      lambda: ssm.put_parameter(Name=PARAM, Value=DUMMY_VALUE, Type="SecureString", Overwrite=True), cleanup=lambda: ssm.delete_parameter(Name=PARAM))
            self.step("ssm get_parameter (with decryption)", lambda: ssm.get_parameter(Name=PARAM, WithDecryption=True)["Parameter"]["Value"] == DUMMY_VALUE or "ERROR(value-mismatch)",
                      needs="ssm put_parameter (SecureString, dummy value)")
            self.step("dynamodb create_table (on-demand)",
                      lambda: ddb.create_table(TableName=TABLE, BillingMode="PAY_PER_REQUEST", AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "N"}],
                                               KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}]), cleanup=lambda: ddb.delete_table(TableName=TABLE))
            self.step("dynamodb wait until active", lambda: ddb.get_waiter("table_exists").wait(TableName=TABLE, WaiterConfig={"Delay": 2, "MaxAttempts": 30}),
                      needs="dynamodb create_table (on-demand)")
            self.step("dynamodb put_item", lambda: ddb.put_item(TableName=TABLE, Item={"pk": {"N": "1"}, "n": {"N": "1"}}), needs="dynamodb wait until active")
            self.step("dynamodb update_item (ADD counter)", lambda: ddb.update_item(TableName=TABLE, Key={"pk": {"N": "1"}}, UpdateExpression="ADD n :o",
                                                                                   ExpressionAttributeValues={":o": {"N": "1"}}), needs="dynamodb wait until active")
            self.step("dynamodb batch_get_item", lambda: ddb.batch_get_item(RequestItems={TABLE: {"Keys": [{"pk": {"N": "1"}}]}}), needs="dynamodb wait until active")
            role_arn = lambda: f"arn:aws:iam::{acct['id']}:role/{ROLE_NAME}"
            self.step("lambda create_function (128 MB, passes the role)",
                      lambda: lam.create_function(FunctionName=FN, Runtime="python3.12", Role=role_arn(), Handler="hello.handler", Code={"ZipFile": hello_zip()},
                                                  MemorySize=128, Timeout=3, Architectures=["x86_64"]),
                      needs="sts get_caller_identity", cleanup=lambda: lam.delete_function(FunctionName=FN))
            self.step("lambda wait until active", lambda: lam.get_waiter("function_active_v2").wait(FunctionName=FN), needs="lambda create_function (128 MB, passes the role)")
            self.step("lambda invoke", lambda: lam.invoke(FunctionName=FN, Payload=b"{}")["StatusCode"] == 200 or "ERROR(bad-status)", needs="lambda wait until active")
            url = {}

            def make_url():
                url["u"] = lam.create_function_url_config(FunctionName=FN, AuthType="AWS_IAM")["FunctionUrl"]  # never printed
            self.step("lambda create_function_url_config (AWS_IAM)", make_url, needs="lambda wait until active",
                      cleanup=lambda: lam.delete_function_url_config(FunctionName=FN))

            def call_url():
                status = 0
                for _ in range(3):  # a brand-new URL can answer 403 for a few seconds
                    status = self.signed_call(url["u"], self.region, self.s.get_credentials().get_frozen_credentials())
                    if status != 403:
                        break
                    self.sleep(3)
                return True if status == 200 else "DENIED" if status == 403 else f"ERROR(http-{status})"
            self.step("lambda function URL: signed request", call_url, needs="lambda create_function_url_config (AWS_IAM)")
            self.step("lambda put_function_concurrency (reserved 1)", lambda: lam.put_function_concurrency(FunctionName=FN, ReservedConcurrentExecutions=1),
                      needs="lambda wait until active")
            if self.session_credentials:  # that deny belongs to the ott-dev policy, not to the console role, so there is nothing to check
                self.results.append(("ec2 describe_instances (must be denied by design)", "SKIPPED (session credentials)"))
                print(f"  {'ec2 describe_instances (must be denied by design)':<56s} SKIPPED (session credentials: the deny belongs to the ott-dev policy)", flush=True)
            else:
                self.step("ec2 describe_instances (must be denied by design)", lambda: ec2.describe_instances(MaxResults=5), expect_denied=True)
        finally:
            print("Cleanup (always runs):", flush=True)
            removed, total = self.run_cleanup()
        counts = {k: sum(1 for _, s in self.results if s.startswith(k)) for k in ("ALLOWED", "DENIED", "ERROR", "SKIPPED")}
        print(f"Summary for {self.region}: {counts['ALLOWED']} allowed, {counts['DENIED']} denied, {counts['ERROR']} errors, {counts['SKIPPED']} skipped; cleanup {removed}/{total}", flush=True)
        log = self.log_dir / f"permtest_{self.region}.json"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps({"region": self.region, "results": dict(self.results), "cleanup": f"{removed}/{total}"}, indent=1))
        return 0 if removed == total and counts["ERROR"] == 0 else 1


def main(argv=None, session_factory=None, signed_call=signed_post, sleep=time.sleep) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--region", required=True, choices=REGIONS)
    ap.add_argument("--profile", default=None, help="named AWS profile (default ott-dev); not allowed together with --session-credentials")
    ap.add_argument("--session-credentials", action="store_true",
                    help="use boto3's default credential chain (e.g. the AWS CloudShell console identity); no profile; skips the EC2 must-be-denied check")
    ap.add_argument("--log-dir", default="logs", help="where the result file is written (relative to the current folder)")
    args = ap.parse_args(argv)
    if args.session_credentials and args.profile:
        ap.error("--session-credentials and --profile cannot be used together")
    profile = None if args.session_credentials else (args.profile or "ott-dev")
    if session_factory is None:
        import boto3
        session_factory = lambda profile, region: boto3.Session(profile_name=profile, region_name=region)  # profile None = default chain
    try:
        session = session_factory(profile, args.region)
    except Exception as exc:
        hint = "check that this shell has AWS credentials" if profile is None else f"create it with: aws configure --profile {profile}"
        print(f"could not open the {'default credentials' if profile is None else 'profile ' + repr(profile)}: {type(exc).__name__} ({hint})")
        return 2
    print("credentials: " + ("default chain (no profile)" if profile is None else f"profile {profile}"), flush=True)
    return Run(args.region, session, signed_call, sleep, args.session_credentials, args.log_dir).go()


if __name__ == "__main__":
    sys.exit(main())
