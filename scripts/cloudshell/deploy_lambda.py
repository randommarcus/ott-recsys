"""AWS CloudShell, one standalone file (standard library + boto3). Creates / updates the Lambda function ott-recsys-serve and smoke-tests it.

Upload this file and lambda.zip (from scripts/build_lambda_zip.py) into ONE folder, then:

    python3 deploy_lambda.py --region eu-central-1 create          # log group (7-day retention), function, Function URL (AWS_IAM only)
    python3 deploy_lambda.py --region eu-central-1 smoke --user-id 1   # calls the function directly (no URL needed) and prints counts and timings
    python3 deploy_lambda.py --region eu-central-1 update-code     # after a new lambda.zip
    python3 deploy_lambda.py --region eu-central-1 show-url        # prints the Function URL (keep it out of the repository)

The function runs as the role ott-recsys-lambda-role (you created it in the console); the account id needed to name that role is read in memory and never
printed. Only the fixed names ott-recsys-serve and /aws/lambda/ott-recsys-serve are used. No provisioned concurrency. Prints words and counts only.
Uses boto3's default credential chain (the console identity)."""
import argparse
import json
import sys
from pathlib import Path

REGIONS = ("eu-central-1", "ap-south-1")
FN, LOG_GROUP, ROLE_NAME = "ott-recsys-serve", "/aws/lambda/ott-recsys-serve", "ott-recsys-lambda-role"
HANDLER = "ott_recsys.serving.handler.handler"


def code_of(exc) -> str:
    return str((getattr(exc, "response", None) or {}).get("Error", {}).get("Code", "")) or type(exc).__name__


def word(exc) -> str:
    c = code_of(exc)
    return "DENIED" if "AccessDenied" in c or "Unauthorized" in c else f"ERROR({c})"


def step(label, fn) -> bool:
    try:
        fn()
        print(f"  {label:<52s} ok", flush=True)
        return True
    except Exception as exc:
        print(f"  {label:<52s} {word(exc)}", flush=True)
        return False


def create(session, region, zip_path: Path, memory: int, timeout: int) -> int:
    sts, logs, lam = (session.client(x, region_name=region) for x in ("sts", "logs", "lambda"))
    acct = {}
    ok = step("identity", lambda: acct.update(id=sts.get_caller_identity()["Account"]))
    if not ok:
        return 1
    try:
        logs.create_log_group(logGroupName=LOG_GROUP)
        print(f"  {'log group':<52s} created", flush=True)
    except Exception as exc:
        if code_of(exc) != "ResourceAlreadyExistsException":
            print(f"  {'log group':<52s} {word(exc)}")
            return 1
        print(f"  {'log group':<52s} already exists", flush=True)
    if not step("log retention 7 days", lambda: logs.put_retention_policy(logGroupName=LOG_GROUP, retentionInDays=7)):
        return 1
    code = zip_path.read_bytes()
    print(f"  zip: {len(code) / 2 ** 20:.1f} MiB", flush=True)
    if not step("create function", lambda: lam.create_function(FunctionName=FN, Runtime="python3.12", Handler=HANDLER, Code={"ZipFile": code},
                                                                Role=f"arn:aws:iam::{acct['id']}:role/{ROLE_NAME}", MemorySize=memory, Timeout=timeout,
                                                                Architectures=["x86_64"], Description="ott-recsys serving (Module 9)")):
        return 1
    ok = step("wait until active", lambda: lam.get_waiter("function_active_v2").wait(FunctionName=FN))
    ok = step("function URL (AuthType AWS_IAM)", lambda: lam.create_function_url_config(FunctionName=FN, AuthType="AWS_IAM")) and ok
    step("reserved concurrency 2 (optional cost guard)", lambda: lam.put_function_concurrency(FunctionName=FN, ReservedConcurrentExecutions=2))
    return 0 if ok else 1


def update_code(session, region, zip_path: Path) -> int:
    lam = session.client("lambda", region_name=region)
    ok = step("update function code", lambda: lam.update_function_code(FunctionName=FN, ZipFile=zip_path.read_bytes()))
    return 0 if ok and step("wait until updated", lambda: lam.get_waiter("function_updated_v2").wait(FunctionName=FN)) else 1


def event(method: str, path: str, body: dict) -> bytes:
    return json.dumps({"requestContext": {"http": {"method": method}}, "rawPath": path, "body": json.dumps(body)}).encode()


def smoke(session, region, user_id: int) -> int:
    lam = session.client("lambda", region_name=region)
    bad = 0
    for label, payload in (("GET /health", event("GET", "/health", {})),
                           ("POST /recommend, precomputed, M2", event("POST", "/recommend", {"user_id": user_id, "record_impressions": False})),
                           ("POST /recommend, precomputed, M3b (SIMULATED device)", event("POST", "/recommend", {"user_id": user_id, "model": "M3b", "device": "mobile", "record_impressions": False}))):
        try:
            r = lam.invoke(FunctionName=FN, Payload=payload)
            out = json.loads(r["Payload"].read())
            body = json.loads(out.get("body", "{}"))
            if "FunctionError" in r or out.get("statusCode") != 200:
                print(f"  {label:<52s} status {out.get('statusCode', '?')}" + (f" ({body.get('error', '')[:60]})" if isinstance(body, dict) and out.get("statusCode") in (400, 404) else ""))
                bad += 1
                continue
            t = body.get("timings_ms", {})
            extra = f"{len(body['items'])} items, server {t.get('server_total')} ms, cold start {body['diagnostics']['cold_start']}" if "items" in body else "ok"
            print(f"  {label:<52s} 200: {extra}", flush=True)
        except Exception as exc:
            print(f"  {label:<52s} {word(exc)}")
            bad += 1
    return 1 if bad else 0


def show_url(session, region) -> int:
    try:
        print(session.client("lambda", region_name=region).get_function_url_config(FunctionName=FN)["FunctionUrl"])
        return 0
    except Exception as exc:
        print(f"show-url: {word(exc)}")
        return 1


def main(argv=None, session_factory=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--region", required=True, choices=REGIONS)
    ap.add_argument("--zip", default="lambda.zip")
    ap.add_argument("--memory", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--user-id", type=int, default=1)
    ap.add_argument("action", choices=["create", "update-code", "smoke", "show-url"])
    args = ap.parse_args(argv)
    if session_factory is None:
        import boto3
        session_factory = lambda region: boto3.Session(region_name=region)
    session = session_factory(args.region)
    print(f"region {args.region}, credentials: default chain (no profile)")
    z = Path(args.zip)
    if args.action in ("create", "update-code") and not z.exists():
        print(f"{z} not found: upload lambda.zip next to this script")
        return 2
    return {"create": lambda: create(session, args.region, z, args.memory, args.timeout), "update-code": lambda: update_code(session, args.region, z),
            "smoke": lambda: smoke(session, args.region, args.user_id), "show-url": lambda: show_url(session, args.region)}[args.action]()


if __name__ == "__main__":
    sys.exit(main())
