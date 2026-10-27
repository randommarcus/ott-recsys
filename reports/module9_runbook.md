# Module 9 runbook: deploying the serving layer

Status (2026-10-27): the PRIMARY route is the laptop with the IAM user `ott-dev` (profile `ott-dev`) in a NEW standalone AWS account (standard sign-up, root user, no
organisation). The previous account, which sat under an organisation service control policy, was closed; its evidence is in "History" at the end. The AWS CloudShell route
(Part B) is kept only as a FALLBACK. Nothing here has been run by me: I never hold AWS credentials, and I have not looked at the account.

Account facts that matter: you have USD $100 of sign-up credits. AWS offers a Free account plan and a Paid plan; on the Free plan the account closes automatically when the
credits are used up or after six months, whichever comes first, and it is upgraded to Paid automatically if it ever joins AWS Organizations
([AWS Free Tier plans](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html), read 2026-10-27). Which plan your account is on, and whether
Lambda, DynamoDB, SSM and CloudWatch are all available on it, are **not confirmed** (check Billing > Free Tier). Plan to tear everything down after the demo either way.
Account setup (what to create by hand, in what order, what not to create): `reports/module9_account_setup.md`. IAM drafts: `reports/module9_iam.md`.

---

# Part A: the laptop route (PRIMARY)

All commands run from the repo folder in your own terminal. The scripts use the profile `ott-dev` by default (`--profile` is optional) and print only words and counts
(no account id, ARN, request id or URL).

## Checkpoint 2: permission tests, in this order (about 2 minutes per Region, $0.00)
1. A read-only check that the profile works and nothing is left over. Every line should say `absent`; the last line `existing: 0`:
```bash
.venv/bin/python scripts/aws_teardown.py
```
2. Frankfurt:
```bash
.venv/bin/python scripts/aws_permtest.py --region eu-central-1
```
3. Mumbai:
```bash
.venv/bin/python scripts/aws_permtest.py --region ap-south-1
```
4. Confirm nothing was left behind (must end with `existing: 0` and no `unverified`):
```bash
.venv/bin/python scripts/aws_teardown.py
```
Paste me the output of steps 2 and 3 (results are also written to `logs/permtest_<region>.json`, git-ignored, labels only).

**What to expect.** In a standalone account the only expected difference from the old account is that the tests PASS: every action `ALLOWED`, the deliberate EC2 call
`DENIED (as designed)` (our own policy), and `cleanup N/N`. Things that could still show up and are NOT policy problems: (a) `lambda put_function_concurrency` can fail with a quota
error because new accounts start with reduced Lambda concurrency quotas ([Lambda quotas](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)); it is an optional cost guard
and the script exit code will be 1 because of the ERROR line; (b) `ssm get_parameter (with decryption)` could need a KMS permission (unconfirmed); (c) a service that the Free plan
does not offer would show as an error naming the code. If a step says `DENIED` with no "(as designed)", paste it to me; do not edit the policy yourself.

## Later steps (each needs your explicit go). Region: eu-central-1
The three scripts under `scripts/cloudshell/` are standalone single files that use boto3's default credential chain; on the laptop you point that chain at your profile with
the `AWS_PROFILE` variable. The laptop already has the files (`build/lambda.zip`, `build/dynamodb/*`; rebuild with `scripts/build_lambda_zip.py --no-smoke` and `scripts/export_dynamodb_items.py`).

1. **Tables and data** (62.3 MiB in four parts, about 84,600 write request units; on-demand writes cost roughly a dollar-ish per million units, price not verified, so well under $0.50):
```bash
AWS_PROFILE=ott-dev .venv/bin/python scripts/cloudshell/load_dynamodb.py --region eu-central-1 --create-tables
```
```bash
AWS_PROFILE=ott-dev .venv/bin/python scripts/cloudshell/load_dynamodb.py --region eu-central-1 --load --dir build/dynamodb
```
```bash
AWS_PROFILE=ott-dev .venv/bin/python scripts/cloudshell/load_dynamodb.py --region eu-central-1 --check --dir build/dynamodb
```
2. **Zilliz collections (checkpoint 3).** First, in the Zilliz console, create a customized API key with the project role **Data Viewer**, restricted to this one cluster, and put it in `.env` as `ZILLIZ_TOKEN_READONLY` (never paste it into chat). Then:
```bash
.venv/bin/python scripts/load_zilliz.py
```
   Dry run: three `integrity: ok` lines, `connection: yes; collections in the cluster: N (ours already present: none ...)`, and `DRY RUN: nothing written ...`.
```bash
.venv/bin/python scripts/load_zilliz.py --confirm LOAD-ZILLIZ
```
   FIRST ZILLIZ WRITE ($0, about 2-5 min): three `loaded: ... rows (expected ...) ok` lines and `load complete: 3 collections, AUTOINDEX (metric IP), loaded`.
```bash
.venv/bin/python scripts/verify_zilliz.py --recall
```
   Read-only key only: lines A1, A2, B each ending `-> PASS`, a `C recall` line (informational; AUTOINDEX search is approximate, so overlap may be a little under 100%), and `verdict: PASS`.
   If A1/A2 say the key may not have search rights, the Data Viewer key cannot search: stop and tell me (decision in `reports/module9_cp3_plan.md` section 0.2).
   If B says `ALLOWED (UNEXPECTED ...)`, the key can write: change its role in Zilliz before going on. `INCONCLUSIVE` means a different error: paste the line.
   Undo at any time: `.venv/bin/python scripts/zilliz_teardown.py` (dry run) and `--confirm DROP-ZILLIZ` (drops only the three project collections).
   **After a writable key dirtied the probe** (first run, 2026-10-27): rebuild only the probe with `.venv/bin/python scripts/load_zilliz.py --reset-probe` (dry run) and
   `--confirm LOAD-ZILLIZ --reset-probe` (uses the full-access key; touches only `ott_write_probe`), then re-run `verify_zilliz.py --recall`.
   `--recall` now runs the level study: Zilliz search `level` 1, 3, 5, 7 and 10 on the same 100 users (recall vs the exact lists, laptop-measured search time that is
   NOT Lambda latency, and the handler's top-18 live vs precomputed). About 1,000 REST searches; expect several minutes. `--level N` runs A1/A2 at level N.
   **Since the 2026-10-27 key decision (option a, the key is write-capable):** run `.venv/bin/python scripts/verify_zilliz.py --skip-write-test --recall`. It prints a
   progress line every 20 users with a rough time left, and each level's result as soon as that level is done; the verdict covers A1 and A2 only.
   Pausing with Ctrl+Z does not stop it (use Ctrl+C, or `jobs -l` then `kill %N`).
3. **Zilliz settings for the function** (reads `ZILLIZ_URI` and `ZILLIZ_TOKEN_READONLY` from `.env` inside Python, prints only `stored: yes/no`; refuses a blank key or the full-access key):
```bash
.venv/bin/python scripts/put_ssm_secrets.py --region eu-central-1
```
   (Alternative with hidden prompts instead of `.env`: `AWS_PROFILE=ott-dev .venv/bin/python scripts/cloudshell/put_zilliz_param.py --region eu-central-1`.)
4. **Lambda** (log group with 7-day retention, function with no environment variables, IAM-only Function URL, optional reserved concurrency 2), then a direct-invoke smoke test (health, precomputed M2, precomputed M3b, and LIVE mode, which proves the function can read the SSM parameter and search Zilliz; a fallback to the precomputed list counts as a failure):
```bash
AWS_PROFILE=ott-dev .venv/bin/python scripts/cloudshell/deploy_lambda.py --region eu-central-1 --zip build/lambda.zip create
```
```bash
AWS_PROFILE=ott-dev .venv/bin/python scripts/cloudshell/deploy_lambda.py --region eu-central-1 smoke --user-id 1
```
   A new zip later: same script, action `update-code`. `show-url` prints the Function URL (keep it out of the repository and chat). The zip is 49.4 MiB against a 50 MiB direct-upload limit; if
   the create call is refused for size, tell me (fallbacks: trim the bundle by about 3 MiB, or an S3 upload, which needs a bucket and a policy change that is your decision).
5. **Demo and latency** (checkpoint 5; scripts not written yet): the laptop signs requests to the IAM-only Function URL with the `ott-dev` profile (same account, and `ott-dev-serving` has
   `lambda:InvokeFunctionUrl` and `lambda:InvokeFunction`; untested until the function exists). Latency predictions are committed before any measurement. Never `AuthType NONE`.
6. **Teardown** (after the demo, or any time; dry run first by leaving off `--confirm`):
```bash
.venv/bin/python scripts/aws_teardown.py --confirm DELETE-OTT
```

---

# Part B: the CloudShell route (FALLBACK ONLY)

Use only if the laptop route is blocked again (for example a new policy appears on the account). The same scripts run inside AWS CloudShell with the console identity's temporary credentials
via `--session-credentials` (default credential chain, no profile; the "EC2 must be denied" check is skipped there).

## What CloudShell gives you (AWS documentation, read 2026-10-26)
- Amazon Linux 2023 shell, **1 vCPU, 2 GiB RAM**, Python 3 and **boto3 pre-installed** (check the versions with the first command below), `zip`/`unzip`, `jq`.
- **1 GB persistent storage per Region** in the home folder; uploads in one CloudShell Region are not visible in another; unused data is deleted after 120 days.
- Sessions end after 20-30 minutes without keyboard/pointer input (a running program does not count) and after about 12 hours in any case.
- The console menu uploads/downloads ONE file at a time (Actions > Upload file / Download file).
- **Not confirmed from the docs:** a maximum size for a single upload; which Regions CloudShell runs in. Our largest upload is 49.4 MiB.

## Files to upload (each script is ONE file; tests enforce it. No project code and no `.env` goes to CloudShell)
| Step | Upload | Size |
|---|---|---|
| Permission test, teardown | `scripts/aws_permtest.py`, `scripts/aws_teardown.py` | 13.7 KB, 7.4 KB |
| Tables + data load | `scripts/cloudshell/load_dynamodb.py`, `build/dynamodb/manifest.json`, `build/dynamodb/users_items_part1.jsonl.gz` ... `part4` | 5.5 KB, 0.8 KB, 4 x about 15.6 MiB (62.3 MiB) |
| Zilliz settings | `scripts/cloudshell/put_zilliz_param.py` | 2.3 KB |
| Lambda | `scripts/cloudshell/deploy_lambda.py`, `build/lambda.zip` | 7.2 KB, 49.4 MiB |

If the zip upload is refused: on the laptop `split -b 20M build/lambda.zip lambda.zip.part-`, upload the pieces, then in CloudShell `cat lambda.zip.part-* > lambda.zip`.

## Commands inside CloudShell (one at a time; uploaded files are in the home folder)
```bash
python3 --version && python3 -c "import boto3; print('boto3', boto3.__version__)"
```
```bash
python3 aws_teardown.py --session-credentials
```
```bash
python3 aws_permtest.py --region eu-central-1 --session-credentials
```
```bash
python3 aws_permtest.py --region ap-south-1 --session-credentials
```
Later steps use the same commands as Part A with `python3`, no `AWS_PROFILE`, and no `--dir`/`--zip` paths: `python3 load_dynamodb.py --region eu-central-1 --create-tables | --load | --check`,
`python3 put_zilliz_param.py --region eu-central-1` (hidden prompts), `python3 deploy_lambda.py --region eu-central-1 create | smoke --user-id 1 | update-code | show-url`, and
`python3 aws_teardown.py --session-credentials --confirm DELETE-OTT`. Delete the uploaded files afterwards (`rm -f users_items_part*.jsonl.gz lambda.zip*`).

## What a console role would need that `ott-dev` does not (only relevant to this fallback)
`iam:PassRole` on `ott-recsys-lambda-role` (draft: `infra/iam/console-role-passrole.json`, not attached, only needed if the identity running the scripts is not `ott-dev`), plus the same
service actions as `ott-dev-serving`. If a demo cannot sign requests from the laptop, run the demo inside CloudShell and download the HTML report.

---

# History: the previous account (closed)

Kept as evidence for the write-up; none of it applies to the current account.
- The previous account was created through the Builder sign-up and sat under an organisation service control policy (SCP) owned by a management account that was not yours.
- Checkpoint 2 there (profile `ott-dev`, both Regions): `logs create_log_group`, `ssm put_parameter`, `dynamodb create_table` and `lambda create_function` were all DENIED by the SCP; `sts` worked;
  the deliberate EC2 call was denied by our own policy; nothing was created. An SCP overrides every IAM policy, so no change on our side could lift it.
- You then reported that a console role session in CloudShell could create DynamoDB tables while `ott-dev` could not (not verified by me); that led to Part B. The account was closed and replaced.
- Lessons kept: the teardown script treats a failed check as `unverified`, never as "absent"; the permission test names an SCP denial in words without printing the error text.
