# Module 9: the AWS CloudShell route (runbook)

Why: your laptop IAM user `ott-dev` is blocked by an organisation service control policy (SCP) for creating log groups, SSM parameters, DynamoDB tables and
Lambda functions (checkpoint 2). You reported that the console role session, reached through CloudShell, can create DynamoDB tables. So the deployment
scripts run INSIDE CloudShell, with the console identity's temporary credentials. Nothing below has been run by me; I have not looked at AWS.

## What CloudShell gives you (AWS documentation, read 2026-10-26)
- Amazon Linux 2023 shell, **1 vCPU, 2 GiB RAM**, Python 3 and **boto3 pre-installed** (check the versions with the first commands below), `zip`/`unzip`, `jq`.
- **1 GB persistent storage per Region** in your home folder. Files uploaded in one CloudShell Region are not visible in another. Unused data is deleted after 120 days.
- Credentials: the identity you signed in to the console with (needs `cloudshell:PutCredentials` for the forwarding). The SCP still applies to that identity.
- Sessions end after 20-30 minutes without keyboard/pointer input (a running program does not count as input) and after about 12 hours in any case.
- The console menu uploads/downloads ONE file at a time (Actions > Upload file / Download file). Upload uses pre-signed S3 URLs, so it also needs `cloudshell:GetFileUploadUrls`.
- **Not confirmed from the docs:** a maximum size for a single upload (the docs only say large files "might be slow" and suggest S3 as an alternative); which Regions CloudShell runs in (the page points to a regional table I could not read). Our largest upload is 49.4 MiB.

## Files you upload, and what each needs
Every script is ONE file (standard library + boto3 only; tests enforce it). No project code and no `.env` goes to CloudShell.

| Step | Upload (from your laptop's `ott-recsys` folder) | Size |
|---|---|---|
| Permission test, teardown | `scripts/aws_permtest.py`, `scripts/aws_teardown.py` | 13.7 KB, 7.4 KB |
| Tables + data load | `scripts/cloudshell/load_dynamodb.py`, `build/dynamodb/manifest.json`, `build/dynamodb/users_items_part1.jsonl.gz` ... `part4` | 5.5 KB, 0.8 KB, 4 x 15.6-15.7 MiB (**62.3 MiB total**) |
| Zilliz settings | `scripts/cloudshell/put_zilliz_param.py` | 2.3 KB |
| Lambda | `scripts/cloudshell/deploy_lambda.py`, `build/lambda.zip` | 7.2 KB, **49.4 MiB** (51,797,851 bytes, sha256 starts f683872a) |

Total about 112 MiB, far below the 1 GB home folder (about 11%). The data is split into four parts of about 16 MiB so no single upload is large; if the 49.4 MiB zip
upload is refused, split it: on the laptop `split -b 20M build/lambda.zip lambda.zip.part-` (uploads 3 pieces), in CloudShell `cat lambda.zip.part-* > lambda.zip`.
If you deploy to BOTH regions you must open CloudShell in each Region and upload again (storage is per Region); I recommend ONE region, eu-central-1 (next to Zilliz).

Rebuild the files on the laptop at any time (no AWS involved):
```bash
.venv/bin/python scripts/build_lambda_zip.py --no-smoke
```
```bash
.venv/bin/python scripts/export_dynamodb_items.py
```

## Checkpoint 2 again, inside CloudShell (about 2 minutes per Region, $0.00)
Open CloudShell (any Region), upload the two scripts, then paste one command at a time:
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
```bash
python3 aws_teardown.py --session-credentials
```
The first teardown must show everything `absent`; the last must end with `existing: 0` and no `unverified`. Results are also saved in `./logs/permtest_<region>.json`
(download with Actions > Download file if you want me to see them; they contain only labels and words). `--session-credentials` = default credential chain, no named
profile; the "EC2 must be denied" line is skipped because that deny belongs to `ott-dev-serving`, not to the console role. If a step says `DENIED (service control policy)`
for the console role too, that Region is closed for that action: paste me the output.

## Later steps (after the permission test passes; each needs your go)
Region used below: eu-central-1.

1. **Tables and data** (upload the load script, `manifest.json` and the four parts). Creates `ott-users` and `ott-bandit` (on-demand, no provisioned capacity), checks each file's
   sha256 against the manifest, writes 20,000 records, then reads 40 back. About 84,600 write request units (about 1 WRU per KB per item); on-demand writes cost
   roughly a dollar-ish per million units (price not verified), so well under $0.50. Takes a few minutes; if the session dies, re-run with `--start-part N`.
```bash
python3 load_dynamodb.py --region eu-central-1 --create-tables
```
```bash
python3 load_dynamodb.py --region eu-central-1 --load
```
```bash
python3 load_dynamodb.py --region eu-central-1 --check
```
2. **Zilliz settings** (after the Zilliz Read-Only key exists; you paste into HIDDEN prompts, nothing is echoed, stored in a file or put in shell history):
```bash
python3 put_zilliz_param.py --region eu-central-1
```
3. **Lambda** (upload `deploy_lambda.py` and `lambda.zip`): creates the log group with 7-day retention, the function (Python 3.12, 1,024 MB, 10 s, role
   `ott-recsys-lambda-role`, no environment variables), a Function URL with `AuthType AWS_IAM` (never public), and tries reserved concurrency 2 as a cost guard.
   Then `smoke` calls the function directly (no URL needed) for the health check and one M2 and one M3b recommendation and prints counts and timings only.
```bash
python3 deploy_lambda.py --region eu-central-1 create
```
```bash
python3 deploy_lambda.py --region eu-central-1 smoke --user-id 1
```
   A new zip later: `python3 deploy_lambda.py --region eu-central-1 update-code`. `show-url` prints the Function URL (keep it out of the repository and chat).
   The smoke test covers the precomputed mode; live mode is tested after the Zilliz collections and the parameter exist (demo step).
4. **Tear everything down** (any time; also run after the demo):
```bash
python3 aws_teardown.py --session-credentials --confirm DELETE-OTT
```
   Then delete the uploaded files from CloudShell (`rm -f users_items_part*.jsonl.gz lambda.zip*`). Unaffected by this runbook: the role, the `ott-dev` user, the budget alert, Zilliz.

## What the console role needs that `ott-dev` did not
1. **The same actions as `ott-dev-serving`** on `ott-*` resources: DynamoDB (create/describe/delete table, item reads/writes), Lambda (create/update/get/delete function, Function URL
   config, invoke, concurrency), CloudWatch Logs (create log group, retention), SSM (`/ott/*`). The permission test shows exactly which are allowed; I do not know what policy your console role has.
2. **`iam:PassRole` on `ott-recsys-lambda-role`** (condition: only to `lambda.amazonaws.com`). `lambda:CreateFunction` fails without it. This is the most likely gap: broad
   "power user" style policies usually exclude IAM actions. If the permtest says `lambda create_function ... DENIED` while the SCP is not named, add the statement in
   `infra/iam/console-role-passrole.json` (replace ACCOUNT_ID) to that role in the console. Not attached by me.
3. `cloudshell:PutCredentials` / `GetFileUploadUrls` for CloudShell itself (you evidently have them).
4. Nothing else: the Lambda execution role `ott-recsys-lambda-role` already exists and is what the function runs as.

## Function URL and the demo from your laptop
Decision: keep `AuthType AWS_IAM` (a public URL would let anyone run up cost, and the plan forbids it). A signed URL needs credentials to CALL, so:
- **Option A (if `ott-dev` may invoke):** the SCP so far blocked create-type actions; `ott-dev-serving` already includes `lambda:InvokeFunctionUrl` and `lambda:InvokeFunction`. The demo script
  will sign requests with the `ott-dev` profile from the laptop. This is untested: it can only be checked once the function exists (the demo script will have a one-line `--check` mode).
- **Option B (default if A is denied):** run the demo and the latency measurements INSIDE CloudShell (it signs with the console role, and `aws lambda invoke` needs no URL), write the HTML
  report there, and download it with Actions > Download file. Server-side latency, the headline number, is unaffected; laptop-to-Lambda network time is then measured separately without
  credentials (TCP/TLS connect time to the Lambda endpoint, as in the plan).
- Rejected unless you decide otherwise: `AuthType NONE`, or a public URL guarded by a secret header (still publicly callable; costs money if found).
The demo and latency scripts are not written yet (checkpoint 5); they will support both options.
