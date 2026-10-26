# Module 9 plan: serving (PLANNING ONLY, nothing created)

Status: proposal. No AWS or Zilliz resource exists for this module. Every state change waits for an explicit go at its checkpoint (section 12).
Simulated parts are labelled: **device = SIMULATED**, **bandit feedback = SIMULATION, NOT AN A/B TEST**, **request clock = SIMULATED**,
**M3b = "Synthetic device effect, pipeline demonstration, not a finding"**.

Facts checked for this plan (official docs, 2026-10-26): Lambda zip 50 MB zipped (direct upload) / 250 MB unzipped incl. layers,
env vars 4 KB total, 6 MB sync payload, memory 128-10,240 MB (1 vCPU at 1,769 MB). Zilliz Free: 5 collections, 5 GB, 4 vector fields per
collection, nq <= 10, topK <= 1,024, insert 2 MB/s, auto-suspend after 7 days idle. Marked **[verify]** = not confirmed from an official page yet.

---

## 1. Architecture and per-request flow

Offline (laptop, TensorFlow allowed) -> artifacts -> online (Lambda, numpy only, no TF / xgboost / pandas / pymilvus).

```
demo client (laptop, SigV4-signed)  --HTTPS-->  Lambda Function URL (AuthType AWS_IAM)
                                                  |  init (once per container): load bundled item arrays, ranker trees, SSM secrets
                                                  |-- DynamoDB ott-users    GetItem(user_id)       user record
                                                  |-- [live mode] Zilliz REST search x2        200 main + 30 lane
                                                  |-- features (numpy) -> M2 ranker (numpy tree evaluator)
                                                  |-- DynamoDB ott-bandit   GetItem + UpdateItem ADD   UCB c=0.2 picks 2 lane slots
                                                  '-> JSON: 20 items, lane flags, banners, per-stage timings (ms)
```

Per request, in order:
1. Parse `{user_id, mode: precomputed|live, model: M2|M3b, device (M3b only), gap_minutes}`. Unknown user -> 404; user with no likes (39 users) -> popularity list.
2. Read the user record from DynamoDB (one GetItem).
3. Get candidates: precomputed mode reads the 230 slots from the record; live mode asks Zilliz (section 3).
4. Build the 38 ranker features: static ones (two-tower score/rank, fold-in NCF logit, content sim, genre affinity, popularity, lane flag) and request-time ones
   (item age, hour, day-of-week, the 6 session features). Same formulas as `rankfeat.build_features`, rewritten in plain numpy and parity-tested against it.
5. Score with M2 (or M3b), drop items not released by the clock, keep the top 18 main-lane items.
6. Bandit: UCB c = 0.2 over the user's eligible lane items using shared arm counts; fill 2 slots; atomically add 1 impression per shown arm.
7. Return the list and the timings. Optional `POST /feedback` sends a **SIMULATED** like (drawn from a hidden rate built from train+validation ratings) to update the arm counts.

Request clock (**SIMULATED**): MovieLens ends in 2019, so "now" = user's last history event + `gap_minutes` (default 10 = mid-session; 2,880 = new session).
Hour and day come from that clock, in UTC like training. History = train + validation (the Module 8 "real" artifacts). Serving never reads the test file (guarded by a test).

## 2. Region: eu-central-1 vs ap-south-1

| | eu-central-1 (Frankfurt) | ap-south-1 (Mumbai) |
|---|---|---|
| Next to Zilliz (aws-eu-central-1) | yes: live search stays in-region (expected a few ms) | no: every live search crosses regions (adds ~1 round trip, expected 100+ ms) [verify by measurement] |
| From your laptop | far: TCP connect 265 ms (1 sample today) | near: TCP connect 45 ms (1 sample today) |
| Precomputed mode | no Zilliz at request time | same; region choice only affects your network round trip |
| Data transfer | in-region | cross-region Zilliz traffic, a few KB per request: ~$0 |
| Service control policy | unknown: tested in section 8 | unknown: tested in section 8 |

Recommendation: **if both regions pass the permission tests, use eu-central-1** (compute next to the vector DB is the right design, and the server-side
latency we report is then clean). Your laptop round trip is measured separately and reported as a fixed offset. If eu-central-1 is blocked, use ap-south-1
and report the cross-region cost of live mode as a finding of the design. Moving the Zilliz free cluster to Mumbai is possible (0 collections today) but
only if Zilliz offers a free cluster there [verify]; I will not touch the cluster.

## 3. Two serving modes

| | A. Precomputed retrieval (primary) | B. Live retrieval |
|---|---|---|
| Candidates | 230 slots computed offline (`test_prep/real/lists.npz`) stored per user | Zilliz search at request time: two-tower collection (top 200 + seen count, max 1,024; seen items removed in Lambda) and content collection with filter `is_cold == true` (top 30) |
| User record | candidate IDs + static feature block (binary, ~8 KB) + session state | two-tower user vector (64), content profile (384, float16), fold-in NCF user embedding, seen ids, genre shares, session state |
| Zilliz needed online | no | yes (read-only key) |
| Strength | fastest, no external dependency | fresh retrieval, shows the vector DB working |
| Parity check | — | for 500 sampled users, B's candidate SETS must equal A's (tie-order differences allowed, as in the Module 8 rehearsal) and final top-20 must match |

Both modes rank with the same M2 code. Fallback if Zilliz fails online: mode B automatically falls back to mode A and says so in the response.

## 4. Lambda packaging

- Runtime Python 3.12, x86_64 (same CPU family as the laptop; arm64 is ~20% cheaper but adds a build step, not worth it here).
- Zip contents: handler + `ott_recsys/serving/` (pure numpy) + numpy wheel (~15 MB zipped) + artifacts:
  ranker trees M2 and M3b converted from xgboost JSON to compact numpy arrays (~1 MB each), two-tower item vectors (4 MB), NCF item side + MLP (~5 MB),
  item genres / release dates / popularity (<1 MB), centered content vectors as float16 (~25 MB), titles.
- **No xgboost in Lambda (checked, CP0).** `xgboost-cpu` 3.4.1 (x86_64 wheel) is small itself (5.8 MB zipped, 23 MB unzipped, no CUDA) but `import xgboost`
  requires scipy (tested: import fails without it). numpy + scipy + xgboost-cpu = 57.9 MB zipped / 192 MB unzipped BEFORE any of our data, which is over the 50 MB
  direct-upload limit and close to the 250 MB unzipped limit. Also unverified: whether the Lambda image provides libgomp. Decision: numpy tree evaluator
  (missing-value default directions, base score), parity-tested against `xgboost.predict` on a large sample (all rows of a 200k-row random sample of the Module 6 features
  plus NaN-heavy rows, atol 1e-5).
- **No pymilvus in Lambda**: it needs grpcio + pandas (~80 MB). Live mode calls the Zilliz REST API v2 with the standard library (`urllib`) [verify endpoint at CP3].
- Target: zip < 50 MB so it uploads directly (no S3 bucket). If it is over, first fix = drop content vectors to the 64-d PCA only where exact parity allows; second = an S3
  bucket `ott-artifacts-*` (would need an extra policy line and a go).
- Build script `scripts/build_lambda_zip.py`: downloads the manylinux numpy wheel with `pip download --platform`, writes `build/lambda.zip` (gitignored), prints size and sha256.
- Float16 content vectors: parity test must show the M2 top-20 unchanged for >= 99% of 1,000 sampled users; otherwise keep float32 and use S3.

## 5. Function URL vs API Gateway

| | Function URL, AuthType AWS_IAM (proposed) | API Gateway HTTP API |
|---|---|---|
| Extra price | none (Lambda price only) | ~$1 per million requests after its free tier [verify] |
| Auth | SigV4 signed by your ott-dev credentials; anonymous calls rejected | IAM / JWT / Lambda authorizers |
| Throttling, routes, custom domain, WAF | none | yes |
| Resources to create | 1 config on the function | API, stage, integration, route, permission |

Choice: **Function URL with AWS_IAM**. Never AuthType NONE (a public URL lets anyone run up cost). Cost guard instead of throttling: reserved concurrency 2 on the
function if the account quota allows it (new accounts may have a total of 10, which blocks reserving) [verify in the permission test]. Fallback if the policy blocks
Function URLs: API Gateway HTTP API with IAM auth.

## 6. Resources and least-privilege IAM (NOT attached)

All names start `ott-`, all tagged `project=ott-recsys`. Region = chosen region R (permission tests use both).

| Resource | Name | Settings | Cost estimate |
|---|---|---|---|
| DynamoDB table | `ott-users` | on-demand, key `user_id` (N) | storage ~160 MB (free tier 25 GB); one-time load ~20k items x ~8 WRU ≈ 160k WRU ≈ $0.20-0.30 [verify price for R] |
| DynamoDB table | `ott-bandit` | on-demand, key `arm_id` (N); one item per new-release arm with counters `n` (impressions) and `s` (SIMULATED likes) | ~$0 |
| Lambda | `ott-recsys-serve` | 1,024 MB, 10 s timeout, no VPC, no provisioned concurrency | within always-free 1M requests / 400k GB-s per month |
| Function URL | on `ott-recsys-serve` | AuthType AWS_IAM | $0 |
| IAM role | `ott-recsys-lambda-role` | execution role, **created by you in the console** (policy in `reports/module9_iam.md`); ott-dev only gets PassRole on it | $0 |
| Log group | `/aws/lambda/ott-recsys-serve` | created by us BEFORE the function, retention 7 days | within free 5 GB |
| SSM parameter (one) | `/ott/zilliz` | SecureString, AWS-managed key, standard tier; value is one JSON object holding URI and read-only token | $0 |
| (Zilliz) collections | `movie_items_twotower_id`, `movie_items_content` | HNSW, IP | Free plan, $0 |

Not used: EC2, NAT gateway, VPC, provisioned concurrency, S3 (unless section 4 needs it), API Gateway (unless section 5 fallback).

**REVISED (CP0): the IAM policies now live in `reports/module9_iam.md` and supersede the JSON below.** Changes: you create the Lambda execution role yourself in the
console; `ott-dev` has NO `iam:Create*`, `iam:Attach*`, `iam:Put*` or any role-editing action, only `iam:PassRole` on the single role ARN
`arn:aws:iam::ACCOUNT_ID:role/ott-recsys-lambda-role` (passed to lambda.amazonaws.com only). The older JSON below is kept only to show what was replaced.

Superseded policy for the `ott-dev` IAM user (replace ACCOUNT_ID; for one region, drop the other from the lists):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "DynamoOttTables", "Effect": "Allow",
     "Action": ["dynamodb:CreateTable", "dynamodb:DeleteTable", "dynamodb:DescribeTable", "dynamodb:UpdateTable",
                "dynamodb:TagResource", "dynamodb:ListTagsOfResource",
                "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem", "dynamodb:BatchWriteItem", "dynamodb:Query"],
     "Resource": ["arn:aws:dynamodb:eu-central-1:ACCOUNT_ID:table/ott-*", "arn:aws:dynamodb:ap-south-1:ACCOUNT_ID:table/ott-*"]},
    {"Sid": "LambdaOttFunctions", "Effect": "Allow",
     "Action": ["lambda:CreateFunction", "lambda:DeleteFunction", "lambda:GetFunction", "lambda:GetFunctionConfiguration",
                "lambda:UpdateFunctionCode", "lambda:UpdateFunctionConfiguration", "lambda:InvokeFunction", "lambda:InvokeFunctionUrl",
                "lambda:CreateFunctionUrlConfig", "lambda:GetFunctionUrlConfig", "lambda:DeleteFunctionUrlConfig",
                "lambda:PutFunctionConcurrency", "lambda:DeleteFunctionConcurrency", "lambda:TagResource", "lambda:ListTags"],
     "Resource": ["arn:aws:lambda:eu-central-1:ACCOUNT_ID:function:ott-*", "arn:aws:lambda:ap-south-1:ACCOUNT_ID:function:ott-*"]},
    {"Sid": "LogsOttGroups", "Effect": "Allow",
     "Action": ["logs:CreateLogGroup", "logs:DeleteLogGroup", "logs:PutRetentionPolicy", "logs:TagResource",
                "logs:FilterLogEvents", "logs:GetLogEvents", "logs:DescribeLogStreams"],
     "Resource": ["arn:aws:logs:eu-central-1:ACCOUNT_ID:log-group:/aws/lambda/ott-*", "arn:aws:logs:eu-central-1:ACCOUNT_ID:log-group:/aws/lambda/ott-*:*",
                  "arn:aws:logs:ap-south-1:ACCOUNT_ID:log-group:/aws/lambda/ott-*", "arn:aws:logs:ap-south-1:ACCOUNT_ID:log-group:/aws/lambda/ott-*:*"]},
    {"Sid": "SsmOttParams", "Effect": "Allow",
     "Action": ["ssm:PutParameter", "ssm:GetParameter", "ssm:DeleteParameter", "ssm:AddTagsToResource"],
     "Resource": ["arn:aws:ssm:eu-central-1:ACCOUNT_ID:parameter/ott/*", "arn:aws:ssm:ap-south-1:ACCOUNT_ID:parameter/ott/*"]},
    {"Sid": "IamOttRolesOnly", "Effect": "Allow",
     "Action": ["iam:CreateRole", "iam:DeleteRole", "iam:GetRole", "iam:TagRole",
                "iam:PutRolePolicy", "iam:GetRolePolicy", "iam:DeleteRolePolicy", "iam:ListRolePolicies"],
     "Resource": "arn:aws:iam::ACCOUNT_ID:role/ott-*"},
    {"Sid": "PassOttRoleToLambdaOnly", "Effect": "Allow", "Action": "iam:PassRole",
     "Resource": "arn:aws:iam::ACCOUNT_ID:role/ott-*",
     "Condition": {"StringEquals": {"iam:PassedToService": "lambda.amazonaws.com"}}},
    {"Sid": "DenyOutsideTwoRegions", "Effect": "Deny", "NotAction": ["iam:*", "sts:*"],
     "Resource": "*", "Condition": {"StringNotEquals": {"aws:RequestedRegion": ["eu-central-1", "ap-south-1"]}}},
    {"Sid": "DenyCostlyThings", "Effect": "Deny",
     "Action": ["ec2:*", "lambda:PutProvisionedConcurrencyConfig", "dynamodb:CreateGlobalTable", "dynamodb:UpdateGlobalTable"],
     "Resource": "*"}
  ]
}
```

Lambda execution role `ott-recsys-lambda-role` (inline policy): `logs:CreateLogStream`, `logs:PutLogEvents` on its own log group only (no CreateLogGroup, so it
cannot make a never-expiring group); `dynamodb:GetItem` on `ott-users`, `GetItem`/`UpdateItem` on `ott-bandit`; `ssm:GetParameter` on `/ott/*`.
Trust: `lambda.amazonaws.com` only. Notes: ListTables / ListFunctions need `"Resource": "*"` and are deliberately absent (the teardown uses exact names).
Decrypting SecureString with the AWS-managed key should need no extra KMS line [verify in the permission test]. A test checks this JSON: no `*` action, every Allow
resource contains `ott`, deny statements present.

## 7. Secrets

- Read-only Zilliz key: you create it in the Zilliz console with the **Read-Only cluster role** (the docs list Admin / Read-Write / Read-Only as cluster roles;
  "Data Viewer" appears as a project-level role, so pick whichever the console offers that only allows search/query) [verify in console]. You paste it into `.env` as
  `ZILLIZ_TOKEN_READONLY=...` yourself. I never see it.
- Lambda config: **SSM Parameter Store SecureString (recommended)** vs environment variable:

| | SSM SecureString | Lambda environment variable |
|---|---|---|
| Visible in Lambda console / GetFunctionConfiguration | no | yes, to anyone who can view the function |
| Rotation | change the parameter, no redeploy | redeploy config |
| Cost / latency | free standard tier; one GetParameter at cold start (~tens of ms), cached per container | free, zero latency |

- REVISED: one parameter `/ott/zilliz` (JSON `{"uri": ..., "token": ...}`), read once per container at cold start and kept in a module variable.
- Upload path: `scripts/put_ssm_secrets.py` reads `.env` via `python-dotenv` inside Python, sends the value with boto3 (never on a command line, so it never shows in
  shell history or `ps`), prints only `stored: yes/no` and the error TYPE.
- Logging rules in the handler: secrets live only in module-level variables; never logged; exceptions logged as `type(exc).__name__` only; client error bodies are generic.
- `.env.example` gets `ZILLIZ_TOKEN_READONLY=` (blank). Nothing secret goes in CLAUDE.md, PROGRESS.md, reports or commits.
- Test `tests/test_serving_no_secrets.py`: (a) AST scan of `src/ott_recsys/serving/` and every new script: no print/log/f-string/exception message containing
  uri/token/secret/exc variables; (b) canary run: handler executed locally with fake URI `https://CANARY-URI-7f3a...` and token `CANARY-TOKEN-91c2...` through every error path
  (Zilliz unreachable, HTTP 401, malformed JSON, DynamoDB error, SSM error), assert neither canary appears in stdout, stderr, log records or the response body;
  (c) `.env.example` values for token/URI keys are empty. Mutation check: I temporarily add a bad print to prove the test fails, then remove it.

## 8. Throwaway permission tests (both regions)

`scripts/aws_permtest.py --region eu-central-1` and `--region ap-south-1`, profile `ott-dev`. Each step creates a tiny `ott-permtest-*` resource, uses it, and deletes it
(a `finally` block always attempts cleanup; a cleanup list is written to `logs/permtest_<region>.json` so a crash can be cleaned by the teardown script).
Output per step: `action  ALLOWED | DENIED (AccessDenied / explicit deny in SCP) | ERROR(type)`. No ARNs, account id or request ids printed.

Steps (REVISED: no role creation; the test function uses your pre-created `ott-recsys-lambda-role`, so CP1 now includes creating it): identity check (prints "ok" only)
-> logs create group + 7-day retention -> SSM put/get/delete SecureString `/ott/permtest` (dummy value; needs one extra ott-dev line, in `module9_iam.md`)
-> DynamoDB on-demand table create/put/get/delete -> Lambda create (5-line hello, 128 MB, with PassRole) -> invoke -> Function URL
AWS_IAM create -> signed call -> reserved concurrency 1 (expect it may fail on new-account quota) -> delete URL, function, log group, table, role.
Also checks one denied-by-us action (ec2 describe) to confirm denies work. Estimated cost: $0.00 (seconds of resources, a handful of requests). Runtime ~2 minutes per region.

## 9. Teardown

`scripts/aws_teardown.py` (dry-run by default): checks each exact name from section 6 plus `ott-permtest-*` and the permtest cleanup logs, in BOTH regions, prints what
exists. `--confirm DELETE-OTT` deletes in safe order: Function URL -> function -> log group -> DynamoDB tables -> SSM params -> role inline policy -> role; then re-checks and
prints "remaining: 0". Zilliz collections are dropped only with a separate `--zilliz --confirm DROP-ZILLIZ` (needs the read-write key).

Manual checklist (you): delete or deactivate the ott-dev access key; delete the Zilliz read-only key; check Billing > Bills next day and after 3 days (DynamoDB/CloudWatch
charges appear late); delete the budget alert if you want; remove the `ott-dev` profile from your AWS config yourself.

## 10. Demo: M2 vs synthetic M3b

- `scripts/demo.py` calls the deployed function and writes `reports/module9_demo.html` (plus a printed table).
- Panel 1 **M2 (real model)**: same user at gap 10 min vs 2 days, shows how session context changes the list. Banner: "Context = time + session. Device not used."
- Panel 2 **M3b**: same user and clock, `device=mobile` vs `device=tv`. Red banner on top of the panel, verbatim: **"Synthetic device effect, pipeline demonstration, not a finding"**.
  Plus a third column with shuffled device (the Module 6 control) to show the model only reacts because we planted the effect.
- M3b model: no final M3b exists (only out-of-fold scores). Offline refit once on the Module 6 VALIDATION rows (`ranker/features.npz`, the same rows the out-of-fold
  scores came from) with the Module 6 synthetic labels (SIMULATED device; mobile drops 60% of Drama-without-Comedy likes; same seeds, so the removed-likes count must equal
  the 11,782 in the Module 6 report), Module 6 settings, trees = median of its out-of-fold best iterations (67). Saved OUTSIDE the Module 8 frozen folder as
  `serving/final_M3b.json` with a sha256 so the spent test's manifest is untouched. The test portion is never read (static scan covers the script).
- Bandit panel banner: **"SIMULATION, NOT AN A/B TEST"**: 200 simulated requests + simulated feedback, shows the UCB counts moving.
- Every panel footer: "Device is SIMULATED. Request clock is SIMULATED. MovieLens ratings, not viewing."

## 11. Latency measurement

Predictions committed before any measurement (`reports/module9_latency_predictions.md`). Sequential requests only (no load test).
- Network baseline: 20 TCP+TLS connects from the laptop to each region's Lambda endpoint (no credentials).
- Client end-to-end: per mode (A, B), 200 warm requests over 200 sampled users, plus 20 cold starts (forced by changing a dummy env var, free).
- Server breakdown from the response's timing fields: init, DynamoDB read, Zilliz searches, features, ranker, bandit update; plus Lambda REPORT lines (duration, init duration, max memory).
- Report p50 / p95 / p99 with bootstrap intervals, per mode and region; server time vs network time separated.
- Optional (costs pennies): memory 512 / 1,024 / 1,769 MB for mode A; ask first.
- Initial guesses (to be refined in the predictions file): mode A warm server p50 < 30 ms; mode B co-located server p50 < 150 ms; cold start 1-3 s.

## 12. Order of work and checkpoints

| CP | What | State change | Cost (estimate) | Needs |
|---|---|---|---|---|
| 0 | Local only: `.env.example` placeholder; serving package (numpy features, tree evaluator, handler); parity tests vs `rankfeat`/xgboost; no-secrets test; policy test; M3b refit; local handler run against an in-memory fake DynamoDB; zip build + size. Commit each script before editing it. Show results. | none | $0 | go for this plan |
| 1 | **You, manually**: create the Lambda execution role `ott-recsys-lambda-role` in the console (trust lambda.amazonaws.com, inline policy from `module9_iam.md`); create IAM user `ott-dev`, attach its policy from `module9_iam.md` (your decision), create its access key and run `aws configure --profile ott-dev` in your own terminal (this project now denies me that command and any read of the AWS credential folder); create a $5 AWS budget alert; check whether the account is on the Free plan with credits [verify in console]. | yours | $0 | you |
| 2 | **First AWS write**: permission tests in eu-central-1 then ap-south-1; show the report; pick region. | throwaway, auto-deleted | ~$0.00 | go |
| 3 | **First Zilliz write**: create both collections, insert vectors (~50 MB, < 1 min at 2 MB/s), build index, load. You create the read-only key and add it to `.env`; a test shows it can search but an insert into a scratch collection is refused. | Zilliz free | $0 | go |
| 4 | Deploy in region R: SSM params, log group (7 days), role, DynamoDB tables + load, Lambda, Function URL; smoke test 5 users. | AWS | < $0.50 one-time | go |
| 5 | Commit latency predictions, then demo + latency runs; write results and plain-language explanation; update PROGRESS.md. | requests only | < $0.10 | go |
| 6 | Teardown (dry-run shown first), verify remaining 0, checklist. Recommended right after the demo is recorded; keep at most until the viva if you want it live. | deletes | $0 | go |

Expected total: under $1, most likely $0 within free tiers. To be confirmed in your Billing console after CP2 and CP4.

Fallbacks (CLAUDE.md): Lambda blocked -> FastAPI on EC2 (would need a new plan and a go, EC2 is denied in the policy above);
Zilliz fails -> mode A only, or Milvus on EC2 (same note).
