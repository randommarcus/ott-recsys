# Module 9: IAM drafts (NOT attached to anything)

Files: `infra/iam/ott-dev-policy.json`, `infra/iam/lambda-exec-role-policy.json`, `infra/iam/lambda-trust-policy.json`.
Everything uses the placeholder `ACCOUNT_ID`. Replace it yourself when you paste into the console; never put the real account id in the repository.

## What each file is

| File | Who | What it allows |
|---|---|---|
| `lambda-exec-role-policy.json` | the role the Lambda function runs as (`ott-recsys-lambda-role`, **you create it in the console**) | write its own log stream; read `ott-users`; read and count in `ott-bandit`; read the ONE parameter `/ott/zilliz` |
| `lambda-trust-policy.json` | same role | only the Lambda service may assume it |
| `ott-dev-policy.json` | the `ott-dev` IAM user (the credentials the scripts use), attached as the **customer managed policy `ott-dev-serving`** | create, use and delete `ott-*` DynamoDB tables, Lambda functions, log groups and `/ott/*` SSM parameters in the two regions; list names (four read-only calls, for the teardown check); **pass the one role to Lambda** |

## What ott-dev can NOT do (by design)

- No `iam:Create*`, `iam:Attach*`, `iam:Put*` or any other role-editing action: it cannot create roles or widen any permission. These are also in an explicit Deny, so a later
  careless policy attachment cannot re-enable them. Its only IAM action is `iam:PassRole` on exactly `role/ott-recsys-lambda-role`, and only to `lambda.amazonaws.com`.
- No EC2, no provisioned concurrency, no DynamoDB global tables or backups, nothing outside `eu-central-1` and `ap-south-1`.
- No list actions except one statement, `ListNamesForTeardownCheck` (`dynamodb:ListTables`, `lambda:ListFunctions`, `logs:DescribeLogGroups`, `ssm:DescribeParameters`), which needs `"Resource": "*"` because those calls cannot be scoped to a name. They only return names, so the teardown check can spot leftovers; every write action stays scoped to `ott-*`. The teardown script itself still deletes by exact name and treats "not found" as "already gone".

## Console steps for you (CP1, manual; redone in the NEW standalone account)

The full ordered checklist (root MFA, budget, what NOT to create) is `reports/module9_account_setup.md`; these are its IAM steps. The previous account (under an organisation SCP) was closed.
The only expected difference now is that the permission tests should pass.

1. IAM > Roles > Create role > Custom trust policy: paste `lambda-trust-policy.json`. Skip managed policies. Name it exactly `ott-recsys-lambda-role`.
2. Role > Add permissions > Create inline policy > JSON: paste `lambda-exec-role-policy.json` with ACCOUNT_ID replaced. Name it `ott-recsys-lambda-inline`.
3. IAM > Policies > Create policy > JSON: paste `ott-dev-policy.json` (ACCOUNT_ID replaced) as a **customer managed policy named `ott-dev-serving`** (not an inline policy: see "Why a customer managed policy" below). Then IAM > Users > `ott-dev` (no console access) > attach `ott-dev-serving` > create an access key.
4. In your own terminal: `aws configure --profile ott-dev`. This project's settings deny me that command and any read of the AWS credential folder.

`ott-dev` gets NO other policy: not `ReadOnlyAccess` (the old account used it temporarily; the teardown check works without it, thanks to the names-only list statement), not any admin policy.
`infra/iam/console-role-passrole.json` is a fallback-only draft (CloudShell route) and is not needed for the laptop route.

## Why a customer managed policy, not an inline policy

IAM limits the total size of the inline policies on one user, but allows a larger customer managed policy. From the AWS documentation
([IAM and AWS STS quotas, "IAM and STS character limits"](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_iam-quotas.html), read 2026-10-26):

| Policy type | Size limit (white space is NOT counted) |
|---|---|
| Inline policies on a user (total of all of them) | 2,048 characters |
| Inline policies on a role (total) | 10,240 characters |
| Inline policies on a group (total) | 5,120 characters |
| One customer managed policy | 6,144 characters |

`ott-dev-policy.json` has about 2,730 non-whitespace characters with the `ACCOUNT_ID` placeholder (slightly more with a real 12-digit id), so it does not fit
in a user's inline policy but fits in one managed policy with room to spare. The Lambda execution role policy (about 1,000 characters) fits in the role limit, so it
stays an inline policy on the role. A managed policy also lets you attach, detach and version it without re-pasting JSON.

## Changes made to `ott-dev-policy.json` when it was attached (CP1)

- `lambda:UpdateFunctionUrlConfig` added to `LambdaOttFunctions`.
- New statement `ListNamesForTeardownCheck` (list/describe names only, `Resource: "*"`).
- `LogsOttGroups`: the two `ott-*:*` log-group ARNs removed (redundant: `ott-*` already matches them).

## Things I could not confirm from official docs (check in the permission test)

- Decrypting a SecureString that uses the AWS-managed key should need no `kms:Decrypt` line in the role. If the first call fails with an access error naming KMS, we add one line scoped to that key.
- Newer Function URL rules may need BOTH `lambda:InvokeFunctionUrl` and `lambda:InvokeFunction` for the caller. `ott-dev` has both.
- Creating a function with a log group that already exists: the role does not get `logs:CreateLogGroup` on purpose, so `ott-dev` creates the group (with 7-day retention) before the function.
