# Module 9: IAM drafts (NOT attached to anything)

Files: `infra/iam/ott-dev-policy.json`, `infra/iam/lambda-exec-role-policy.json`, `infra/iam/lambda-trust-policy.json`.
Everything uses the placeholder `ACCOUNT_ID`. Replace it yourself when you paste into the console; never put the real account id in the repository.

## What each file is

| File | Who | What it allows |
|---|---|---|
| `lambda-exec-role-policy.json` | the role the Lambda function runs as (`ott-recsys-lambda-role`, **you create it in the console**) | write its own log stream; read `ott-users`; read and count in `ott-bandit`; read the ONE parameter `/ott/zilliz` |
| `lambda-trust-policy.json` | same role | only the Lambda service may assume it |
| `ott-dev-policy.json` | the `ott-dev` IAM user (the credentials the scripts use) | create, use and delete `ott-*` DynamoDB tables, Lambda functions, log groups and `/ott/*` SSM parameters in the two regions; **pass the one role to Lambda** |

## What ott-dev can NOT do (by design)

- No `iam:Create*`, `iam:Attach*`, `iam:Put*` or any other role-editing action: it cannot create roles or widen any permission. These are also in an explicit Deny, so a later
  careless policy attachment cannot re-enable them. Its only IAM action is `iam:PassRole` on exactly `role/ott-recsys-lambda-role`, and only to `lambda.amazonaws.com`.
- No EC2, no provisioned concurrency, no DynamoDB global tables or backups, nothing outside `eu-central-1` and `ap-south-1`.
- No list actions (`ListTables`, `ListFunctions`). The teardown script uses exact names and treats "not found" as "already gone".

## Console steps for you (CP1, manual)

1. IAM > Roles > Create role > Custom trust policy: paste `lambda-trust-policy.json`. Skip managed policies. Name it exactly `ott-recsys-lambda-role`.
2. Role > Add permissions > Create inline policy > JSON: paste `lambda-exec-role-policy.json` with ACCOUNT_ID replaced. Name it `ott-recsys-lambda-inline`.
3. IAM > Users > Create user `ott-dev` (no console access) > attach `ott-dev-policy.json` as an inline policy (ACCOUNT_ID replaced) > create an access key.
4. In your own terminal: `aws configure --profile ott-dev`. This project's settings deny me that command and any read of the AWS credential folder.

## Things I could not confirm from official docs (check in the permission test)

- Decrypting a SecureString that uses the AWS-managed key should need no `kms:Decrypt` line in the role. If the first call fails with an access error naming KMS, we add one line scoped to that key.
- Newer Function URL rules may need BOTH `lambda:InvokeFunctionUrl` and `lambda:InvokeFunction` for the caller. `ott-dev` has both.
- Creating a function with a log group that already exists: the role does not get `logs:CreateLogGroup` on purpose, so `ott-dev` creates the group (with 7-day retention) before the function.
