# Module 9: AWS account setup checklist (new standalone account)

Do these by hand in the AWS console and your own terminal, in this order. I do none of it and never see keys. Never put an account id, access key, secret or URL in any file or chat.
Policy files: `infra/iam/*.json` (the placeholder `ACCOUNT_ID` is replaced only inside the console box). Details and size limits: `reports/module9_iam.md`.

## Create, in this order
1. **Root user safety.** Strong unique password; turn on **MFA for the root user**; do not create root access keys. After step 4 stop using root for everyday work.
2. **Billing guard-rails.**
   - Billing > confirm the account plan (Free or Paid) and the credit balance (USD $100 at sign-up). Free plan: the account closes when credits run out or after six months, so plan to finish and tear down well before.
   - Budgets > create a **cost budget of $5** per month with e-mail alerts at 80% and 100% of actual cost and at 100% forecast. (Whether budgets are available on the Free plan is not confirmed.)
3. **Role `ott-recsys-lambda-role`** (IAM > Roles > Create role > Custom trust policy: paste `lambda-trust-policy.json`; attach no managed policy). Then add an inline policy named
   `ott-recsys-lambda-inline` from `lambda-exec-role-policy.json`. Name must be exactly this.
4. **Customer managed policy `ott-dev-serving`** (IAM > Policies > Create policy > JSON: paste `ott-dev-policy.json`). It must be a managed policy, not inline (about 2,730 non-whitespace characters;
   inline limit for a user is 2,048, managed 6,144).
5. **User `ott-dev`:** no console access, no group; attach ONLY `ott-dev-serving`; create ONE access key (use case: command line) and keep it only in the AWS credential store.
6. **Laptop profile** (your own terminal; this project denies me this command and any read of the credential folder): `aws configure --profile ott-dev`, default region `eu-central-1`, output `json`.
   Use `--profile ott-dev` every time; do not run a bare `aws configure` (it would create a default profile that other tools could pick up).
7. **Check:** run `.venv/bin/python scripts/aws_teardown.py` from the repo. All lines `absent` and `existing: 0` means the profile and read permissions work. Then continue with `reports/module9_runbook.md`.

## Do NOT create
- Root access keys; a second access key for `ott-dev`; more IAM users, groups or roles; access keys for roles.
- Any extra permission on `ott-dev`: no `AdministratorAccess`, no `PowerUserAccess`, no `ReadOnlyAccess` (the old account needed it; this one does not), no inline policy on the user.
- Anything outside `eu-central-1` and `ap-south-1`; anything not named `ott-*` (or `/ott/*` for SSM).
- EC2, NAT gateway, VPC resources, provisioned concurrency, API Gateway, an S3 bucket (unless you decide later), CloudTrail trails, DynamoDB backups or global tables.
- A Function URL with `AuthType NONE` (public).
- Joining AWS Organizations or Control Tower: on the Free plan that upgrades the account to Paid automatically.
- Any file in the repository containing an account id, key, secret, or URL.
