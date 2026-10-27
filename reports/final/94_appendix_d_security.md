# Appendix D. Security and Access Design

The policies are kept as JSON files in the repository (`infra/iam/`) with the account number replaced by a placeholder. Automated tests check them for the rules below.

## D.1 Who Can Do What

**Table D.1** *The identities and their permissions*

| Identity | Used for | Allowed | Explicitly blocked |
|---|---|---|---|
| IAM user `ott-dev` (managed policy `ott-dev-serving`) | Running every deployment, test and teardown script from the laptop | Create, use and delete DynamoDB tables, Lambda functions, log groups and parameters whose names start with "ott", in Frankfurt and Mumbai only; pass the one Lambda role to Lambda; list resource names for the teardown check | Creating, attaching or changing any IAM role or policy; EC2; provisioned concurrency; DynamoDB backups and global tables; any other region |
| Role `ott-recsys-lambda-role` | The Lambda function at run time | Write its own logs; read user records; read and update the bandit counters; read the one Zilliz parameter | Everything else |
| Zilliz key (stored in AWS Systems Manager) | Live search from the function | Search the three project collections (it could also write; see Section 7.3) | Nothing else in the project used it |

The `ott-dev` policy is about 2,730 characters long, above the 2,048-character limit for policies attached directly to a user, so it is a customer managed policy (limit 6,144 characters).

## D.2 How Secrets Were Handled

- No key, password, account number or private address appears in the code, the reports or the project history. Secrets live in a local `.env` file that is excluded from version control, and the coding assistant was blocked from reading that file and the AWS credential folder.
- The Zilliz key reaches the Lambda function only through one encrypted parameter in AWS Systems Manager, read at start-up and kept in memory. The function has no environment variables holding secrets.
- Every script prints only words and counts. Error messages show the type of error, never its text, because error text can contain addresses or keys. Automated tests feed fake "canary" secrets through every failure path and check that they never appear in any output.
- The function URL accepts only requests signed with AWS credentials; it is not open to the public.

## D.3 Safe Teardown

- Two scripts remove everything the project created: one for AWS (`aws_teardown.py`) and one for Zilliz (`zilliz_teardown.py`). Both default to a dry run that only reports, need a typed confirmation word to delete, touch only the project's own names, and re-check afterwards.
- A check that cannot confirm a resource is gone (for example, because of a permission error) is reported as "unverified", never as "absent".
- After the final measurements, the teardown removed all AWS resources in both regions and all three Zilliz collections, and a final dry run confirmed nothing was left.
