# Appendix E. Deployment Runbook (Summary)

The full runbook is `reports/module9_runbook.md` in the repository. Every command below was run by me from the laptop, in this order, with the IAM user's profile. Each step prints only words and counts.

**Table E.1** *Deployment steps*

| Step | What happens | Script |
|---|---|---|
| 1. Account set-up (by hand) | Root MFA, budget alert, Lambda role, managed policy, IAM user and its key, local profile | (console; checklist in `reports/module9_account_setup.md`) |
| 2. Permission test | Creates, uses and deletes tiny test resources in Frankfurt and Mumbai; reports ALLOWED or DENIED per action | `aws_permtest.py` |
| 3. Vector database | Checks the exports, creates the three Zilliz collections and loads 49,843 movie vectors | `load_zilliz.py` |
| 4. Verification | Searches with the function's key through two client libraries; optional level study | `verify_zilliz.py` |
| 5. DynamoDB | Creates two on-demand tables and loads 20,000 user records; reads 40 back | `cloudshell/load_dynamodb.py` |
| 6. Zilliz key | Stores the key in one encrypted parameter | `put_ssm_secrets.py` |
| 7. Lambda | Creates the log group (7-day retention), uploads the 49.4 MiB package, creates the signed-only function URL | `cloudshell/deploy_lambda.py` |
| 8. Smoke test | Health check, precomputed and synthetic-device requests, and a live request | `cloudshell/deploy_lambda.py` |
| 9. Measurements | Latency study, then the demo page | `latency_study.py`, `demo.py` |
| 10. Teardown | Removes every AWS resource and Zilliz collection, then confirms | `aws_teardown.py`, `zilliz_teardown.py` |

A fallback route through AWS CloudShell exists in the runbook for accounts where the laptop route is blocked by an organization's policy (Section 7.4); it was not needed for the final deployment.
