# Reports: What Each File Is

Every number in the final report comes from one of these files. Each was written by a script and committed, so it can be checked later.

**How to read the names.** `ml-25m_u20000_s42` means MovieLens 25M, a sample of 20,000 users, random seed 42. `.txt` files are the readable output; `.json` files hold the same numbers for scripts.

**Two kinds of files.** *Predictions* files were written and committed **before** the experiment they describe was run, so the results can't be bent to fit them. *Results* files are the output of the run itself.

## Start Here

| File | What it is |
|---|---|
| `final/OTT_Recommender_Report.pdf` | The final project report (opens in the browser) |
| `final/OTT_Recommender_Report.docx` | The same report as an editable Word file (download to open) |
| `module8_test_results.txt` | The one-time test results: the main evidence |
| `module8_predictions_vs_actuals.md` | What was predicted before the test against what happened |
| `TEST_UNLOCK_LOG.json` | Proof that the test set was opened once (`"attempts": 1`) |
| `module9_demo.html` | The demo page from the deployed service (download and open in a browser) |

## By Module

| Module | File | What it is |
|---|---|---|
| 1. Data | `module1_audit_ml-25m_u20000_s42.txt` | Data audit: users, movies, sparsity, sessions, the time splits |
| 2. Baselines | `module2_validation_ml-25m_u20000_s42.txt` / `.json` | Simple methods (popularity, item-kNN and others) on validation data |
| | `module2_validation_ml-latest-small.txt` / `.json` | The same, on the small MovieLens set used while building the code |
| 3. Content | `module3_embeddings_ml-25m_u20000_s42.txt` | TMDB description coverage and the Sentence-BERT embeddings |
| | `module3_content_ml-25m_u20000_s42.txt` | Recommending from descriptions alone |
| 4. Two-tower | `module4_predictions.md` | Predictions, written before training |
| | `module4_two_tower_ml-25m_u20000_s42.txt` | Two-tower model results on validation |
| | `module4b_candidate_mix_ml-25m_u20000_s42.txt` | Mixing two-tower and content candidates into the 230-slot lists |
| 5. NCF | `module5_predictions.md` | Predictions, written before training |
| | `module5_ncf_ml-25m_u20000_s42.txt` | Neural Collaborative Filtering results on validation |
| 6. Ranker | `module6_predictions.md` | Predictions, written before training |
| | `module6_ranker_ml-25m_u20000_s42.txt` | XGBoost ranker with and without context features |
| 7. Bandit | `module7_predictions.md` | Predictions, written before the runs |
| | `module7_bandit_ml-25m_u20000_s42.txt` | UCB bandit for new releases (SIMULATION, NOT AN A/B TEST) |
| | `module7_ranker_robustness_ml-25m_u20000_s42.txt` | Robustness checks on the ranker |
| | `module7b_predictions.md`, `module7b_ucb_tuning_ml-25m_u20000_s42.txt` | Choosing the UCB setting c (SIMULATION) |
| 8. Test | `module8_protocol.md` | The rules for the one-time test, fixed in advance |
| | `module8_predictions.md` | Predictions, committed before the test was opened |
| | `module8_test_results.txt`, `TEST_UNLOCK_LOG.json` | The test results and the run log |
| | `module8_predictions_vs_actuals.md` | Predictions against results |
| | `rehearsal/` | A dry run on validation data to check the plumbing (its numbers mean nothing) |
| 9. Serving | `module9_plan.md` | The serving design |
| | `module9_iam.md`, `module9_account_setup.md` | AWS permissions and the account checklist |
| | `module9_cp3_plan.md` | Plan for loading the vector database |
| | `module9_runbook.md` | Step-by-step deployment and teardown |
| | `module9_latency_predictions.md` | Speed predictions, written before measuring |
| | `module9_latency_results.md` / `.json` | Measured speed of the deployed service |
| | `module9_demo.html` / `.json` | The demo page and its numbers |
| Report | `final_report_plan.md` | The plan for the final report |
| | `final/` | Report chapters (Markdown), figures, the Word file and `REFERENCE_CHECKS.md` |

The day-by-day story of decisions and results is in [`../PROGRESS.md`](../PROGRESS.md).
