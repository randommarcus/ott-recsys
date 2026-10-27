# Hyper-Personalized OTT Recommendation Engine

An MBA (Business Analytics) Semester 4 project. It builds a full movie recommendation pipeline for a streaming (OTT) service, tests it against simple baselines on held-out future data, and serves it from AWS Lambda with a hosted vector database.

The full write-up is the final report: [`reports/final/OTT_Recommender_Report.docx`](reports/final/OTT_Recommender_Report.docx) (Markdown sources in [`reports/final/`](reports/final/)).

## What It Does

A request for one user goes through four stages:

1. **Candidate generation.** A two-tower model (TensorFlow/Keras, 64-dimensional) picks 200 titles from about 50,000. A content lane adds up to 30 more from Sentence-BERT embeddings of TMDB descriptions. The vectors live in Milvus (Zilliz Cloud).
2. **Ranking.** An XGBoost ranker orders the 230 candidates using 37 features, including a Neural Collaborative Filtering (NCF, TensorFlow) score, the hour, the day and the session length.
3. **New releases.** A UCB multi-armed bandit (c = 0.2) fills reserved slots with new titles and learns from feedback.
4. **Serving.** An AWS Lambda function (Python 3.12, numpy only) reads the user's record from DynamoDB, scores and returns the list. TensorFlow runs offline only; the function gets precomputed embeddings and a numpy copy of the models.

## Headline Results

On the test period, opened once after the predictions were written down (`reports/module8_predictions.md`). Brackets are 95% bootstrap intervals over users.

| Claim | Result |
|---|---|
| Ranker (with context) vs two-tower order, NDCG@10 | +0.0245 [+0.0229, +0.0261] |
| Ranker (no context) vs two-tower order, NDCG@10 | +0.0232 [+0.0217, +0.0247] |
| Gain from context features, NDCG@10 | +0.0013 [+0.0004, +0.0021] |
| Two-tower vs item-kNN, Recall@200 | +0.0473 [+0.0441, +0.0505] |
| UCB vs UCB1, likes per 1,000 new-release impressions (SIMULATION, NOT AN A/B TEST) | +237.1 [+232.2, +242.0] |
| UCB vs Thompson sampling, same measure (SIMULATION, NOT AN A/B TEST) | +31.9 [+27.9, +35.9] |

Serving, measured on the deployed function in Frankfurt (eu-central-1): warm median 91.1 ms with precomputed candidates and 121.9 ms with live vector search. Full detail in `reports/module9_latency_results.md`.

The full test output is `reports/module8_test_results.txt`.

## Read This Before Using the Numbers

- **Device is simulated.** MovieLens has no device data. The device-aware model (M3b) was trained on synthetic labels to show the plumbing works. It is not evidence that device matters.
- **The bandit results are a simulation** replayed on historical ratings, not an A/B test with real users.
- **Ratings stand in for viewing.** A rating of 4 or more counts as a "like". MovieLens users often rate many films in one sitting, so session features are weak signals.
- **A subset was used:** 20,000 users from MovieLens 25M (seed 42), because of a 16 GB, CPU-only laptop.
- **Hour and day are UTC**, not local viewing time.

## Repository Layout

| Path | Contents |
|---|---|
| `src/ott_recsys/` | The library: data and splits (`config.py`, `data.py`), baselines, evaluation, two-tower, content, NCF, ranker, bandit |
| `src/ott_recsys/serving/` | The Lambda function: handler, DynamoDB store, Zilliz search client, numpy tree and NCF evaluators, UCB |
| `scripts/` | One script per step (see below); deployment, teardown and report builders |
| `scripts/cloudshell/` | DynamoDB load and Lambda deploy scripts (also usable from AWS CloudShell) |
| `tests/` | Pytest suite, all offline (cloud calls are faked) |
| `infra/iam/` | IAM policies for the deploy user and the Lambda role, with the account number replaced by a placeholder |
| `reports/` | One report per module, the pre-registered predictions, the test results, latency study and demo page |
| `reports/final/` | The final report: Markdown chapters, figures and the Word document |
| `PROGRESS.md` | Dated log of every module, decision and result |
| `data/`, `models/` | Created by the scripts; not committed |

Dataset-specific settings sit in one place, `src/ott_recsys/config.py` and `src/ott_recsys/data.py`, so the code can be pointed at another dataset.

## Setup

Needs Python 3.12 (TensorFlow does not support newer versions yet). Tested on WSL2 Ubuntu, CPU only.

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install --python .venv/bin/python -r requirements.lock.txt
```

`requirements.txt` lists the direct dependencies; `requirements.lock.txt` pins the exact versions used.

Copy `.env.example` to `.env` and fill in only what you need. `TMDB_API_KEY` is needed to fetch descriptions; the Zilliz and AWS values only for the serving part. Never commit `.env`.

## Running the Pipeline

Data is not included. Download MovieLens 25M (about 250 MB) from GroupLens:

```bash
.venv/bin/python scripts/download_movielens.py 25m
```

Every step reads the dataset settings from environment variables. The project used:

```bash
export OTT_DATASET=ml-25m OTT_SAMPLE_USERS=20000 OTT_SAMPLE_SEED=42
```

Then the modules in order. Each one writes its report to `reports/` and is described in `PROGRESS.md`.

| Module | Script(s) | What it does |
|---|---|---|
| 1 | `run_module1.py`, `build_user_split.py` | Audit, sessions, time-based splits |
| 2 | `run_module2.py` | Baselines on validation (popularity, item-kNN and others) |
| 3 | `fetch_tmdb.py`, `build_embeddings.py`, `run_module3.py` | TMDB descriptions, Sentence-BERT embeddings, content retrieval |
| 4 | `run_module4_all.py`, `run_module4b.py`, `build_candidates.py` | Two-tower model and the 230-slot candidate lists |
| 5 | `run_module5_all.py` | NCF |
| 6 | `run_module6_all.py` | XGBoost ranker with context features |
| 7 | `run_module7_all.py`, `run_module7b.py` | UCB bandit simulation |
| 8 | `prepare_test.py`, `unlock_test.py` | The one-time test evaluation |
| 9 | `build_serving_artifacts.py`, `refit_m3b.py`, `build_lambda_zip.py`, `export_dynamodb_items.py` | Serving bundle and package, all local |

The heavier modules (4 to 7) take hours on a laptop and run under a memory guard; the docstring at the top of each `*_all.py` script shows how to launch it in the background.

**The test set is locked.** `unlock_test.py --confirm UNLOCK-TEST-ONCE` refuses to run a second time, by design. Use `unlock_test.py --rehearsal` to check the plumbing on validation data.

## Tests

```bash
.venv/bin/python -m pytest -q
```

361 tests, about 3 minutes on a laptop. No test calls AWS, Zilliz or TMDB.

## Deploying (Costs Money)

The serving stack was deployed, measured and then fully removed. To deploy it again, follow `reports/module9_runbook.md` (account checklist in `reports/module9_account_setup.md`). In short: permission test, load Zilliz, load DynamoDB, store the Zilliz key in SSM, deploy Lambda, smoke test, then tear down with `scripts/aws_teardown.py` and `scripts/zilliz_teardown.py`. The teardown scripts default to a dry run.

The function URL accepts only signed AWS requests. No account number, key or endpoint is stored in this repository.

## Rebuilding the Report

```bash
.venv/bin/python scripts/make_report_figures.py
```

```bash
.venv/bin/python scripts/build_report_docx.py
```

Figures are drawn only from saved results in `reports/`. `scripts/check_report_style.py` checks the text.

## Data Credits

- **MovieLens 25M** by GroupLens Research, University of Minnesota. F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5, 4: 19:1-19:19. https://doi.org/10.1145/2827872. The data is not redistributed here; see the GroupLens usage licence.
- **TMDB.** This product uses the TMDB API but is not endorsed or certified by TMDB.

## AI Assistance

Most of the code, the tests and the first drafts of the reports were written with Claude Code (Anthropic) under my direction. I chose the design at each step, approved every cost, and ran every cloud command myself. Details are in Appendix H of the report.

## Author

Sudarshan, MBA Business Analytics, Semester 4.
