# Appendix A. Reproducibility

Every number in this report comes from a committed script and a saved result file in the project repository ([repository link]). The project history holds more than 80 commits, and about 360 automated tests cover the code.

## A.1 Environment

- Python 3.12 in a virtual environment managed with uv; exact package versions in `requirements.lock.txt`.
- Main libraries: TensorFlow 2.20 with Keras 3, sentence-transformers, XGBoost, numpy, pandas, boto3, pymilvus.
- Hardware: a laptop with 16 GB of RAM (about 9 GB usable under WSL2) and no GPU. A memory guard script was set to stop any run above 7.5 GB.

## A.2 Data

- MovieLens 25M, downloaded by `scripts/download_movielens.py`; a fixed random sample of 20,000 users (seed 42) taken by the shared data module.
- TMDB descriptions and release dates fetched by `scripts/fetch_tmdb.py` with the project's own API key (stored outside the repository). TMDB data is not redistributed.

## A.3 Scripts by Module

**Table A.1** *Where each part of the work lives*

| Module | What it does | Main scripts |
|---|---|---|
| 1 and 2 | Data audit, time splits, test lock, evaluation code, baselines | `run_module1.py`, `build_user_split.py`, `run_module2.py` |
| 3 | TMDB fetch, Sentence-BERT embeddings, content retrieval | `fetch_tmdb.py`, `build_embeddings.py`, `run_module3.py` |
| 4 and 4b | Two-tower training, candidate mixes, exports for the vector database | `train_two_tower.py`, `run_module4_all.py`, `run_module4b.py` |
| 5 | NCF training and the 230-slot candidate lists | `build_candidates.py`, `train_ncf.py`, `run_module5_all.py` |
| 6 | Ranker inputs, XGBoost ranker, context ladder | `build_ranker_features.py`, `run_module6_all.py` |
| 7 and 7b | Bandit simulation and tuning | `run_module7_all.py`, `run_module7b.py` |
| 8 | Frozen artifacts and the one-shot test run | `prepare_test.py`, `unlock_test.py` |
| 9 | Serving: bundle, Lambda zip, deployment, Zilliz, latency, demo, teardown | `build_serving_artifacts.py`, `build_lambda_zip.py`, `load_zilliz.py`, `verify_zilliz.py`, `latency_study.py`, `demo.py`, `aws_teardown.py`, `zilliz_teardown.py` |
| Report | Figures and this document | `make_report_figures.py`, `build_report_docx.py`, `check_report_style.py` |

## A.4 The One-Shot Test Run

- The test portion was locked in code during development: reading it raised an error, and automated scans checked that no tuning code referred to it.
- The protocol (`reports/module8_protocol.md`) and predictions (`reports/module8_predictions.md`) were committed in commit e3d5da3 before the test was opened.
- Before the run, a provenance check re-ran the committed preparation script from a clean copy of the repository: all 8 compared artifacts came out byte-identical to the frozen ones.
- The test-opening script ran once, on 26 October 2026 from 16:08 to 16:17, on code commit cc20239, with 12 frozen artifacts checked against their recorded fingerprints. Its log (`reports/TEST_UNLOCK_LOG.json`) records status "done" and one attempt. The script refuses to run again because its own log exists in the project history.
- The results file (`reports/module8_test_results.txt`) was committed unedited; its SHA-256 fingerprint begins dd475cea.

## A.5 Serving Measurements

- Latency predictions were committed in commit 519e241 before the latency study ran; the study script refuses to start if that file has changed.
- Raw latency results: `reports/module9_latency_results.json`. Demo page: `reports/module9_demo.html`.
- The level study of the vector database (Figure 5.6) was run once from the laptop; its numbers are recorded in the progress log.
