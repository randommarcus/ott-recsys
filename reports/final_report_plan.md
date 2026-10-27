# Final MBA report: proposed structure and plan (PLAN ONLY, waiting for your go)

Status: proposal, 2026-10-27. No guide-supplied structure exists, so this follows the usual shape of an MBA Business Analytics capstone: a business problem, an
analytical solution, rigorous evidence, and business recommendations. Target: **about 40-45 pages of main text** (A4, 12 pt, 1.5 spacing) plus appendices.

## Ground rules for the writing
1. **Claims hierarchy.** Headline findings = the pre-registered one-shot test results P1-P5 (Module 8) and the Module 9 serving measurements (latency vs pre-committed
   predictions, recall, cost). Everything else (validation-period tables, ablations) is supporting evidence and is labelled as such. Misses are reported as they are.
2. **Labels stay.** Device = SIMULATED; M3b = "Synthetic device effect, pipeline demonstration, not a finding"; bandit = "SIMULATION, NOT AN A/B TEST"; request clock = SIMULATED;
   MovieLens ratings are ratings, not viewing.
3. **No new experiments.** Every number and figure is produced from files already in `reports/` and `data/processed/` by one committed script; the test portion is never
   read again (the lock is spent; the script only reads the saved results).
4. **Authorship.** I draft; you review, rewrite in your own voice where you want, and own the final text. Please check your institution's rules on AI assistance; the report
   includes a short, honest "tools used" statement (draft text provided, your decision to keep or edit).
5. **References:** only well-known primary sources (listed below); every reference is marked for you to verify before submission. No invented citations.

## Structure (page budget in brackets)

**Front matter:** title page; declaration; acknowledgements; **executive summary (1.5 p)**: problem, approach, the five headline results, serving results, three recommendations;
contents; lists of figures and tables; abbreviations.

**1. Introduction (3 p).** The OTT recommendation problem (choice overload, engagement, new-release exposure); objectives mapped to the four required components
(candidate generation, ranking with context, bandit for new releases, serverless serving); scope and constraints (student budget, laptop + free tiers, TensorFlow required,
public data instead of viewing logs); research questions; roadmap.

**2. Background and literature (5 p).** Two-stage recommenders (retrieval then ranking); two-tower retrieval; sentence embeddings for content (SBERT); neural collaborative
filtering (NCF/NeuMF); learning-to-rank with gradient-boosted trees (LambdaMART, XGBoost); context-aware recommendation; multi-armed bandits (UCB1, Thompson sampling) for
cold start; vector databases and approximate nearest-neighbour search; serverless serving. Ends with the gap this project addresses: an end-to-end, honestly evaluated
pipeline on a student budget.

**3. Data (4 p).** MovieLens 25M (20,000-user sample, seed 42) + TMDB descriptions; what a "like" is; the per-user temporal split (70/10/20) and why random splits mislead;
the one-shot locked test; data audit numbers; what the data cannot say (ratings are not viewing; no device; UTC time).

**4. Methodology (10 p).**
- 4.1 Evaluation design: metrics (Recall@K, NDCG@10, coverage) in plain language; baselines (popularity, item-kNN, content); paired bootstrap intervals; pre-registration
  and predictions-before-results; the one-shot test protocol and its guards.
- 4.2 Candidate generation: ID-only two-tower (TensorFlow), SBERT content retrieval, the 200 + 30 candidate design with a new-release lane.
- 4.3 Ranking: NCF as a feature; XGBoost ranker; context features (time, session); the ablation ladder M0-M3; the synthetic device scenario (labelled).
- 4.4 New releases: UCB bandit, simulation design (SIMULATION, NOT AN A/B TEST), tuning on the first half and evaluating on the second.
- 4.5 Serving architecture: Lambda + DynamoDB + Zilliz; precomputed vs live mode; numpy-only runtime; security (least-privilege IAM, SSM, no secrets in code).

**5. Results (10 p).**
- 5.1 Headline test results P1-P5 with intervals, and predictions vs actuals.
- 5.2 Retrieval (two-tower vs kNN vs content; new-release coverage).
- 5.3 Ranking and the value of context (M2 - M0 small but real).
- 5.4 Bandit simulation (tuned UCB vs UCB1 vs Thompson; caveats).
- 5.5 Serving: deployment outcome; latency (warm 91 / 122 ms, cold about 2.3 s; 5 of 8 predictions inside); live-search recall (about 92% at level 5) and what users would
  see (about 93% top-18 agreement); cost of the deployment.

**6. Business discussion (5 p).** What the numbers mean for an OTT business: the ranker's lift as an engagement proxy; context gives little here, so spend effort on
candidates and ranking first; how to give new releases exposure (reserved slots + tuned UCB) and its cost to the main list; serving choices (precomputed as primary, live
retrieval when freshness matters; cost per million requests estimated from the measured durations, prices to verify); what to measure in a real A/B test (KPIs, guardrails);
build-vs-buy.

**7. Limitations, risks and ethics (3 p).** Proxy data; simulations; one test period with calendar leakage; known users only; approximate search; the write-capable
Zilliz key on the free tier; the account/SCP episode as a lesson in cloud governance; privacy, filter bubbles, popularity bias, fairness to new titles.

**8. Conclusion and future work (2 p).**

**References.** **Appendices:** A reproducibility (repository, scripts, seeds, test lock log, provenance check); B full result tables; C pre-registrations and
predictions vs actuals (Modules 4-9); D security and IAM design; E serving runbook summary; F demo page screenshots; G glossary for non-technical readers; H tools-used statement.

## Figures (about 10, made by one script from saved results, no re-runs)
1 system architecture diagram; 2 data timeline and split; 3 the funnel (catalog -> 230 candidates -> 20 shown); 4 headline P1-P5 with 95% intervals; 5 context ablation
ladder; 6 retrieval recall by method (warm/cold); 7 bandit: likes per 1,000 impressions by strategy; 8 serving latency by stage (precomputed vs live); 9 live-search recall by
search level; 10 demo screenshot (M3b panel with its banner).

## Production
- Source: Markdown chapters in `reports/final/` + figures in `reports/final/figures/`, committed chapter by chapter so you can review as we go.
- Output format, **your choice**: (a) **Word (.docx)**, which most guides accept and you can edit; needs one local install (`python-docx` or `pandoc` via pip into the
  project's environment: free, no cloud); or (b) PDF; or (c) Markdown only (you copy it into your own template).
- Order of work, one checkpoint each: (1) figures script + figures; (2) chapters 3-5 (evidence first); (3) chapters 1, 2, 6-8; (4) executive summary, appendices, references;
  (5) assemble the document; a final consistency check that every number in the text matches a saved result file.

## Decisions I need from you
1. This structure and page budget (or changes).
2. Output format: Word (recommended), PDF, or Markdown only, and whether I may install `python-docx` (or `pandoc`) locally.
3. Citation style (APA 7th is common for MBA work, or Harvard).
4. Whether to include the "tools used" statement, and in what wording.
