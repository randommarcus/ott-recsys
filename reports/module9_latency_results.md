# Module 9, checkpoint 5: latency results vs predictions, and the demo

Run by the project owner on 2026-10-27 from the laptop (India) against the Lambda in eu-central-1 (Python 3.12, x86_64, 1,024 MB), DynamoDB on-demand and the Zilliz
free cluster (level 5), all in Frankfurt. Raw numbers: `module9_latency_results.json`; predictions committed beforehand: `module9_latency_predictions.md`
(sha256 02c88aa456f6..., commit 519e241). Everything was torn down afterwards (both teardown scripts: `remaining: 0`; dry run: `existing: 0` in both Regions).

## Warm requests (200 users per mode, sequential, through the IAM-signed Function URL)

| | precomputed | live |
|---|---|---|
| server time p50 [95% bootstrap interval] | **91.1 ms** [85.9, 96.4] | **121.9 ms** [121.2, 123.6] |
| server time p95 / p99 | 118.5 / 121.7 ms | 144.2 / 163.9 ms |
| stage medians: DynamoDB user read / Zilliz search / features / ranker / bandit | 3.2 / 0.0 / 1.8 / **77.8** / 0.1 ms | 3.2 / **34.4** / 1.6 / **76.1** / 3.3 ms |
| laptop end-to-end p50 / p95 / p99 (includes India-Frankfurt) | 338 / 361 / 382 ms | 367 / 400 / 437 ms |
| errors / live fallbacks | 0 / - | 0 / 0 |

Cold starts (10, forced by a configuration change; 5 per mode): Lambda **Init Duration median 572 ms** (max 625); the **first request then took a median 1,710 ms**
(precomputed 1,387-1,759 ms, live 1,689-2,071 ms); max memory 204 MB (precomputed) and 220 MB (live) of 1,024. Laptop TCP+TLS connect: median 492 ms.

## Measured vs predicted (5 inside, 3 misses; reported as they are, no re-run)

| id | quantity | predicted | measured | verdict |
|---|---|---|---|---|
| L1 | warm server, precomputed, median | 8-30 ms | 91.1 ms | **MISS** |
| L2 | warm server, live, median | 25-150 ms | 121.9 ms | inside |
| L3 | live minus precomputed, medians | 15-130 ms | 30.7 ms | inside |
| L4 | laptop end-to-end, precomputed, median | 150-500 ms | 338.4 ms | inside |
| L5 | cold start Init Duration, median | 800-3,000 ms | 572.1 ms | **MISS** (lower) |
| L6 | max memory used | 150-450 MB | 220 MB | inside |
| L7 | laptop TCP+TLS connect, median | 100-400 ms | 492.2 ms | **MISS** |
| L8 | live fallbacks in the warm run | 0 | 0 | inside |

## What the numbers say (plain language)

- **The ranker is the cost, not the database.** Of the 91 ms of a precomputed request, 78 ms is the gradient-boosted ranker scoring about 230 candidates with 351 trees in numpy.
  The same code takes about 11 ms on the laptop. Lambda at 1,024 MB gets well under one CPU core (AWS: one full vCPU at 1,769 MB), and the prediction scaled the laptop figure
  far too optimistically; that is why L1 missed. Possible fixes, NOT tried (they would be a new experiment, not a re-run): more memory (1,769 MB = 1 vCPU, higher cost per ms),
  fewer trees, or a faster tree evaluator.
- **Live retrieval adds only about 31 ms at the median.** The two Zilliz searches inside Frankfurt take 34 ms together; DynamoDB reads take about 3 ms.
- **Cold starts: the short Init is misleading.** The function builds its app (loads the 38 MB bundle, starts the stores) on the FIRST request, not during Lambda's init phase.
  So Init is short (572 ms, below the predicted range: L5 measured the wrong phase) and the first request is long (about 1.7 s). A user hitting a cold container waits roughly
  2.3 s in total; warm requests are about 0.1 s on the server.
- **The bandit-stage median (0.1 ms vs 3.3 ms) is a threshold effect, not a cost of live mode.** Only 98 of the 200 users (49%) have an eligible new-release arm
  (checked locally with exact search: identical in both modes), so the median sits on the boundary between "no DynamoDB read" and "one read of about 3 ms". Live mode's
  approximate lane tips it just over.
- **From India, the network dominates the user's wait:** opening a connection takes about 490 ms (L7 missed: it is slower than predicted), and with a reused connection a
  request still takes 340-370 ms end to end, of which only 90-120 ms is server time.

## Demo (`module9_demo.html`, from real requests to the deployed function)
- A. M2, the same user mid-session vs after a 3-day break: 2 of 18 titles change (context matters a little for this user; one user only, illustration).
- B. M3b, SIMULATED device, banner "Synthetic device effect, pipeline demonstration, not a finding": average Drama-without-Comedy titles in the top 18 over 40 users:
  **tv 5.9, mobile 3.5** (M2, which ignores device: 6.7). The planted effect (mobile likes of those titles removed in training) is visible end to end; it says nothing about real viewers.
- C. Live vs precomputed for one user: 16 of 18 titles shared; no fallback.
- D. UCB new-release slots under SIMULATED feedback (SIMULATION, NOT AN A/B TEST): 200 requests, 145 impressions (only about half the users have eligible new releases),
  simulated like rate 40%.
