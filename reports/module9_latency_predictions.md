# Module 9, checkpoint 5: latency predictions (written BEFORE any latency measurement)

Committed before `scripts/latency_study.py` has ever run. Honest disclosure: these were written AFTER seeing the checkpoint-4 smoke test (first-call server times
128-155 ms precomputed, 9.9-11 ms warm M3b, 422-439 ms live including the SSM read and the first Zilliz connection) and the laptop measurements (handler compute
about 13 ms per request on the laptop; laptop-to-Zilliz search 241 ms median with connection reuse). They are therefore informed guesses, not blind ones.

Setup that the numbers refer to: Lambda `ott-recsys-serve` in eu-central-1, Python 3.12, x86_64, 1,024 MB; Zilliz free cluster in aws-eu-central-1, search level 5;
DynamoDB on-demand in eu-central-1. Client = the laptop in India calling the IAM-signed Function URL over one reused HTTPS connection. Sequential requests only.

| id | Quantity | Prediction |
|---|---|---|
| L1 | warm server time (`server_total`), precomputed mode, median | 8 to 30 ms |
| L2 | warm server time, live mode, median | 25 to 150 ms |
| L3 | warm server time, live minus precomputed, difference of medians | 15 to 130 ms (almost all of it the two Zilliz searches) |
| L4 | warm client end-to-end through the Function URL from the laptop, precomputed, median | 150 to 500 ms (network dominates) |
| L5 | cold start: Lambda "Init Duration", median | 800 to 3,000 ms |
| L6 | max memory used (any request) | 150 to 450 MB of 1,024 |
| L7 | TCP+TLS connect time from the laptop to the Function URL host, median | 100 to 400 ms |
| L8 | live-mode fallbacks to precomputed during the warm run | 0 |

Machine-readable copy (the study script reads this block and prints measured vs predicted; do not edit after the run):
```json
{"L1": [8, 30], "L2": [25, 150], "L3": [15, 130], "L4": [150, 500], "L5": [800, 3000], "L6": [150, 450], "L7": [100, 400], "L8": [0, 0]}
```

Interpretation rule: a measured value outside its range is reported as a miss, as it is; no re-run to "fix" a miss. The headline is server-side latency (L1-L3);
client time from India (L4, L7) is reported separately and is not a property of the serving design.
