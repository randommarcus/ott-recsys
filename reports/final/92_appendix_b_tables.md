# Appendix B. Full Result Tables

All tables in this appendix are secondary results, except where marked as a headline claim (P1 to P5). End-to-end (e2e) metrics count every liked movie; reachable (rr) metrics count only likes that were inside the 230 candidates.

**Table B.1** *Ordering the 230 candidates, test set, 19,530 users (mean, 95% interval in brackets)*

| Method | e2e NDCG@10 | e2e Recall@20 | rr NDCG@10 | rr Recall@20 |
|---|---|---|---|---|
| Random order | 0.0340 [0.033, 0.035] | 0.0553 | 0.0413 | 0.0950 |
| Popularity | 0.0861 [0.084, 0.088] | 0.1110 | 0.1042 | 0.1902 |
| Two-tower's own order | 0.1383 [0.136, 0.141] | 0.1880 | 0.1630 | 0.2836 |
| NCF, fold-in | 0.1095 [0.107, 0.112] | 0.1535 | 0.1316 | 0.2492 |
| NCF, stale | 0.1020 [0.100, 0.104] | 0.1430 | 0.1227 | 0.2338 |
| 50/50 blend, fold-in | 0.1410 [0.138, 0.143] | 0.1888 | 0.1668 | 0.2944 |
| 50/50 blend, stale | 0.1353 [0.133, 0.138] | 0.1828 | 0.1599 | 0.2848 |
| M0 ranker, no context | 0.1615 [0.159, 0.164] | 0.2068 | 0.1917 | 0.3256 |
| M1 ranker, + hour and day | 0.1597 [0.157, 0.162] | 0.2054 | 0.1893 | 0.3216 |
| **M2 ranker, + session (primary)** | **0.1628 [0.160, 0.166]** | **0.2100** | **0.1937** | **0.3315** |
| M2 with stale NCF | 0.1602 [0.157, 0.163] | 0.2055 | 0.1906 | 0.3245 |
| Oracle (perfect order) | 0.7765 | 0.5733 | 1.0000 | 0.9914 |

**Table B.2** *Retrieval over the whole catalog, test set, 19,530 users*

| Method | Recall@10 | Recall@50 | Recall@200 | NDCG@10 | Recall@200, established titles | Recall@200, cold titles |
|---|---|---|---|---|---|---|
| Two-tower (ID only) | 0.1193 | 0.3174 | **0.5850** | 0.1383 | 0.5856 | 0.0000 |
| Item-kNN | 0.0993 | 0.2759 | 0.5377 | 0.1258 | 0.5382 | 0.0000 |
| Content (centered SBERT) | 0.0103 | 0.0310 | 0.0783 | 0.0125 | 0.0783 | 0.0221 |
| Popularity | 0.0555 | 0.1634 | 0.3541 | 0.0743 | 0.3545 | 0.0000 |

**Table B.3** *Bandit simulation on the test window: simulated likes per 1,000 impressions (SIMULATION, NOT AN A/B TEST)*

| Strategy | 266 titles (20+ ratings) | 761 titles (5+ ratings, main set) | 2,830 titles (1+ rating) |
|---|---|---|---|
| Random | 411.3 | 371.6 | 385.9 |
| Greedy | 619.9 | 633.3 | 644.1 |
| Epsilon-greedy (0.05) | 707.8 | 769.0 | 844.9 |
| UCB1 (textbook constant) | 623.8 | 570.2 | 484.2 |
| **UCB, c = 0.2 (pre-registered)** | **727.1** | **807.3** | **871.6** |
| UCB1-Tuned | 722.0 | 774.5 | 747.9 |
| Thompson sampling | 720.4 | 775.3 | 786.6 |
| Oracle (knows the true rates) | 748.7 | 833.2 | 921.0 |

On the main set, the exposure Gini coefficient (0 = every title shown equally, 1 = one title gets everything) was 0.15 for random, 0.53 for UCB1, 0.90 for Thompson sampling, 0.91 for UCB1-Tuned, 0.95 for epsilon-greedy, 0.98 for UCB with c = 0.2 and 0.99 for greedy. Greedy never showed 88% of the titles; UCB with c = 0.2 showed every title at least once.

**Table B.4** *Cost of reserving slots for new releases, M2 ranking, test set (difference against no reservation)*

| Reserved slots | NDCG@10 | Recall@20 |
|---|---|---|
| 2 at the end (positions 19, 20) | 0.0000 | -0.0138 [-0.0146, -0.0130] |
| 3 at the end (positions 18 to 20) | 0.0000 | -0.0208 [-0.0217, -0.0198] |
| 2 inside the top 10 (positions 5, 10) | -0.0175 [-0.0180, -0.0170] | -0.0138 [-0.0146, -0.0130] |

**Table B.5** *Latency of the deployed service, 200 warm requests per mode (milliseconds)*

| | Precomputed | Live |
|---|---|---|
| Server time: median [95% interval] | 91.1 [85.9, 96.4] | 121.9 [121.2, 123.6] |
| Server time: 95th percentile [95% interval] | 118.5 [108.6, 120.7] | 144.2 [141.1, 157.3] |
| Server time: 99th percentile | 121.7 | 163.9 |
| Laptop end to end: median / 95th / 99th | 338 / 361 / 382 | 367 / 400 / 437 |
| Stage medians: DynamoDB read / search / inputs / ranker / bandit | 3.2 / 0.0 / 1.8 / 77.8 / 0.1 | 3.2 / 34.4 / 1.6 / 76.1 / 3.3 |
| Cold start (5 per mode): start-up / first request, medians | 583 / 1,487 | 561 / 1,786 |
| Peak memory | 204 MB | 220 MB |

**Table B.6** *Live search quality by Zilliz search level, 100 users*

| Level | Main-lane recall of the exact candidates | New-release lane recall | Top-18 overlap, live against precomputed | Same top-18 set |
|---|---|---|---|---|
| 1 (default) | 87.0% | 86.7% | 89.9% | 13 of 100 |
| 3 | 87.0% | 88.0% | 89.9% | 13 of 100 |
| **5 (chosen)** | **91.8%** | **92.6%** | **92.9%** | **24 of 100** |
| 7 | 91.8% | 93.3% | 92.9% | 24 of 100 |
| 10 | 91.8% | 93.4% | 92.9% | 24 of 100 |

**Table B.7** *The context ladder on validation, 15,530 held-out users, out-of-fold scores (supporting evidence)*

| Method | e2e NDCG@10 | e2e Recall@20 |
|---|---|---|
| Two-tower's own order | 0.1242 | 0.2321 |
| NCF order | 0.0899 | 0.1836 |
| 50/50 blend | 0.1233 | 0.2340 |
| M0 ranker | 0.1370 | 0.2508 |
| M1 ranker | 0.1379 | 0.2512 |
| M2 ranker | 0.1388 | 0.2529 |
| M3a ranker (+ SIMULATED independent device) | 0.1389 | 0.2522 |
