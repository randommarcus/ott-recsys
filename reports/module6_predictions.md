# Module 6 predictions (written 2026-10-26, BEFORE any ranker code was run)

Setting: 18,530 validation users; the 230-slot lists (ID-only two-tower top 200 + 30 new-release-lane picks); label = liked in
validation; candidates released AFTER the user's request time are removed for every method (decision approved by the user).
Reference (Module 5, unfiltered lists, held-out users, end-to-end NDCG@10 / Recall@20): two-tower order 0.1233 / 0.230;
NCF 0.0890 / 0.181; untuned 50/50 blend 0.1226 / 0.232; popularity 0.0601 / 0.117; random 0.0255 / 0.057.
Facts measured before the plan: 7.9% of slots (12.4% for requests before 2003) are movies not yet released at request time;
request-to-last-interaction gap has a median of 12 seconds; median session length so far at request is 32 ratings.

| # | Quantity | Prediction |
|---|---|---|
| 1 | Release filter alone, two-tower order | NDCG@10 change within +-0.003; Recall@20 down 0.002 to 0.006 (likes on unreleased titles become unreachable) |
| 2 | Ranker M0 (no context) vs two-tower order | Beats it (80% confident): +0.010 to +0.025 NDCG@10, +0.01 to +0.03 Recall@20 |
| 3 | M0 vs NCF order and vs the 50/50 blend | Beats both (90%) |
| 4 | +hour/day (M1 - M0) | No detectable gain (80%): paired interval includes 0 |
| 5 | +session (M2 - M1) | Positive and significant (65%), +0.005 to +0.02 NDCG@10, concentrated in long rating sprees (artifact of bulk rating) |
| 6 | +simulated device, independent of behaviour (M3a - M2) | Null (90%): abs diff <= 0.001, interval includes 0; feature importance small but non-zero |
| 7 | Synthetic scenario (b): M3b - M2b | Clear gain (90%): +0.005 to +0.02 NDCG@10 |
| 8 | Shuffled-device control on the same synthetic labels: M3b_shuffled - M2b | Gain vanishes (85%): abs diff <= 0.001, interval includes 0; M3b - M3b_shuffled clearly positive |
| 9 | Top features by gain | Two-tower score or rank first (70%); hour, day and device near the bottom |
| 10 | Lane items in the ranker's top 20 | <= 2% of top-20 slots; the ~19 in-list lane likes are far too few to judge lane quality |
| 11 | Held-out 15,530 users vs all 18,530 | gap <= 0.003 NDCG@10 |
| 12 | Trees chosen by early stopping | 100-400 |
| 13 | Time | feature build <= 10 min; whole ladder (35 fits) <= 60 min |
