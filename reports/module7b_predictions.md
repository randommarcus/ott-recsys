# Module 7b predictions (written 2026-10-26, BEFORE any tuning run)

THIS IS A SIMULATION, NOT AN A/B TEST. Reward = "liked given rated".
Protocol: choose the UCB constant c (bonus c*sqrt(ln t / n)) from the grid {0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0} using the FIRST half of
the request stream and selection seeds only (per arm set); evaluate on the SECOND half of full-stream runs with DIFFERENT seeds. UCB1 (c = sqrt 2) is
reported on the same second half. For fairness epsilon-greedy's epsilon gets the same first-half tuning over {0.01, 0.02, 0.05, 0.1, 0.2}.
Reference from Module 7 (whole stream, 629 arms, likes per 1,000 impressions / regret as % of random's): random 382/100, greedy 647/44,
eps-greedy 0.1 774/16, UCB1 584/57, UCB1+prior 602/53, Thompson 790/13, oracle 851.

| # | Quantity | Prediction |
|---|---|---|
| T1 | Selected c | small: between 0.05 and 0.35 on every arm set (70%); NOT on a grid edge (75%) |
| T2 | Selected c differs across the three arm sets | at least two of the three differ (60%) |
| T3 | UCB(c*) vs UCB1 (c = sqrt 2), 629 arms, second half | large gain: regret as % of random's falls from ~55% to 15-25% (65%); gain over UCB1 > +100 likes per 1,000 (80%) |
| T4 | UCB(c*) vs Thompson, 629 arms | Thompson still ahead or tied: Thompson - UCB(c*) in [-10, +60] likes per 1,000 (60%); UCB(c*) beats greedy (80%) |
| T5 | UCB1-Tuned (no constant to tune) | beats UCB1 (85%) but is worse than UCB(c*) (75%) |
| T6 | UCB(c*) + content prior vs UCB(c*) | small positive gain, +0 to +25 likes per 1,000 (60%), smaller than the +18 seen for plain UCB1 at c = sqrt 2 only if exploration is already low (50%) |
| T7 | Tuned epsilon | eps* in {0.02, 0.05} (70%); tuned eps beats eps = 0.1 by less than 10 likes per 1,000 on the second half (70%) |
| T8 | Second half vs whole stream | every learning strategy's like rate is higher in the second half than over the whole stream (85%) |
| T9 | Second-half order on 629 arms | Thompson ~ eps* >= UCB(c*) > UCB1-Tuned > greedy > UCB1 > prior-only > random (35% for this exact order) |
| T10 | Exposure | UCB(c*) concentrates much more than UCB1: Gini rises from ~0.5 to >= 0.8 (70%) |
