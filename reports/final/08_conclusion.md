# 8. Conclusion and Future Work

## 8.1 Answers to the Research Questions

**RQ1. Does a learned ranker beat the candidate generator's own order?** Yes, clearly. On the test set, the M2 ranker improved NDCG@10 by +0.0245 (about 18%), and the ranker without context by +0.0232. Both gains were larger than I predicted.

**RQ2. Does context help?** Only a little, on this data. Time and session together added +0.0013 NDCG@10 (under 1%); time of day on its own slightly hurt. The synthetic device test showed that the pipeline can learn a device effect when one exists, but MovieLens holds no real device or local time, so real context effects remain untested.

**RQ3. Does the two-tower model retrieve better than item-based nearest neighbors?** Yes. It found 58.5% of users' future likes in its top 200, against 53.8% for item-kNN, a difference of +0.0473 that matched my prediction.

**RQ4. Which bandit strategy works best for new releases?** In the simulation, UCB with a tuned constant (c = 0.2) captured 94% of the possible gain, beating textbook UCB1 by 237 simulated likes per 1,000 impressions and Thompson sampling by 32. Tuning mattered more than the choice of algorithm: untuned UCB1 was the worst of the learning strategies, behind even plain greedy. These are SIMULATION results, NOT an A/B test.

**RQ5. Can a content lane surface titles the other methods cannot see?** Yes. On movies with no ratings in training, every collaborative method scored exactly zero, while the content model found some of them. A 30-title content lane appended to the candidate list found about 10% of users' likes on brand-new titles without hurting the rest of the list.

**RQ6. Can it run cheaply and fast on serverless infrastructure?** Yes, with one bottleneck. The deployed system answered warm requests in 91 ms (precomputed) and 122 ms (live) of server time, with zero errors across 400 timed requests, at an estimated $2.90 to $3.40 per million requests. The ranker takes most of that time, and cold starts take about 2 seconds.

## 8.2 What I Take Away

Looking back, three lessons stand out for me.

First, **the biggest gains came from combining signals, not from collecting new ones.** The ranker added about 18% using inputs the system already had. Context, which sounded like the exciting part, added almost nothing here.

Second, **tuning beat cleverness.** The textbook bandit looked like a poor choice until its one constant was tuned on held-out data. The same was true of small details like centering the text embeddings (a five-fold gain) and the popularity correction in the two-tower model (the difference between a working model and a broken one).

Third, **being strict about evaluation was worth the effort.** Writing predictions down first, comparing every model with simple baselines and looking at the test set only once made the results slower to get but much easier to trust. It also made my mistakes visible, which is the point.

## 8.3 Future Work

If I had more time, data or budget, these are the things I would do next, roughly in order of value:

1. **Run an online A/B test** on a real platform or a beta group, with viewing hours as the main metric and the guardrails from Table 6.2. Nothing else would tell as much.
2. **Use real viewing logs with local time and device**, then re-run the context ladder. This is the only way to answer RQ2 properly.
3. **Make the ranker faster:** give the function one full processor core, try fewer trees, and move data loading into the start-up phase to shorten cold starts.
4. **Add Indian-language content.** A multilingual text model would let the content lane handle Hindi, Tamil, Telugu and other titles, which matters far more for an Indian platform than for MovieLens.
5. **Treat fairness of exposure as a goal**, for example a bandit that guarantees each new release a minimum number of impressions, and measure the cost of that guarantee.
6. **Fix the approximate-search gap** by testing other index settings or a paid vector database tier, and confirm whether live search can reach the 99% recall the documentation suggests.

## 8.4 Closing

The brief asked for a hyper-personalized OTT recommendation engine with four specific parts. I built all four, connected them into one working service, deployed it on AWS, measured it and took it down again. The learned ranker delivered a clear gain, the content lane and the bandit gave new releases a path to the screen, and the whole thing ran for a few dollars per million requests.

To be fair, it is a strong prototype, not a finished product. It still needs real viewing data and an online test before anyone should bet a business on it. Still, every number in this report can be traced to a committed script and a saved result, and every claim was checked before it was made. For a one-month project, I am happy with that.
