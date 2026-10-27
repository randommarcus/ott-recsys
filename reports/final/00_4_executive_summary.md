# Executive Summary

**The problem.** India has about 601 million OTT viewers and 148 million paid subscriptions (afaqs!, 2025). Each platform offers thousands of titles, but a viewer decides within seconds whether the home screen has anything worth watching. Recommendations decide what fills that screen. They also have to give new releases a fair chance, use the viewing context where it helps, and run fast and cheaply.

**What I built.** Following the project brief, I built a complete recommendation engine with four parts and deployed it on AWS:

1. **Candidate generation:** a two-tower model in TensorFlow narrows 33,392 movies to 200 candidates per user, and a Sentence-BERT content lane adds 30 new releases that no rating-based model can see. The vectors were also loaded into a Zilliz (Milvus) vector database.
2. **Ranking:** an XGBoost ranker with 37 inputs, including an NCF score from TensorFlow, the time of day and the user's session, orders the candidates.
3. **New releases:** a UCB bandit fills 2 of the 20 slots on screen with new titles.
4. **Serving:** a numpy-only AWS Lambda function with DynamoDB, in precomputed and live vector-search modes.

The data was MovieLens 25M (a 20,000-user sample, 3.1 million ratings) with TMDB movie descriptions. Device is not in the data, so every device value is SIMULATED, and the bandit was evaluated in a SIMULATION, NOT AN A/B TEST.

**How I evaluated it.** I split the data by time, compared every model with simple baselines, and wrote down my predictions before every major run. The test set was locked in code and opened exactly once, for five pre-registered claims.

**What I found.** All five headline results pointed in the predicted direction:

- **The learned ranker works.** It improved the quality of the top 10 (NDCG@10) by 18% over the candidate generator's own order (+0.0245, 95% interval +0.0229 to +0.0261).
- **Context barely helps on this data.** Time and session added under 1% (+0.0013). Time of day on its own made results slightly worse.
- **The two-tower model is the better retriever.** It found 58.5% of users' future likes in its top 200, against 53.8% for item-based nearest neighbors.
- **Tuning the bandit mattered more than the algorithm.** In the simulation, UCB with a tuned constant captured 94% of the possible gain, beating the textbook UCB1 by 42% and Thompson sampling by 4%.
- **The content lane is the only path for brand-new titles.** Rating-based models scored zero on them.

The deployed service answered warm requests in 91 ms (precomputed) and 122 ms (live) of server time, with zero errors in 400 timed requests. Most of that time is the ranker, not the database. Cold starts take about 2 seconds. Live vector search returned about 92% of the exact candidates and 93% of the same top-18 titles. Serving would cost roughly $3 per million requests at published AWS prices.

**What I recommend.** For an Indian OTT platform:

1. Invest first in a two-stage system with a learned ranker; it gave the biggest, most reliable gain from data the platform already has.
2. Treat context as a hypothesis: log real local time and device, and keep them only if an online test shows a gain.
3. Reserve the last two slots of the first screen for new releases, filled by a tuned bandit.
4. Serve precomputed recommendations by default, use live search where freshness matters, and fix the ranker's speed before scaling.
5. Confirm everything with a two-week A/B test before rollout, because all results here are offline.

**Limitations.** MovieLens ratings are a stand-in for viewing, the audience is not India's, device is simulated, and the bandit is simulated. The read-only database key on the free vector-database tier could still write, which I accepted as a stated limitation for the demo.
