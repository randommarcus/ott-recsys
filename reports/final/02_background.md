# 2. Background and Literature

This chapter covers the ideas the project builds on. I kept it to what is needed to follow the later chapters, and for each idea I note where it shows up in my own design.

## 2.1 Two Ways to Recommend

There are two classic families of recommenders.

**Collaborative filtering** learns from behavior: people who liked the same things in the past will probably like the same things in the future. It needs no knowledge of what a movie is about. Item-based nearest neighbors, which recommends movies similar to the ones a user already liked (similar meaning "liked by the same people"), was one of the first methods to work at web scale (Sarwar et al., 2001). Matrix factorization, popularized during the Netflix Prize, instead learns a short vector for every user and every movie so that their product predicts a rating (Koren et al., 2009). Most modern models, including the two-tower model in this project, are descendants of that idea.

**Content-based filtering** looks at the items themselves: genres, cast, descriptions. It can recommend a movie on the day it is added to the catalog, which collaborative filtering cannot. However, a description says less about taste than the behavior of thousands of other viewers, so content methods are usually weaker for established titles.

Real systems combine both. In my system, the main candidates come from a collaborative model and the new-release lane comes from a content model (Section 4.3).

## 2.2 The Two-Stage Design

Scoring an entire catalog with a heavy model for every request does not scale. YouTube's published design splits the job in two: a fast candidate generator narrows millions of videos to hundreds, and a ranker with many more features orders those hundreds (Covington et al., 2016). This two-stage pattern is now standard, and it is the backbone of my project (Figure 4.1).

The candidate generator is often a **two-tower model**: one neural network encodes the user, another encodes the item, and the score is how closely the two vectors align. Because item vectors can be computed in advance, finding the best candidates becomes a nearest-neighbor search. One known weakness is that training with in-batch negatives over-penalizes popular items, since they appear in many batches. Yi et al. (2019) proposed a correction based on each item's sampling frequency, which I used (and which turned out to be essential, Section 4.3).

## 2.3 Text Embeddings for Content

To compare movie descriptions, they first have to become numbers. Sentence-BERT fine-tunes a BERT language model so that sentences with similar meaning end up as nearby vectors, which makes comparing two texts a simple dot product (Reimers & Gurevych, 2019). I used a small, fast version of it to embed TMDB plot descriptions for the new-release lane.

## 2.4 Neural Collaborative Filtering, and a Word of Caution

Neural collaborative filtering (NCF) replaces the plain dot product of matrix factorization with a neural network that can learn more complex patterns between user and item vectors (He et al., 2017). It became a common baseline in research and is one of the components the project brief asked for.

However, the field has a reproducibility problem. When Ferrari Dacrema et al. (2019) re-ran a set of recent neural recommendation papers, most of the reproducible ones could be beaten by simple, well-tuned older methods such as item-based nearest neighbors. This finding shaped how I worked. Every model in this project had to beat simple baselines on the same data, with intervals, before I believed it. As it turned out, NCF on its own did not beat the two-tower model's order, so I used it as one input to the ranker instead (Section 4.4).

## 2.5 Learning to Rank

Ranking is not the same as predicting a rating. What matters is the order of the list, and mistakes near the top matter more than mistakes at position 150. Learning-to-rank methods optimize the order directly. The LambdaMART family trains gradient-boosted decision trees on pairs of items for the same user, weighting each pair by how much swapping them would change a ranking metric such as NDCG (Burges, 2010). XGBoost is a fast, widely used implementation of gradient boosting that supports this kind of ranking objective (Chen & Guestrin, 2016). My ranker is an XGBoost model of this type.

## 2.6 Context-Aware Recommendation

Context-aware recommenders add information about the situation, such as time, location, device or company, to information about the user and item (Adomavicius & Tuzhilin, 2011). The idea is intuitive: the same person may want different things on a weekday morning commute and on a Saturday night. The practical question is how much context adds once a model already knows the user's taste and recent activity. That is what the context ladder in this project measures (Section 4.4).

## 2.7 Bandits for New Items

A multi-armed bandit is a model of repeated choice under uncertainty. Each option (an "arm") has an unknown reward rate, and the decision maker has to balance trying uncertain options with using the ones that already look good. Lattimore and Szepesvári (2020) give a full treatment of the theory.

Two strategies matter here. **UCB** (Upper Confidence Bound) adds an optimism bonus to each arm's observed average, which shrinks as the arm is tried more often; Auer et al. (2002) proved that a simple version, UCB1, performs close to the best possible in the long run. **Thompson sampling**, one of the oldest ideas in the field (Thompson, 1933), picks each arm with the probability that it is the best one, given what has been seen so far.

Evaluating a bandit offline is hard. You only see the outcome of what was actually shown. Li et al. (2010) showed how to replay logged data fairly, but only when the logged choices were made at random. MovieLens ratings are not random exposures, which is why my bandit results come from a simulation (Section 4.5).

## 2.8 Vector Databases

A two-tower model turns retrieval into a search for the item vectors closest to a user vector. Doing that exactly means comparing against every item, which gets slow as catalogs grow. Approximate nearest-neighbor indexes trade a little accuracy for a lot of speed. One of the most widely used is HNSW, a layered graph that lets the search jump quickly toward the right neighborhood (Malkov & Yashunin, 2020). Milvus is an open-source database built around such indexes (Wang et al., 2021), and Zilliz Cloud, which I used, is a hosted version of it. The price of approximation shows up in my results as live search returning about 92% of the exact candidates (Section 5.6).

## 2.9 Measuring Recommenders

Offline metrics such as Recall@K and NDCG (Järvelin & Kekäläinen, 2002) are the standard way to compare recommenders on historical data. They are convenient, but they measure agreement with past behavior, not business value. Jannach and Jugovac (2019) reviewed how companies actually measure the value of recommenders (click-through, sales, engagement, retention) and found that offline accuracy gains do not always translate into business gains. The reliable way to measure business impact is a controlled online experiment, an A/B test, run with care about sample size, duration and guardrail metrics (Kohavi et al., 2020).

This gap between offline and online evaluation runs through my whole report. Every result in Chapter 5 is offline. Chapter 6 discusses what an online test would need to confirm.

## 2.10 Where This Project Fits

Most published work studies one component at a time: a better retriever, a better ranker or a better bandit, usually measured offline on a single metric. In contrast, there are few end-to-end, openly documented examples that take a recommender from raw data all the way to a deployed, measured service, with every claim checked against baselines and a locked test set.

My project tries to fill that gap at a small scale. It builds all four components the brief required, connects them into one working service, and evaluates each part with the same discipline: predictions written down first, simple baselines, intervals, and one look at the test set. Its value is less in any single model and more in showing how the pieces fit together, what each one actually contributes, and what it costs to run.
