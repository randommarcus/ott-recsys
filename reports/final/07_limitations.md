# 7. Limitations, Risks and Ethics

Like any product review, this one would be incomplete without the weak points. Some of them come from the data, some from the way I evaluated, and some from engineering shortcuts. None of them overturns the main results, but each one limits how far those results can be stretched.

## 7.1 The Data Is a Stand-In

- **Ratings are not viewing.** Every result predicts what a user will rate 4 stars or more, not what they will click, watch or finish. Most of the "sessions" in MovieLens are rating sprees, not evenings of viewing (Section 3.4).
- **The audience is not India's.** MovieLens users are movie enthusiasts on a research website, rating a largely English-language, Hollywood-heavy catalog. An Indian OTT audience watches series as well as films, in many languages, often on phones and shared family TVs. The methods transfer; the exact numbers do not.
- **Context data is weak.** Timestamps are UTC, and there is no device at all. Every device value in this report is SIMULATED, so the report can say nothing about real device effects.

## 7.2 The Evaluation Has Limits

- **One test period, known users.** The test set covers each user's next period of activity. It measures how well the system serves people it already knows, not brand-new subscribers.
- **Some calendar leakage remains.** The per-user split keeps each user's own future out of training, but other users' later ratings can still be in training for someone else's test period. The calendar split used for new releases and the bandit does not have this issue.
- **Many comparisons, five findings.** The test run computed many numbers. Only the five pre-registered claims are findings. Some of the secondary differences with intervals that exclude zero may be chance.
- **The bandit is simulated.** Its results assume that each title has a fixed like rate estimated from real ratings, that a shown title is always rated, and that rates do not change over time. Real viewers ignore many recommendations, and tastes shift. SIMULATION, NOT AN A/B TEST.
- **Offline is not online.** None of the results measures viewing time, retention or revenue. Chapter 6 explains what an A/B test would need to show.

## 7.3 Engineering Shortcuts and Their Risks

- **Live search is approximate.** On the free Zilliz cluster, live search returned about 92% of the exact candidates even at the highest search level, and I could not find out why it stopped there (Section 5.6). Precomputed mode does not have this issue.
- **The "read-only" database key could write.** I created two keys intended to be read-only for the Lambda function. On the free Zilliz cluster, both could still insert data, which my verification script caught. I accepted this as a known limitation for the demo, kept the key separate from the admin key so it can be revoked on its own, stored it encrypted, and limited it to one cluster that held nothing but rebuildable project data. A production system should use a paid tier with properly enforced roles.
- **Cold starts are slow.** A request that lands on a freshly started function waits about 2.1 to 2.4 seconds, because the function loads its data on the first request (Section 5.6).
- **The account is temporary.** The AWS account runs on sign-up credits. If it is on AWS's free plan, it closes when the credits run out or after six months. The deployment was always meant to be torn down after measurement, and it was.

## 7.4 A Lesson in Cloud Governance

One episode deserves a mention because it is a common real-world trap. My first AWS account was created through a sign-up route that placed it inside an organization I did not control. That organization had a service control policy, a rule set by the organization's owner that overrides any permission inside the account. My own permission tests showed that creating log groups, parameters, DynamoDB tables and Lambda functions was blocked in both regions, even though my own policy allowed all of them.

No amount of fixing on my side could have helped. I closed that account and opened a standalone one, where the same tests passed. The lesson for a business is simple: before planning a cloud project, check who controls the account and what rules sit above it. My permission-test script made the problem visible in two minutes instead of after a failed deployment.

## 7.5 Ethics and Responsible Recommendation

**Privacy.** Viewing history reveals a lot about a person: their tastes, beliefs, language and sometimes their health or relationships. In India, the Digital Personal Data Protection Act, 2023 sets rules for how such personal data is collected, used and kept (Ministry of Law and Justice, 2023). This project used only public, anonymous research data. A real deployment would need consent, a clear purpose, data minimization and limits on how long logs are kept. Even in this project, logs were kept for only 7 days, and the only user data that left the laptop was anonymous MovieLens user numbers with their model vectors and rated-movie lists, all deleted after the measurements.

**Popularity bias.** Recommenders tend to push already popular titles, because those have the most data. Over time, this can starve the long tail and smaller producers (Abdollahpouri et al., 2017). The reserved new-release slots in this project are one countermeasure, and catalog coverage should be a guardrail metric in any online test (Table 6.2).

**Feedback loops.** A recommender trained on what people watched, when what they watched was shaped by the recommender, can make everyone's suggestions more alike over time without making anyone happier (Chaney et al., 2018). Exploration, which the bandit provides, and occasional random exposure in the logs both help to break this loop.

**Fairness to new titles.** The choice between tuned UCB and Thompson sampling (Section 6.3) is also an ethical one. Tuned UCB earns more but gives most impressions to a few titles; Thompson sampling spreads them more evenly. A platform should make that choice deliberately, not by default.

**Honest labeling.** Simulated and synthetic results can easily be mistaken for real findings, especially in a business setting. That is why every device value, the synthetic M3b scenario and every bandit number in this report carry a label, both in the text and in the demo itself.

## 7.6 Use of an AI Assistant

I built this project with the help of Claude Code, an AI coding assistant. It wrote most of the code under my direction, ran the local analyses and drafted much of this report in my writing style, which I then reviewed and edited. Many options came from the assistant, but the final decisions on scope, design trade-offs, cloud spending and what to claim were mine, and every cloud action was run by me. Appendix H gives the full statement. I mention it here because it is a real factor in how the work was done, and the reader should know it.
