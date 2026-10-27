# 6. What This Means for an OTT Business

The results in Chapter 5 are offline numbers on MovieLens. A platform manager would rightly ask what they mean in rupees, hours watched and subscriber churn. I cannot answer that directly without a live test. However, the results do point to clear priorities, and the serving measurements give real cost figures. This chapter covers both.

## 6.1 The Ranker Is Where the Value Is

The single biggest gain in the project came from the learned ranker. On the test set it improved the quality of the top 10 by about 18% over the candidate generator's own order (P1), and it raised the share of a user's future likes that land on a 20-title screen from 18.8% to 21.0%.

In business terms, that means more of the titles a person will actually enjoy show up where they look first. Gomez-Uribe and Hunt (2015) tie this kind of improvement to fewer cancellations at Netflix, and that is the mechanism I would expect on an Indian platform too: a viewer who finds something good quickly is less likely to leave the app or the subscription.

There is a catch, and it matters for any investment decision. Offline improvements do not always become business improvements (Jannach & Jugovac, 2019). An 18% better offline score is a strong reason to run an online test, not a guarantee of 18% more viewing.

Interestingly, the gain did not come from new data. Every input to the ranker was already available. It came from combining existing signals well. For a platform, that is good news: a learned ranker is a modelling investment, not a data-collection one.

## 6.2 Do Not Pay for Context Before You Can Measure It

The project brief put context (device, time and session) at the center. On this data, context added less than 1% (P3), and time of day on its own made the test result slightly worse.

I would not conclude that context is useless for Indian OTT. MovieLens has UTC timestamps and rating sprees, not local evenings and viewing sessions, so it is a poor place to find context effects (Section 5.4). On top of this, the synthetic test showed that even a strong planted device effect moves the offline metric only a little. Connected TV is growing fast in India (afaqs!, 2025), and phone and TV viewing may well differ.

My recommendation is to treat context as a hypothesis to test cheaply, not a feature to build first:

- log real local time, device and session data properly;
- add those inputs to the ranker (the pipeline already supports them);
- keep them only if an online test shows a gain.

## 6.3 Promoting New Releases Has a Price, and It Is Worth Paying

Content is an OTT platform's biggest cost, so new titles need an audience. The project shows both sides of that trade-off.

**The cost.** Reserving the last 2 of 20 slots for new releases left the top 10 untouched and reduced the share of future likes on screen by 1.4 percentage points (about 6.6% of what the ranker achieves). Putting the new titles inside the top 10 instead hurt the top 10 directly. If a platform wants to promote new releases, the bottom of the first screen is the cheaper place to do it.

**The benefit.** Without the reserved slots, new titles never reached the ranker's top 20 at all. With them, a tuned UCB strategy captured 94% of the possible gain over picking new titles at random (SIMULATION, NOT AN A/B TEST).

**The strategy choice.** Tuned UCB earned the most in the simulation but concentrated impressions on a few leading titles. Thompson sampling earned about 4% less but spread exposure more evenly. This becomes a business decision, not a technical one. A platform that mainly wants viewing should prefer tuned UCB. A platform with commitments to producers or content partners (for example, a guaranteed level of visibility for each new release) may prefer Thompson sampling, or tuned UCB with a minimum number of impressions per title.

## 6.4 What It Costs to Serve

Because the system was actually deployed and measured, I can estimate its running cost from real durations instead of guesses. Table 6.1 uses AWS's published US East prices: $0.20 per million Lambda requests, $0.0000166667 per GB-second of Lambda time, and $0.125 per million read units and $0.625 per million write units for on-demand DynamoDB (Amazon Web Services, n.d.-a, n.d.-b). Frankfurt prices are similar but not identical, so treat these as estimates.

**Table 6.1** *Estimated serving cost per million home-screen requests*

| Item | Precomputed mode | Live mode | How I estimated it |
|---|---|---|---|
| Lambda | about $1.87 | about $2.37 | 1 GB of memory for about 100 ms (precomputed) or 130 ms (live), plus the request fee |
| DynamoDB | about $1.00 | about $1.00 | one user record read; for about half the requests, a read of about 10 new-release counters and 2 counter updates |
| Zilliz vector search | none | not estimated | the free cluster was enough here; a production plan would cost extra |
| **Total** | **about $2.90** | **about $3.40 plus Zilliz** | |

To put that in context: a platform with one million daily users who each open the home screen three times a day makes about 90 million requests a month. At these rates, serving them in precomputed mode would cost roughly $260 a month (about ₹22,000, assuming ₹85 to the US dollar purely for scale), before logging, data transfer and the offline training runs. Simply put, the serving layer is not where the money goes. Content and marketing dwarf it.

The cost figures also point to two engineering priorities:

- **The ranker is the slow part.** It takes 78 of the 91 ms of a precomputed request (Section 5.6). Giving the function 1,769 MB, which AWS ties to one full processor core (Amazon Web Services, n.d.-c), should cut that time considerably at a similar cost per request, because the higher price per millisecond is offset by fewer milliseconds. Fewer trees would help as well. Both are cheap to try.
- **Cold starts are the bad case for users.** A user who hits a freshly started function waits about 2.1 to 2.4 seconds instead of about 0.1 seconds. Keeping a few instances warm (provisioned concurrency) removes this but costs money around the clock. For a home screen, I would first move the data loading into the start-up phase and measure again.

**Precomputed or live?** Precomputed mode is exact, faster and cheaper. Live mode costs about 31 ms more and returns about 93% of the same top 18, but it reflects catalog changes immediately, without waiting for the next batch job. A practical design is to serve precomputed candidates by default and use live search only where freshness matters, such as a "new this week" row.

## 6.5 What a Real Launch Would Measure

Before any rollout, the system should go through an A/B test: a random share of users gets the new recommender, the rest keep the current one, and both groups are compared over the same period (Kohavi et al., 2020). Table 6.2 sketches the metrics I would use.

**Table 6.2** *A proposed A/B test plan*

| Role | Metric |
|---|---|
| Primary | Viewing hours per active user per week |
| Secondary | Share of sessions that start a title from the first row; 30-day retention of subscribers; plays of new releases |
| Guardrails (must not get worse) | 95th-percentile response time; error rate; share of the catalog shown at least once (to catch a popularity trap); complaints and "not interested" actions |
| Design | Randomize by user, run at least two full weeks to cover weekly patterns, and decide the sample size in advance from the smallest effect worth acting on |

The bandit, in particular, needs this. Its numbers in this report come from a simulation, and its real value depends on how often people click a new title, which the simulation does not model.

## 6.6 Build or Buy?

Managed recommendation services exist, for example Amazon Personalize, and for a platform without a data science team they are a sensible starting point. They are faster to launch and need less specialist staff.

Building in-house, as in this project, makes sense when the recommender is a core part of the product. It allows full control over the ranking logic (such as reserved slots for new releases), the evaluation and the cost. This project shows that the in-house route does not have to be expensive to run. The hard part is the people and the discipline of evaluation, not the servers.

## 6.7 My Recommendations

If I were advising an Indian OTT platform based on this work, my verdict would be:

1. **Invest in a two-stage system with a learned ranker first.** It delivered the largest and most reliable gain, using data the platform already has.
2. **Test context before building around it.** Log real local time and device data, then let an online test decide.
3. **Reserve the last two slots of the first screen for new releases**, filled by a tuned bandit, and choose between tuned UCB and Thompson sampling based on how much the platform values even exposure across titles.
4. **Serve precomputed recommendations by default**, add live vector search where freshness matters, and fix the ranker's speed before scaling up.
5. **Prove it online.** Run a two-week A/B test with clear guardrails before rolling anything out.
