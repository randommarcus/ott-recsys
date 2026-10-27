# 1. Introduction

## 1.1 The Indian OTT Market Has a Choice Problem

India is now one of the largest streaming audiences in the world. According to the Ormax OTT Audience Report 2025, 601.2 million Indians watch content on OTT platforms, about 41% of the population. Active paid subscriptions stand at 148.2 million (a figure that includes telecom bundles and aggregators), and connected TV users grew 87% in a single year to 129.2 million (afaqs!, 2025).

As a result, platforms are competing hard for the same viewer's evening. Most households have access to more than one service, and each service offers thousands of movies and series in several languages.

That abundance creates a problem. A viewer opens the app, sees a home screen with a couple of dozen titles and decides within seconds whether anything looks worth watching. If nothing does, they scroll for a while and then switch to another app, or to YouTube, or close the app altogether.

Recommendation systems exist to solve exactly this. Netflix has published one of the few public estimates of what they are worth: about 80% of the hours streamed on Netflix came from recommendations, and the company estimated that personalization and recommendations together save it more than a billion US dollars a year, mostly through lower cancellations (Gomez-Uribe & Hunt, 2015). Even if an Indian platform gets a fraction of that effect, the business case is clear.

## 1.2 The Business Problem

From a business point of view, an OTT recommender has to do four things well.

**1. Put the right titles on the first screen.** The first row matters most, because many users never scroll past it. The system has to pick a handful of titles out of thousands, for each user, every time they open the app.

**2. Give new releases a fair chance.** Platforms spend heavily on new content and need people to see it. However, a brand-new title has no viewing history, so a system that learns from history tends to ignore it. This is usually called the cold-start problem.

**3. Use the moment, not just the person.** What someone wants at 11 pm on a phone may differ from what they want on Sunday afternoon on a TV. In principle, context such as time and device should improve recommendations. Whether it does in practice is an open question that this project tests.

**4. Run fast and cheaply at scale.** A recommendation that arrives after the home screen has loaded is useless. It also has to be cheap enough to compute for millions of users several times a day.

## 1.3 What I Set Out to Build

The project brief asked for a hyper-personalized OTT recommendation engine with four specific components. Table 1.1 maps each one to what I built and where it is covered in this report.

**Table 1.1** *Required components and how they were delivered*

| Required component | What I built | Chapters |
|---|---|---|
| Candidate generation with a two-tower model and Sentence-BERT embeddings in Milvus | ID-only two-tower model in TensorFlow for the main candidates, plus a 30-title new-release lane from Sentence-BERT descriptions; both sets of vectors loaded into Zilliz Cloud (hosted Milvus) | 4.3, 5.2, 5.6 |
| Ranking with NCF and an XGBoost ranker using context (device, hour, session length) | NCF in TensorFlow used as an input to an XGBoost ranker with 37 inputs; a ladder of models that adds hour, day and session; device tested with SIMULATED values | 4.4, 5.3, 5.4 |
| A UCB multi-armed bandit for new releases | UCB with a tuned constant filling 2 of 20 slots; compared with five other strategies in a simulation | 4.5, 5.5 |
| Serving on AWS Lambda and DynamoDB | A numpy-only Lambda function in Frankfurt with DynamoDB, plus live vector search in Zilliz; deployed, measured and torn down | 4.6, 5.6 |

The brief also set some ground rules that shaped the work: TensorFlow for the neural models, data split by time and never at random, every model compared with simple baselines, and simulated parts clearly labeled.

## 1.4 Research Questions

I turned the business problem into six questions. RQ1 to RQ4 match the five headline claims (P1 to P5) that were fixed, with predicted answers, before the final test (Section 3.6). RQ5 is answered from the supporting evidence, and RQ6 from the deployed system.

- **RQ1.** Does a learned ranker put better titles at the top than the candidate generator's own order? (P1, P2)
- **RQ2.** Does context (time of day, day of week, session) improve the ranking? (P3)
- **RQ3.** Does the two-tower model find more of a user's future likes than a classic item-based method? (P4)
- **RQ4.** Which bandit strategy gives new releases the most successful exposure, and does tuning matter? (P5, simulation)
- **RQ5.** Can a content lane surface new titles that the other methods cannot see at all?
- **RQ6.** Can the whole system run on low-cost serverless infrastructure with acceptable speed?

Chapter 5 answers each of them with numbers, and Chapter 8 sums up the answers.

## 1.5 Scope and Constraints

This was a one-month MBA project, built by one person on a laptop. That set some hard limits, and I think it is fair to state them up front.

- **Hardware.** A laptop with 16 GB of RAM (about 9 GB usable under Windows Subsystem for Linux) and no GPU. Everything was sized to fit: a 20,000-user sample instead of the full dataset, and 64-number vectors instead of the larger ones used in industry.
- **Budget.** Only free tiers and free credits: AWS free tier, the free Zilliz Cloud cluster and the free TMDB API. The cloud deployment was sized to stay within these allowances; Chapter 6 estimates what it would cost beyond them.
- **Data.** No real OTT viewing data is public, so I used MovieLens ratings with TMDB descriptions (Chapter 3). There is no device in this data, so every device value is SIMULATED.
- **Tools.** TensorFlow was required by my guide for the neural models. The deployed function had to work without it, which shaped the serving design (Section 4.6).

Within these limits, my aim was not to beat the state of the art. It was to build the whole pipeline end to end, measure every part honestly, and be clear about what the numbers do and do not show.

## 1.6 How This Report Is Organized

Chapter 2 reviews the methods this project builds on. Chapter 3 describes the data and how it was split. Chapter 4 explains how each component was built and how it was evaluated. Chapter 5 presents the results, starting with the five headline findings. Chapter 6 discusses what the results mean for an OTT business, including costs. Chapter 7 covers the limitations, risks and ethical questions, and Chapter 8 concludes with the answers to the research questions and ideas for future work. The appendices hold the full tables, the pre-registered predictions, the security design, the demo, a glossary and a statement on the tools I used.
