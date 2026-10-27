# Appendix G. Glossary for Non-Technical Readers

**Table G.1** *Terms used in this report*

| Term | Plain meaning |
|---|---|
| A/B test | Showing two versions to randomly split groups of real users and comparing what they do |
| Bandit (multi-armed) | A method that decides what to show next while still learning which options are good, balancing trying new options with using proven ones |
| Baseline | A simple method that a new model must beat to be worth its complexity |
| Bootstrap interval | A range for a result, found by resampling the users many times; if it excludes zero, the difference is unlikely to be luck |
| Candidate generation | The first, fast step that narrows thousands of titles to a few hundred for each user |
| Cold start | The difficulty of recommending a title (or to a user) with no history yet |
| Cold start (serverless) | The extra delay when a cloud function has to start up before answering its first request |
| Collaborative filtering | Recommending from what similar people liked, without knowing what the items are about |
| Content-based filtering | Recommending from what the items are (genre, description) |
| Embedding | A list of numbers that represents a movie, a user or a text, so that similar things get similar numbers |
| Fold-in | Updating a user's vector with new likes without retraining the whole model |
| Gini coefficient (exposure) | How unevenly impressions are spread across titles: 0 means equal, 1 means one title gets everything |
| Lambda (AWS) | A cloud service that runs code only when a request arrives and charges by the millisecond |
| NDCG@10 | A score for the top 10 of a list that gives more credit for liked titles near the top |
| Out-of-fold | Scoring each user with a model that was trained without that user, to avoid flattering results |
| Pre-registration | Writing down predictions before running an experiment |
| Ranker | The second step that orders the candidates using many signals |
| Recall@K | The share of a user's future likes that appear in their top K |
| Service control policy | A rule set by an organization's owner that overrides permissions inside its member accounts |
| Two-tower model | A neural network with one part for users and one for items, trained so that a user's vector points toward items they will like |
| UCB | Upper Confidence Bound, a bandit method that adds an optimism bonus to options that have not been tried much |
| Vector database | A database that quickly finds the stored vectors closest to a given one |
