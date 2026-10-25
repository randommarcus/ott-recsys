# Project: Hyper-personalized OTT recommendation engine (MBA Sem 4)
Student project, ~1 month deadline. User: Sudarshan, MBA Business
Analytics, not a software engineer. Build it and teach me as we go.

## Required components (from the problem statement)
1. Candidate generation: two-tower model + Sentence-BERT embeddings in Milvus
2. Ranking: NCF + XGBoost ranker using context (device, hour, session length)
3. UCB multi-armed bandit for new releases
4. Serving: AWS Lambda + DynamoDB
TensorFlow is MANDATORY (my guide). Use TF/Keras for two-tower and NCF.
PyTorch only appears via sentence-transformers.

## Constraints
- Laptop: WSL2, 16GB RAM, CPU only. Use a data subset; small embeddings (64).
- Heavy runs go to free Colab. Vector DB: Zilliz Cloud free tier.
- Data: MovieLens + TMDB descriptions. Device is simulated; say so.
- Split data by time, never randomly. Always compare to baselines.
- Keep dataset-specific code in one config + one data module (reusable later).
- TensorFlow runs offline only; Lambda gets precomputed embeddings.

## Working rules
- One module at a time. Small commits with clear messages.
- After each module: explain in plain language what it does and why.
- Write simple tests. Keep a progress log in PROGRESS.md.
- Never commit secrets; use .env.
- Ask me before manual steps or anything that costs money.

## Fallbacks
two-tower fails -> SBERT retrieval; NCF fails -> matrix factorization;
Zilliz fails -> Milvus on EC2; Lambda fails -> FastAPI on EC2.
