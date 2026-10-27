# Appendix H. Tools Used and AI Assistance

## H.1 Software and Services

- **Modelling:** Python 3.12, TensorFlow 2.20 and Keras 3 (two-tower model, NCF), sentence-transformers (Sentence-BERT), XGBoost (ranker), numpy and pandas.
- **Data:** MovieLens 25M (GroupLens Research) and The Movie Database (TMDB) API.
- **Cloud:** AWS Lambda, DynamoDB, Systems Manager Parameter Store, CloudWatch Logs and IAM, in eu-central-1 (Frankfurt); Zilliz Cloud free cluster (hosted Milvus), also in Frankfurt.
- **Report:** matplotlib for the figures and python-docx to assemble this document, both driven by scripts in the repository.

## H.2 Use of an AI Assistant

I used Claude Code (Anthropic), an AI coding assistant, throughout this project.

**What the assistant did.** Under my direction, it wrote most of the code and the automated tests, ran the analyses on my laptop, proposed designs and options at each step, drafted the module reports and the progress log, and drafted this report in my writing style from my earlier published articles, which I then reviewed and edited.

**What I did.** I set the scope and the order of work, chose between the options at every decision point (for example, the evaluation rules, the cloud region, how to handle the database key, and which results to treat as findings), approved every step that could cost money, created and controlled the AWS and Zilliz accounts, and ran every command that touched the cloud myself. I reviewed the results and the text and take responsibility for the claims in this report.

**Safeguards.** The assistant was blocked from reading my secrets file and my AWS credentials. Every result in this report comes from a committed script and a saved file, so the numbers can be checked independently of the assistant. The text was checked by a script for typical AI phrasing and against the saved results for every number.
