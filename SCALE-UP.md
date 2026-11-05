# Scale-up plan

Goal: a general data-analytics tool that works with many model providers and many data sources, keeping the core rule: no silent choices that could change a number, and every number traceable to a query result.

## Done in this repo

- One `CHAT_PROVIDERS` registry: Baseten, Gemini, Groq, OpenRouter, Cerebras, Ollama, any OpenAI-compatible endpoint, plus the original Meta path. 429 backoff honours `Retry-After`.
- Databases: SQLite, PostgreSQL, BigQuery (optional, cost-capped).
- Tests use fakes; no live provider or BigQuery call has been made.

## Next steps (smallest first)

1. Live check of each provider with one real question; fix defaults.
2. Live check of BigQuery on a public dataset with a Data Viewer + Job User service account.
3. `datasets.toml` so a database can be added without editing Python.
4. More warehouses, one backend file plus tests each: Snowflake, Databricks SQL, Redshift, MySQL, SQL Server.
5. CSV / Excel / Parquet (local, S3, GCS) via a read-only DuckDB.
6. Secrets handling beyond `.env`.
7. Docker image, then a chat-style web UI, then multi-user auth.
8. Provider-agnostic evaluation set with known answers.

## Risks

- Free tiers may train on prompts: public or synthetic data only; later, block free providers on connected company databases.
- Small models may loop on verification rejections: measure before recommending one.
- Olist is CC BY-NC-SA: keep it out of anything shared or commercial.
- The 5 GB BigQuery scan cap is a guess.
