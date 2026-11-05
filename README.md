# Data Viz Agent

> A general data-analytics agent built on the foundation of the Meta Data Analytics and Visualization Agent hackathon project (RanveerRD/Meta-Data-Analytics-and-Visualization-Agent). Core rule kept: no silent choices that could change a number, and every number traceable to a query result. See [SCALE-UP.md](SCALE-UP.md) for the plan.

An AI data analyst for small and medium businesses, built for the Meta Global AI Developer Hackathon (26 Oct – 4 Nov 2026).

Ask a question in plain English. The agent plans its work, writes and runs read-only SQL, checks every number against what the database actually returned, and answers. Its defining rule: **it never makes a silent choice that could change a number.** If "revenue", "top", the time period or the date column could mean more than one thing, it stops and asks you, offering options taken from your real data, and then lists every decision (with proof) under the answer.

The model is Meta's Muse Spark, called through the Meta Model API. The database layer is swappable: point the agent at any SQLite file or PostgreSQL database you have read access to. It can answer across several databases at once (`--dataset all`), draw charts, publish them to a BI tool, and answer over WhatsApp.

## Contents

- [What it looks like](#what-it-looks-like)
- [Datasets used in the demo](#datasets-used-in-the-demo)
- [The five agent patterns](#the-five-agent-patterns)
- [Safety](#safety)
- [How a question is answered](#how-a-question-is-answered)
- [Setup](#setup)
- [Console commands](#console-commands)
- [Charts and BI tools](#charts-and-bi-tools)
- [WhatsApp front end (prototype)](#whatsapp-front-end-prototype)
- [Adding a database](#adding-a-database)
- [Model and cost](#model-and-cost)
- [Tests](#tests)
- [Known limits](#known-limits)
- [Roadmap](#roadmap)
- [Credits and licences](#credits-and-licences)

## What it looks like

```
adventureworks> what is the profit margin
...
clarification: Which period should the profit margin cover?
  1. All data (2022-05-30 to 2025-06-29) [31465]
  2. 2024 only [14244]
  0. something else (type your own)
choice (number): 0
type your answer: 2023

clarification: Which cost should I use for the 2023 profit margin?
  1. History-matched standard cost (28657049.85) [28657049.85]
     Matches cost in effect on each 2023 order date.
  2. Current standard cost (30680034.98) [30680034.98]
     Values all 2023 units at current standard cost.
choice (number): 1

2023 profit margin is 9.33% — revenue of 31604921.95 less history-matched
standard cost of 28657049.85 leaves profit of 2947872.09.

Decisions:
- period: 2023 (you chose: "2023")
- revenue: Subtotal revenue (31604921.95) (you chose: option 1)
- cost: History-matched standard cost (28657049.85) (you chose: option 1)
- order_status scope: status 5 includes all 31465 orders (not asked: no alternative found, query q2)

Gaps:
- none found

verification: 4 numbers matched, 0 unmatched
```

## Datasets used in the demo

The agent ships with no data. The demo uses two public sample datasets, which you download yourself (setup steps below). Neither is stored in this repository.

| Dataset | Database | Source and licence |
| --- | --- | --- |
| Olist Brazilian E-Commerce Public Dataset | SQLite | By Olist, via Kaggle, CC BY-NC-SA 4.0. Used here only for a non-commercial hackathon demonstration; not redistributed |
| AdventureWorks sample database | PostgreSQL | By Microsoft (`microsoft/sql-server-samples`, MIT licence), PostgreSQL conversion scripts from `lorint/AdventureWorks-for-Postgres` |

## The five agent patterns

| Pattern | Where it lives |
| --- | --- |
| Tool use | Ten tools with strict function schemas (`agent/tools/`): `list_tables`, `get_schema`, `run_sql`, `set_checklist`, `plan_action`, `record_finding`, `ask_clarification`, `propose_note`, `submit_answer`, `make_chart` |
| Reflection | `record_finding` and `submit_answer` check the work in code: every number must appear in a recorded query result, every checklist item must be proven or listed as a gap, and every decision needs proof. Rejections explain what to fix; after three the answer is flagged as unverified |
| Plan and execute | The agent must call `set_checklist` (what must be settled, and what evidence settles it) and then `plan_action` before `run_sql` is allowed |
| Orchestrator / worker | `--dataset all`: a routing-only planner picks which databases are relevant, one ordinary agent run (the worker) answers per database, and a merge step combines the answers. Every number in the merged reply must come from a worker's verified answer (`agent/orchestrator.py`) |
| Memory and context management | A per-question analysis record on disk (append-only audit trail), history compaction between questions, and a definitions store (`/notes`, `/forget`) where a person approves saved defaults; the model can only propose and can never approve its own notes |

## Safety

The agent can read data and nothing else. There are no shell or file tools, and no tool lets the model write to a database, publish to a BI tool or send a message.

- **PostgreSQL:** a dedicated `analyst_reader` role with SELECT-only rights, `default_transaction_read_only`, a 10-second `statement_timeout`, re-applied before every query so nothing a query does can switch it off. Only read queries execute (every statement runs inside a read-only cursor).
- **SQLite:** opened `mode=ro` with `PRAGMA query_only` and an authorizer that allows only SELECT/READ/FUNCTION actions and a short list of read-only pragmas (blocks `ATTACH`, writes and every other pragma), plus a 10-second timeout.
- **Everywhere:** one statement per call, 200-row result cap, tool output is treated as data and never as instructions (prompt-injection stance).
- **Charts:** drawn by code from query results the agent has already recorded and cited; the model cannot invent a number on a chart.
- **BI publishing:** only a person can trigger it (`/publish`), and saving to Metabase needs a console `y/n`.
- **WhatsApp:** every webhook request is checked against Meta's HMAC signature, only phone numbers on an allowlist are served, the server listens on loopback only, and the token is sent only in an Authorization header.
- **Secrets:** API keys and database passwords live in `.env` (git-ignored), are read without printing, and never appear in logs or error messages.
- **Memory:** nothing becomes a saved default without a person pressing `y`.

## How a question is answered

1. `set_checklist` and `plan_action` fix what must be proven.
2. The agent reads only the schemas it needs, then asks for scope (for example the period) if the question doesn't state it.
3. It gathers one figure per candidate definition, then calls `ask_clarification`. Each option shows evidence from a recorded query and a plain-language explanation. You can type a number, `0` for your own answer, or a question about the options.
4. After your choices it runs only the chosen combination, records a finding for each key number, and submits.
5. Code, not the model, renders **Decisions** (stated by you, chosen by you, or the only value in the data), **Gaps**, and the verification line.

Every run is saved as JSON in `workspace/runs/` (git-ignored).

## Setup

Requirements: Python 3.12 or newer (developed on 3.14, the test suite also passes on 3.13). For PostgreSQL work you also need a PostgreSQL server and `psql`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env     # then fill in META_API_KEY (and ANALYST_READER_PASSWORD for Postgres)
```

To run the tests or Datasette, also install `requirements-dev.txt` (see Tests).

Create a Meta Model API key at the Meta Model API developer site and put it in `.env` as `META_API_KEY`. The key is never printed.

### Using your own database

The agent works on any SQLite file or PostgreSQL database you can open read-only. Register it once in `agent/datasets.py` (see [Adding a database](#adding-a-database)), then run `.venv/bin/python -m agent --dataset <name>`. For PostgreSQL, create a SELECT-only login for the agent first; `scripts/create_readonly_role_adventureworks.sh` is a worked example and prints seven PASS/FAIL checks.

### Demo dataset 1: Olist (SQLite)

1. Download the Olist Brazilian E-Commerce dataset from Kaggle (`olistbr/brazilian-ecommerce`) and unzip the 9 CSV files into `data/raw/olist/`.
2. Build the database: `.venv/bin/python scripts/load_olist.py` → `data/olist.sqlite`.
3. Run: `.venv/bin/python -m agent --dataset olist`

### Demo dataset 2: AdventureWorks (PostgreSQL)

Extra requirements: PostgreSQL 17, `psql`, `curl`, `unzip` and Ruby.

1. Start PostgreSQL (for example `brew services start postgresql@17`).
2. `scripts/load_adventureworks.sh` downloads Microsoft's AdventureWorks CSVs and the PostgreSQL install scripts and builds the `adventureworks` database (Ruby is used by the conversion script).
3. Set `ANALYST_READER_PASSWORD` in `.env`, then run `scripts/create_readonly_role_adventureworks.sh`. It creates the read-only role and prints seven PASS/FAIL checks.
4. Run: `.venv/bin/python -m agent --dataset adventureworks`

Connection settings default to `localhost:5432`, database `adventureworks`, user `analyst_reader`; override with `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`.

### Both at once

`.venv/bin/python -m agent --dataset all`, then ask for example: *"what is the top product category in each database?"* Clarification prompts are tagged with the database they come from.

## Console commands

| Command | What it does |
| --- | --- |
| `/datasets` | List available datasets (no model call) |
| `/notes` | List approved saved definitions for the current dataset |
| `/forget <key>` | Retire a saved definition |
| `/chart [n] [bar\|line]` | Draw one of the charts offered for the last answer |
| `/publish [n] [bar\|line]` | Share an offered chart through a BI tool (asks for approval before saving anywhere) |
| `/help` | Show this list |
| empty line | Exit |

At a clarification prompt: a number picks an option, `0` lets you type your own answer, Enter picks the "(last used)" option when one is marked.

## Charts and BI tools

**Drawing a chart.** Ask for one in plain words ("show monthly orders in 2017 as a line chart"), or use `/chart` after an answer. The `make_chart` tool only charts a query the agent has already run, recorded and cited in a finding. The result must not be cut by the row cap, the columns must exist, the value axis must be numeric, there must be 2 to 50 points, and any number in the title must come from the rows. Bar and line charts are supported. Axis labels are laid out by measuring the text (rotation, then widening, then thinning), and missing months or years in a time series break the line and add a code-written "Chart note: no data for ..." line instead of drawing a false connection. Charts are saved as PNG files in `workspace/charts/` (git-ignored).

After an answer the console offers up to three charts that fit the result; which are offered is decided by code, and you decide which is drawn.

**Publishing to a BI tool (`/publish`).** The same recorded SQL can be handed to an open-source BI tool, so the chart becomes interactive. Each dataset names its connector in the registry.

- **Olist → Datasette.** The connector builds a link that opens the query as a chart in Datasette (read-only, no write access). Development-only install: `.venv/bin/pip install -r requirements-dev.txt` (Datasette and its chart plugin), then in another terminal `.venv/bin/datasette serve --immutable data/olist.sqlite` (the console prints this hint with your path; Datasette's default address is `http://127.0.0.1:8001`). Set `DATASETTE_URL` if you use another address.
- **AdventureWorks → Metabase.** The connector saves the recorded SQL as a native question through the Metabase API, after a console `y/n` approval. Run Metabase locally with the pinned image: `docker run -d -p 127.0.0.1:3000:3000 --name metabase metabase/metabase:v0.65.2.6`. In Metabase add a PostgreSQL connection named `adventureworks` that uses the `analyst_reader` role (from inside Docker the host is usually `host.docker.internal`), create an API key, and put it in `.env` as `METABASE_API_KEY` (and `METABASE_URL` if not `http://localhost:3000`). The key is read from the environment, is never printed and is never sent through a redirect.

Why two tools: Apache Superset blocks SQLite connections by default and Metabase supports SQLite only through a community driver, so the SQLite dataset uses Datasette and the PostgreSQL dataset uses Metabase.

## WhatsApp front end (prototype)

The same agent answers over WhatsApp using the WhatsApp Business Cloud API: questions, numbered clarification options, the Decisions and verification text, and drawn charts sent back as images.

What happens in a chat:

- The first message asks which dataset to use (a numbered list); send `change dataset`, `/dataset`, or a dataset name to switch.
- Clarifications arrive as numbered text. Reply with a number, `0` and then your own answer, or ask a question. An unanswered clarification stops after 5 minutes.
- Ask for a chart in words ("show monthly orders in 2017 as a line chart") and the PNG arrives after the text answer.
- `help` lists the datasets, their tables and example questions without calling the model.
- Console-only commands (`/chart`, `/publish`, `/notes`) are not available.

Setup:

1. In Meta for Developers create an app with the WhatsApp product, a phone number (the free test number works), and a system-user access token. The WhatsApp account must be subscribed to your app (`POST /{waba-id}/subscribed_apps`); without that the webhook handshake succeeds but no messages arrive.
2. Fill in the `WHATSAPP_*` values in `.env` (see `.env.example`): token, phone number ID, app secret, verify token, API version, and `WHATSAPP_ALLOWED_NUMBERS` (digits only, with country code; messages from any other number are ignored). Leave `WHATSAPP_DATASET` empty to let the user pick, or set a dataset name to fix it.
3. Start the webhook server: `.venv/bin/python -m agent.whatsapp` (it listens on `127.0.0.1:8080`, path `/webhook`).
4. Expose it with a tunnel, for example `cloudflared tunnel --url http://127.0.0.1:8080` (use `127.0.0.1`, not `localhost`). Put `<tunnel address>/webhook` and your verify token in the WhatsApp Configuration page in Meta. A quick tunnel gets a new address every time it restarts.

The WhatsApp front end was built and tested on Meta's test number with an unpublished app. It is a prototype, not a production service (see known limits).

## Adding a database

Databases are registered in one place, `agent/datasets.py`: a name, a description, a label and a backend factory. A new database needs a backend class that implements `DatabaseBackend` (`agent/backend.py`), a read-only login for it, and one registry entry. The orchestrator contains no database names, and `--dataset` choices, banners, WhatsApp dataset lists and per-dataset memory files all derive from the registry.

## Model and cost

Default model: `muse-spark-1.3-contributor` on the Meta Model API (OpenAI-compatible Responses API, called through the open-source `openai` Python client). Set `META_MODEL` in `.env` to use another model.

**Contributor mode note:** in contributor mode Meta may use prompts and completions for model training. Use it only with public or synthetic data, as in this demo (Olist and AdventureWorks are public sample datasets). For real client data use a Standard-mode model.

Measured cost for one question is roughly 2–3 US cents at contributor rates (about 200k input tokens and 15–20k output tokens including all rounds). A cross-database question runs two agents plus a planner and a merge step.

Optional open-weight provider: set `MODEL_PROVIDER=baseten` in `.env` to run the open-weight model `meta-models/muse-spark-open-1026` through Baseten's OpenAI-compatible Chat Completions endpoint (`https://inference.baseten.co/v1`, key in `BASETEN_API_KEY`). Baseten reasoning output is never shown or used. Measured on one console question: about US$0.054 at list prices ($1.25 per 1M input tokens, $0.15 per 1M cached input, $4.25 per 1M output); in that test it asked more clarifications and wrote longer answers than the Meta default. The Meta default is the path used and verified for the demo; Baseten is a verified-working option, not the default. The contributor-mode note above applies to the Meta contributor model only.

## Model providers

Set `MODEL_PROVIDER` in `.env`. Each provider except Meta is one entry in `CHAT_PROVIDERS` (`agent/provider.py`) and shares one Chat Completions code path.

| `MODEL_PROVIDER` | Key variable | Model variable | Notes |
| --- | --- | --- | --- |
| `meta` (default) | `META_API_KEY` | `META_MODEL` | Original Responses API path |
| `baseten` | `BASETEN_API_KEY` | `BASETEN_MODEL` | Strict tool schemas, one-time fallback |
| `gemini` | `GEMINI_API_KEY` | `GEMINI_MODEL` | Google's OpenAI-compatible endpoint |
| `groq` | `GROQ_API_KEY` | `GROQ_MODEL` | |
| `openrouter` | `OPENROUTER_API_KEY` | `OPENROUTER_MODEL` | |
| `cerebras` | `CEREBRAS_API_KEY` | `CEREBRAS_MODEL` | |
| `ollama` | none | `OLLAMA_MODEL` | Local; `OLLAMA_BASE_URL` to override |
| `openai_compatible` | `OPENAI_COMPAT_API_KEY` | `OPENAI_COMPAT_MODEL` | Needs `OPENAI_COMPAT_BASE_URL` |

Not yet verified live: only Meta and Baseten have been used with a real key. The others are covered by tests with fakes. Default model names and free-tier limits change often, so check each provider's docs. HTTP 429, 502, 503 and 504 responses are retried up to 3 times, honouring `Retry-After` (capped at 30 s). **Free tiers often use prompts for training: send only public or synthetic data through them.**

To check a provider with your own key, run `MODEL_PROVIDER=gemini python scripts/provider_smoke.py`. It asks one question over a tiny synthetic database and checks the figures (3 orders, 450), so no real data is sent.

## BigQuery (optional)

```bash
pip install -r requirements-bigquery.txt
```

Set `BIGQUERY_DATASET=project.dataset` (and `BIGQUERY_PROJECT` to the project that is billed, which public datasets require to be your own), then run `python -m agent --dataset bigquery`. Use a service account with only BigQuery Data Viewer and Job User. Every query is dry-run first: anything but a `SELECT` is rejected, and queries estimated over `BIGQUERY_MAX_BYTES` (default 5 GB, a guess to tune) are refused; the real job carries the same cap. Not yet tried against live BigQuery.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q
```

The suite has 320 passing tests plus 10 that skip when their prerequisites are missing (a PostgreSQL server with the read-only role, and loopback networking). All provider, WhatsApp and BI tests run against fakes: no network, no API key.

## Known limits

- The agent cannot catch a choice that matters if the model never lists it. The instructions, the fixed key list (`revenue`, `top_metric`, `order_status`, `year_date`, `cost`, `period`) and the proof rules reduce this; they do not eliminate it.
- "Same filters across all options" and "ask the scope first" are instructions to the model, not rules enforced by code.
- The "(last used)" marker depends on the model's option wording and often won't appear.
- The save-as-default prompt is offered only for the last clarification of a turn.
- Answers cite verified numbers, but verification proves a number came from a query result, not that the query was the right one. The Decisions section exists so you can see which choices shaped the number.
- Clarifying questions vary between runs: the same question can get a different (but still verified) set of options.
- Descriptive questions with no number to verify ("what dataset is the sales data in?") only work when the agent can back the answer with a query result; otherwise the agent says it could not answer from the data.
- Charts are bar and line only, single dataset only (not `--dataset all`).
- A BI link or Metabase question points at a tool running on your own machine; it is not shared outside it. Chart images are not captured from Datasette or Metabase (Metabase's image endpoint returned 404 on the pinned version).
- WhatsApp: prototype on a test number with an unpublished app; one dataset per conversation (no cross-database questions); text and chart images only; a quick tunnel changes address on restart; a message sent while a clarification is waiting is taken as the answer; a model failure is retried several times before the agent gives up.
- The AdventureWorks download scripts track the upstream `master` branch rather than a pinned commit.
- A routing-only planner can misroute a vague cross-database question; each worker still reports what its database cannot answer under Gaps.

## Roadmap

- WhatsApp: reply buttons and list messages for clarifications, a production number and app review, cross-database questions.
- Open-weight Muse Spark through Baseten as a selectable model provider (done: `MODEL_PROVIDER=baseten`).
- Apache Superset for PostgreSQL sources, and image snapshots of BI charts for messaging.
- A chat-style web UI.
- Docker packaging.
- Dataset configuration file (`datasets.toml`) so databases can be added without editing Python.
- Stop after repeated identical tool errors.
- Scratch-copy tools so the agent can clean a copy of the data without ever touching the source.
- Cached results with freshness checks.

## Credits and licences

- **Models:** Meta Muse Spark via the Meta Model API.
- **Olist Brazilian E-Commerce Public Dataset** by Olist, via Kaggle, licensed CC BY-NC-SA 4.0 (non-commercial, share-alike). The data is not included in this repository; download it yourself. Used here only for a non-commercial hackathon demonstration.
- **AdventureWorks** sample database by Microsoft (`microsoft/sql-server-samples`, MIT licence). The PostgreSQL conversion scripts are from [lorint/AdventureWorks-for-Postgres](https://github.com/lorint/AdventureWorks-for-Postgres). Neither the data nor the scripts are included in this repository; `scripts/load_adventureworks.sh` downloads them.
- **Datasette** (Apache 2.0) and **Metabase** (open source, AGPL) are run as separate tools and are not bundled; Metabase is used through its API only.
- The idea of tools that refuse to accept what they cannot verify, and of a plan-first record, was inspired by the Meta `legal-agent-workshop`. This project's code is written independently.
