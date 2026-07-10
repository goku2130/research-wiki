# research-wiki

A fully-local, agent-built knowledge base — your own [rl-llm-wiki](https://huggingface.co/rl-llm-wiki).

## Architecture

```
taxonomy.yaml (your topics)
      │
      ▼
agent.py ──► Qwen 3.6 35B-A3B  (RTX 5090, :8181)  orchestrator + tool calls
      │           │ search_web / read_url
      │           ▼
      │      SearXNG (:8888)  self-hosted metasearch
      │
      └──► Gemma 4 31B  (RTX 3090, :8182)  writes the cited article
             │
             ▼
      topics/*.md  +  sources/*.md   (git-committed wiki)
```

## Setup (once)

```bash
# python deps
uv venv && uv pip install -r requirements.txt

# start the model servers + search (first run pulls the searxng image)
docker compose up -d
docker compose logs -f qwen      # wait for "server listening"
```

## Run a cycle

```bash
.venv/bin/python agent.py            # fill the least-covered topic
.venv/bin/python agent.py grpo       # force a specific topic slug
```

Each cycle: picks a topic → researches via Qwen+SearXNG → Gemma writes it →
saves `topics/<slug>.md` + `sources/*.md` → updates `taxonomy.yaml` + this index → you `git commit`.

## Make it autonomous

Loop it with cron, or ask Claude Code to run `/loop` / `/schedule` on `agent.py`.

## Edit your interests

Everything is driven by **`taxonomy.yaml`** — add the topics you care about there.
```
