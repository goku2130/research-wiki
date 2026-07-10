#!/usr/bin/env python3
"""
Controlled A/B: Gemini-centric vs fully-local composition, on IDENTICAL inputs.

Control: gather+distill runs ONCE (local) and is frozen to a fixture; both arms
consume the same distilled sources + same prompts. Only the writer/reviewer
backend varies.

  local  : writer=local Gemma (3090),  reviewer/fact-check=local Qwen (5090)
  gemini : writer=Gemma 4 31B (API),   reviewer/fact-check=Gemini 3.1 Flash Lite (API)

Each arm runs N times (stochasticity). A blind, randomized neutral judge
(gemini-flash-latest) scores the two articles. Objective metrics reported too.

Run:  python experiment.py <slug> [runs]
"""
import json
import random
import re
import sys
import time
from pathlib import Path

import yaml
import agent

EXPDIR = agent.ROOT / "experiments"
RUNS_DEFAULT = 2

# 3-way WRITER comparison: only the writer varies; the reviewer/fact-checker is
# held CONSTANT (local Qwen) across all arms so the writer is the sole variable.
ARMS = {
    "local":   lambda: (agent.GEMMA, "local"),
    "gemma4":  lambda: (agent.google_client(), "gemma-4-31b-it"),
    "mistral": lambda: (agent.mistral_client(), "mistral-large-latest"),
}
CONST_REVIEWER = lambda: (agent.QWEN, "local")
JUDGE_MODEL = "gemini-flash-latest"


def set_backend(arm: str) -> None:
    agent.WRITER, agent.WRITER_MODEL = ARMS[arm]()
    agent.REVIEWER, agent.REVIEWER_MODEL = CONST_REVIEWER()  # constant across arms


def build_fixture(topic: dict) -> list[dict]:
    """Freeze gather+distill ONCE (local Qwen + OCR) so all arms get identical inputs."""
    fx = EXPDIR / topic["slug"] / "fixture.json"
    if fx.exists():
        print(f"== using cached fixture: {fx}")
        return json.loads(fx.read_text())["distilled"]
    print("== building fixture (gather + OCR distill, once)...")
    set_backend("local")  # distillation uses local Qwen for all arms
    distilled = agent.gather_and_distill(topic)
    fx.parent.mkdir(parents=True, exist_ok=True)
    fx.write_text(json.dumps({"topic": topic, "distilled": distilled}, indent=2))
    print(f"== froze {len(distilled)} sources -> {fx}")
    return distilled


def metrics(article: str) -> dict:
    return {"words": len(article.split()),
            "citations": len(re.findall(r"\[source:", article))}


def run_arm(arm: str, topic: dict, distilled: list[dict], tax: dict, runs: int) -> dict:
    set_backend(arm)
    print(f"\n== arm '{arm}': writer={agent.WRITER_MODEL}, reviewer={agent.REVIEWER_MODEL}")
    out = []
    for i in range(runs):
        t = time.time()
        article, oq, meta = agent.compose(topic, distilled, tax, verbose=False)
        m = metrics(article)
        m.update(seconds=round(time.time() - t), open_qs=len(oq),
                 rounds=len(meta["rounds"]), article=article)
        out.append(m)
        print(f"   run {i+1}: {m['seconds']:>4}s  {m['words']:>4}w  "
              f"{m['citations']:>2} cites  {m['rounds']} rounds  {m['open_qs']} openQ")
    (EXPDIR / topic["slug"] / f"{arm}.md").write_text(out[0]["article"])
    return {"runs": out,
            "avg_seconds": round(sum(r["seconds"] for r in out) / runs),
            "avg_words": round(sum(r["words"] for r in out) / runs),
            "avg_citations": round(sum(r["citations"] for r in out) / runs)}


JUDGE_SYS = (
    "You are an impartial expert judge of technical wiki articles on the SAME topic, "
    "written from the SAME sources. Score each 1-10 on depth/precision, citation "
    "grounding, surfaced disagreement, and clarity. Reason briefly, then on the last "
    "lines output 'SCORES: 1=<n> 2=<n> 3=<n>' and 'RANKING: <best>,<mid>,<worst>' "
    "using the article numbers."
)


def judge(topic: dict, arm_articles: dict) -> dict:
    # blind + randomized: number the articles, hide which arm is which
    items = list(arm_articles.items())          # [(arm, article), ...]
    random.shuffle(items)
    labels = {str(i + 1): arm for i, (arm, _) in enumerate(items)}
    blocks = "\n\n".join(f"=== ARTICLE {i+1} ===\n{art}" for i, (_, art) in enumerate(items))
    r = agent.google_client().chat.completions.create(
        model=JUDGE_MODEL, temperature=0.2, max_tokens=4000,
        messages=[{"role": "system", "content": JUDGE_SYS},
                  {"role": "user", "content": f"Topic: {topic['title']}\n\n{blocks}"}])
    out = agent.strip_thoughts(r.choices[0].message.content or "")
    rank = re.search(r"RANKING:\s*([\d,\s]+)", out, re.I)
    order = [labels.get(n.strip(), "?") for n in rank.group(1).split(",")] if rank else []
    scores = {labels[m.group(1)]: float(m.group(2))
              for m in re.finditer(r"([123])\s*=\s*(\d+(?:\.\d+)?)", out)
              if m.group(1) in labels}
    return {"ranking": order, "scores": scores, "blind_labels": labels, "raw": out[-700:]}


def main() -> None:
    slug = sys.argv[1] if len(sys.argv) > 1 else "rlvr-3way"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else RUNS_DEFAULT
    tax = yaml.safe_load(agent.TAXONOMY.read_text())
    topic = next((t for t in tax["topics"] if t["slug"] == slug),
                 {"slug": slug, "title": "Reinforcement Learning with Verifiable Rewards",
                  "notes": "RLVR, verifiable rewards, reasoning"})
    print(f"### 3-WAY WRITER A/B: {topic['slug']} — {runs} runs/arm "
          f"(reviewer held constant = local Qwen)")

    distilled = build_fixture(topic)
    if not distilled:
        print("fixture empty; aborting"); return

    results = {arm: run_arm(arm, topic, distilled, tax, runs) for arm in ARMS}
    verdict = judge(topic, {arm: results[arm]["runs"][0]["article"] for arm in ARMS})

    print("\n" + "=" * 66 + f"\n### RESULTS (identical OCR inputs, {len(distilled)} "
          f"frozen sources)\n" + "=" * 66)
    print(f"{'metric':<14}" + "".join(f"{a:>14}" for a in ARMS))
    for k, label in [("avg_seconds", "speed (s)"), ("avg_words", "words"),
                     ("avg_citations", "citations")]:
        print(f"{label:<14}" + "".join(f"{results[a][k]:>14}" for a in ARMS))
    print(f"{'judge score':<14}" + "".join(
        f"{verdict['scores'].get(a,'-'):>14}" for a in ARMS))
    print(f"\nblind judge ({JUDGE_MODEL}) ranking (best->worst): "
          f"{' > '.join(verdict['ranking'])}")
    print("rationale (tail):\n" + verdict["raw"])

    (EXPDIR / topic["slug"] / "results.json").write_text(json.dumps(
        {"topic": topic, "n_sources": len(distilled), "results": results,
         "judge": verdict}, indent=2, default=str))
    print(f"\nsaved -> experiments/{topic['slug']}/")


if __name__ == "__main__":
    main()
