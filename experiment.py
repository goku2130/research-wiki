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

ARMS = {
    "local":  {"writer": lambda: (agent.GEMMA, "local"),
               "reviewer": lambda: (agent.QWEN, "local")},
    "gemini": {"writer": lambda: (agent.google_client(), "gemma-4-31b-it"),
               "reviewer": lambda: (agent.google_client(), "gemini-flash-lite-latest")},
}
JUDGE_MODEL = "gemini-flash-latest"


def set_backend(arm: str) -> None:
    agent.WRITER, agent.WRITER_MODEL = ARMS[arm]["writer"]()
    agent.REVIEWER, agent.REVIEWER_MODEL = ARMS[arm]["reviewer"]()


def build_fixture(topic: dict) -> list[dict]:
    """Freeze gather+distill ONCE (local) so both arms get identical inputs."""
    fx = EXPDIR / topic["slug"] / "fixture.json"
    if fx.exists():
        print(f"== using cached fixture: {fx}")
        return json.loads(fx.read_text())["distilled"]
    print("== building fixture (gather + distill, once, local)...")
    set_backend("local")  # distillation uses local Qwen for both arms
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
    "You are an impartial expert judge of technical wiki articles. Score each article "
    "1-10 on: depth/precision, citation grounding, surfaced disagreement, and clarity. "
    "Reason briefly, then on the last lines output 'SCORES: A=<n> B=<n>' and "
    "'WINNER: A' or 'WINNER: B' or 'WINNER: TIE'."
)


def judge(topic: dict, local_art: str, gemini_art: str) -> dict:
    # blind + randomized: hide which arm is which
    pair = [("local", local_art), ("gemini", gemini_art)]
    random.shuffle(pair)
    labels = {"A": pair[0][0], "B": pair[1][0]}
    content = (f"Topic: {topic['title']}\n\n=== ARTICLE A ===\n{pair[0][1]}\n\n"
               f"=== ARTICLE B ===\n{pair[1][1]}")
    r = agent.google_client().chat.completions.create(
        model=JUDGE_MODEL, temperature=0.2, max_tokens=4000,
        messages=[{"role": "system", "content": JUDGE_SYS},
                  {"role": "user", "content": content}])
    out = agent.strip_thoughts(r.choices[0].message.content or "")
    win = re.search(r"WINNER:\s*([AB]|TIE)", out, re.I)
    winner = win.group(1).upper() if win else "?"
    return {"winner_arm": labels.get(winner, "tie/unknown"),
            "blind_labels": labels, "raw": out[-600:]}


def main() -> None:
    slug = sys.argv[1] if len(sys.argv) > 1 else "rlvr-overview-exp"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else RUNS_DEFAULT
    tax = yaml.safe_load(agent.TAXONOMY.read_text())
    topic = next((t for t in tax["topics"] if t["slug"] == slug),
                 {"slug": slug, "title": "Reinforcement Learning with Verifiable Rewards",
                  "notes": "RLVR, verifiable rewards, reasoning"})
    print(f"### EXPERIMENT: {topic['slug']} — {runs} runs/arm")

    distilled = build_fixture(topic)
    if not distilled:
        print("fixture empty; aborting"); return

    results = {arm: run_arm(arm, topic, distilled, tax, runs) for arm in ARMS}
    verdict = judge(topic, results["local"]["runs"][0]["article"],
                    results["gemini"]["runs"][0]["article"])

    print("\n" + "=" * 60 + "\n### RESULTS (identical inputs, "
          f"{len(distilled)} frozen sources)\n" + "=" * 60)
    print(f"{'metric':<16}{'local':>14}{'gemini':>14}")
    for k, label in [("avg_seconds", "speed (s)"), ("avg_words", "words"),
                     ("avg_citations", "citations")]:
        print(f"{label:<16}{results['local'][k]:>14}{results['gemini'][k]:>14}")
    print(f"\nblind judge ({JUDGE_MODEL}) winner: {verdict['winner_arm']}  "
          f"(A/B were {verdict['blind_labels']})")
    print("judge rationale (tail):\n" + verdict["raw"])

    (EXPDIR / topic["slug"] / "results.json").write_text(json.dumps(
        {"topic": topic, "n_sources": len(distilled), "results": results,
         "judge": verdict}, indent=2, default=str))
    print(f"\nsaved -> experiments/{topic['slug']}/  (local.md, gemini.md, results.json)")


if __name__ == "__main__":
    main()
