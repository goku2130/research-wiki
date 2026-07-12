#!/usr/bin/env python3
"""
Pedagogical primer generator: takes a FACT-CHECKED reference article (topics/<slug>.md)
and reorganizes it into a crisp, teachable primer (primer/<slug>.md). It REORGANIZES
already-verified prose — it must not introduce any fact/number/claim absent from the
reference — so the primer inherits the reference's grounding (no new fact-check needed).

Run:  python make_primer.py <slug> [slug2 ...]      (or no args = all comprehensive topics)
"""
import re
import sys
import time

import yaml
import agent

PRIMER_SYS = (
    "You are a brilliant technical teacher writing a CRISP, pedagogical primer for an "
    "expert-audience wiki. You are given a COMPREHENSIVE, already-fact-checked reference "
    "article. Your job is to REORGANIZE and DISTILL it into a lesson that teaches the "
    "topic fast and well — NOT to summarize mechanically.\n\n"
    "HARD CONSTRAINT: use ONLY facts, numbers, equations, and claims that appear in the "
    "reference. Introduce NOTHING new — no fresh statistics, papers, or assertions. You "
    "are re-teaching verified material, so the primer stays trustworthy by construction.\n\n"
    "STYLE (this is the whole point):\n"
    "- Open with a SCAFFOLD: one short paragraph on what this is, what the reader will "
    "understand by the end, and what it connects to.\n"
    "- Teach the CORE MECHANISM with intuition first, then the single most important "
    "equation (keep essential LaTeX with $...$ / $$...$$). Explain WHY it works, not just "
    "what it is. Anticipate the common confusion and pre-empt it.\n"
    "- Include ONE short runnable check (a ```python fenced block, ~8-15 lines) that makes "
    "the central mechanism concrete with asserts — only if the reference's math supports it.\n"
    "- Surface the 1-2 LOAD-BEARING disagreements or caveats (not all of them).\n"
    "- End with a one-line 'Current status' and a 'Full reference' pointer line.\n"
    "- DROP: exhaustive variant catalogs, hyperparameter dumps, tangential theory, long "
    "results tables. Those live in the reference. Aim for ~1000-1500 words of clean prose.\n"
    "- Do NOT use inline [source:<id>] tags — the primer reads clean; traceability is the "
    "linked reference. Output ONLY the primer markdown, starting with a '# ' title."
)


def make_primer(slug: str) -> str:
    p = agent.TOPICS_DIR / f"{slug}.md"
    if not p.exists():
        return f"{slug}: no reference"
    parts = p.read_text().split("---", 2)
    front = yaml.safe_load(parts[1]) or {}
    body = parts[2]                      # includes References; the model can see them but won't cite
    prompt = (f"Reference topic: {front.get('title', slug)}\n\n"
              f"REFERENCE ARTICLE (fact-checked; reorganize, do not add):\n\n{body}")
    t = time.time()
    agent.log(slug, "primer: writing")
    resp = agent._chat(agent.writer_client(), model=agent.WRITER_MODEL, temperature=0.5,
                       max_tokens=8000,
                       messages=[{"role": "system", "content": PRIMER_SYS},
                                 {"role": "user", "content": prompt}])
    primer = agent.strip_thoughts(agent.content_text(resp.choices[0].message)).strip()
    primer = agent.strip_preamble(agent.fix_latex(primer))
    out_dir = agent.ROOT / "primer"
    out_dir.mkdir(exist_ok=True)
    fm = {"title": front.get("title", slug), "kind": "primer",
          "reference": f"../topics/{slug}.md", "updated": time.strftime("%Y-%m-%d")}
    ref_line = f"\n\n---\n*Full reference (citations, derivations, variants):* [{front.get('title', slug)}](../topics/{slug}.md)\n"
    (out_dir / f"{slug}.md").write_text(agent.fm(fm) + primer + ref_line)
    words = len(primer.split())
    agent.log(slug, f"primer: {words}w ({time.time()-t:.0f}s)")
    return f"{slug}: primer {words}w"


def main() -> None:
    slugs = sys.argv[1:]
    if not slugs:
        tax = yaml.safe_load(agent.TAXONOMY.read_text())
        slugs = [t["slug"] for t in tax["topics"] if t.get("status") == "done"]
    for s in slugs:
        print(make_primer(s), flush=True)


if __name__ == "__main__":
    main()
