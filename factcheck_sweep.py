#!/usr/bin/env python3
"""
Fact-check remediation sweep: the review/fact-check gate was silently truncating on
large articles (thinking ate the token budget), so every comprehensive article passed
WITHOUT its grounding being verified. This re-runs the now-fixed fact-check (nemotron,
32k) + LaTeX validation over each existing article and revises ONLY what's flagged, in
preserve mode — no structural rewrite, no shrink, no new sources. Depth is preserved.

Only touches NVIDIA (nemotron) — no Google/throttle usage. Run: python factcheck_sweep.py [workers] [max]
"""
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import agent

WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
MAX_TOPICS = int(sys.argv[2]) if len(sys.argv) > 2 else 999
_lock = threading.Lock()


def sh(*args) -> None:
    subprocess.run(args, cwd=agent.ROOT, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sweep_one(topic: dict, tax: dict) -> str:
    slug = topic["slug"]
    p = agent.TOPICS_DIR / f"{slug}.md"
    if not p.exists():
        return f"{slug}: missing"
    parts = p.read_text().split("---", 2)
    front = yaml.safe_load(parts[1]) or {}
    body = parts[2].split("## References")[0].strip()
    open_qs = front.get("open_questions") or []
    cited = front.get("sources") or []
    distilled = [d for sid in cited if (d := agent._load_source(sid))]
    if not distilled:
        return f"{slug}: no sources loaded"
    t0 = time.time()
    agent.log(slug, f"SWEEP start ({len(distilled)} sources)")
    try:
        article, new_qs, meta = agent.factcheck_article(topic, body, open_qs, distilled, tax)
    except Exception as e:  # noqa: BLE001
        agent.log(slug, f"FAILED ({str(e)[:80]})")
        return f"{slug}: FAILED ({str(e)[:60]})"
    g = sum(r["grounding"] for r in meta["rounds"])
    lx = sum(r["latex"] for r in meta["rounds"])
    if article.strip() == body.strip():
        agent.log(slug, f"clean — no changes ({time.time()-t0:.0f}s)")
        return f"{slug}: clean"
    agent.save(topic, distilled, article, new_qs,
               maturity=front.get("maturity", "comprehensive"))   # preserve maturity
    agent.log(slug, f"REVISED — {g} grounding, {lx} latex fixed ({time.time()-t0:.0f}s)")
    with _lock:
        sh("git", "add", "-A")
        sh("git", "commit", "-m", f"factcheck: {slug} ({g} grounding, {lx} latex fixed)")
        sh("git", "push", "origin", "main")
    return f"{slug}: revised ({g}g {lx}l)"


def main() -> None:
    tax = yaml.safe_load(agent.TAXONOMY.read_text())
    done = [t for t in tax["topics"] if t.get("status") == "done"][:MAX_TOPICS]
    print(f"### FACT-CHECK SWEEP: {len(done)} topics, {WORKERS} concurrent")
    print(f"### fact-checker = {agent.FACTCHECK_MODEL} (NVIDIA; no Google quota used)", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(sweep_one, t, tax): t["slug"] for t in done}
        for i, f in enumerate(as_completed(futs), 1):
            print(f"  [{i}/{len(done)}] {f.result()}  ({time.time()-t0:.0f}s elapsed)", flush=True)
    print(f"### sweep done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
