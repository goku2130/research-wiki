#!/usr/bin/env python3
"""
Enrichment burst: re-visit every DONE topic, gather NEW (uncited) sources, DEEPEN
the existing article (never overwrite), and advance maturity developing->comprehensive.
Anti-duplication is handled inside agent.enrich(): sources are filtered by normalized
source_id before AND after distillation, and the deepen prompt preserves existing text.

Run:  python enrich_all.py [workers] [max_topics]
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


def enrich_one(topic: dict, tax: dict) -> str:
    slug = topic["slug"]
    t0 = time.time()
    agent.log(slug, "ENRICH start")
    try:
        changed = agent.enrich(topic, tax)
    except Exception as e:  # noqa: BLE001
        agent.log(slug, f"FAILED ({str(e)[:80]})")
        return f"{slug}: FAILED ({str(e)[:80]})"
    if not changed:
        agent.log(slug, f"no new sources ({time.time()-t0:.0f}s)")
        return f"{slug}: unchanged (already current)"
    agent.log(slug, f"ENRICHED ({time.time()-t0:.0f}s)")
    # serialize index regen + commit + push (enrich() already wrote the topic file)
    with _lock:
        fresh = yaml.safe_load(agent.TAXONOMY.read_text())
        agent.regenerate_index(fresh)
        sh("git", "add", "-A")
        sh("git", "commit", "-m", f"enrich: {slug} ({time.strftime('%FT%TZ', time.gmtime())})")
        sh("git", "push", "origin", "main")
    return f"{slug}: enriched"


def main() -> None:
    tax = yaml.safe_load(agent.TAXONOMY.read_text())
    done = [t for t in tax["topics"] if t.get("status") == "done"][:MAX_TOPICS]
    print(f"### ENRICH: {len(done)} done topics, {WORKERS} concurrent")
    print(f"### models: orch={agent.ORCH_MODEL} | distill={agent.DISTILL_MODEL} "
          f"| review={agent.REVIEWER_MODEL} | writer={agent.WRITER_MODEL}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(enrich_one, t, tax): t["slug"] for t in done}
        for i, f in enumerate(as_completed(futs), 1):
            print(f"  [{i}/{len(done)}] {f.result()}  ({time.time()-t0:.0f}s elapsed)", flush=True)
    print(f"### enrichment done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
