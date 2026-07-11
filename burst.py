#!/usr/bin/env python3
"""
Parallel backlog burst: generate N topics concurrently to raise throughput.
Uses both Mistral keys (rotated) for the concurrent writers and local Qwen
(--parallel 4) for orchestrator/distill/review. File writes are per-topic; the
taxonomy update + git commit/push are serialized under a lock (no races).

Run:  python burst.py [workers] [max_topics]
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


def generate(topic: dict, tax: dict) -> str:
    slug = topic["slug"]
    try:
        distilled = agent.gather_and_distill(topic)
        if not distilled:
            return f"{slug}: no sources"
        article, open_qs, _ = agent.compose(topic, distilled, tax, verbose=False)
        agent.save(topic, distilled, article, open_qs)
    except Exception as e:  # noqa: BLE001
        return f"{slug}: FAILED ({str(e)[:80]})"
    # serialize taxonomy update + commit + push
    with _lock:
        tax = yaml.safe_load(agent.TAXONOMY.read_text())
        for t in tax["topics"]:
            if t["slug"] == slug:
                t["status"] = "done"
        agent.TAXONOMY.write_text(yaml.safe_dump(tax, sort_keys=False, allow_unicode=True))
        agent.regenerate_index(tax)
        sh("git", "add", "-A")
        sh("git", "commit", "-m", f"auto: {slug} ({time.strftime('%FT%TZ', time.gmtime())})")
        sh("git", "push", "origin", "main")
    return f"{slug}: ok ({len(distilled)} sources, {len(article.split())}w)"


def main() -> None:
    tax = yaml.safe_load(agent.TAXONOMY.read_text())   # read-only snapshot for siblings
    todo = [t for t in tax["topics"] if t.get("status") != "done"][:MAX_TOPICS]
    print(f"### BURST: {len(todo)} topics, {WORKERS} concurrent, writer={agent.WRITER_MODEL}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(generate, t, tax): t["slug"] for t in todo}
        for i, f in enumerate(as_completed(futs), 1):
            print(f"  [{i}/{len(todo)}] {f.result()}  ({time.time()-t0:.0f}s elapsed)")
    print(f"### done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
