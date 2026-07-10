#!/usr/bin/env python3
"""
One research -> write -> commit cycle for the local wiki.

  Qwen (orchestrator, :8181)  researches a topic using search_web + read_url tools,
  then hands vetted sources to
  Gemma (writer, :8182)       which drafts a cited article.

Run:  python agent.py            # fills the least-covered topic
      python agent.py <slug>     # force a specific topic
"""
import json
import os
import re
import sys
import time
import datetime as dt
from pathlib import Path

import requests
import yaml
import trafilatura
from openai import OpenAI

ROOT = Path(__file__).parent
TAXONOMY = ROOT / "taxonomy.yaml"
TOPICS_DIR = ROOT / "topics"
SOURCES_DIR = ROOT / "sources"

QWEN = OpenAI(base_url=os.getenv("QWEN_URL", "http://localhost:8181/v1"), api_key="local")
GEMMA = OpenAI(base_url=os.getenv("GEMMA_URL", "http://localhost:8182/v1"), api_key="local")
SEARXNG = os.getenv("SEARXNG_URL", "http://localhost:8888")

MAX_TOOL_STEPS = 16          # cap orchestrator tool calls per cycle
PAGE_CHARS = 6000            # truncate fetched pages to keep context sane


# ----------------------------------------------------------------- tools
def search_web(query: str, max_results: int = 6) -> list[dict]:
    r = requests.get(
        f"{SEARXNG}/search",
        params={"q": query, "format": "json", "categories": "general"},
        timeout=30,
    )
    r.raise_for_status()
    out = []
    for item in r.json().get("results", [])[:max_results]:
        out.append({"title": item.get("title"), "url": item.get("url"),
                    "snippet": (item.get("content") or "")[:300]})
    return out


def read_url(url: str) -> str:
    try:
        html = requests.get(url, timeout=30, headers={"User-Agent": "research-wiki/1.0"}).text
        text = trafilatura.extract(html, include_comments=False) or ""
    except Exception as e:  # noqa: BLE001
        return f"[fetch failed: {e}]"
    return text[:PAGE_CHARS] if text else "[no extractable text]"


TOOLS = [
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the web. Returns a list of {title, url, snippet}.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 6}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_url",
        "description": "Fetch a URL and return cleaned main text.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "submit_research",
        "description": "Call ONCE when finished, with the vetted sources you will cite.",
        "parameters": {"type": "object", "properties": {
            "sources": {"type": "array", "items": {"type": "object", "properties": {
                "title": {"type": "string"},
                "url": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}}},
                "required": ["title", "url", "key_points"]}}},
            "required": ["sources"]}}},
]

DISPATCH = {"search_web": search_web, "read_url": read_url}
SUBMIT_TOOL = TOOLS[-1]   # submit_research


def _read_urls(messages: list) -> list[str]:
    """Pull every URL the orchestrator actually fetched via read_url."""
    urls = []
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            if tc.get("function", {}).get("name") == "read_url":
                try:
                    u = json.loads(tc["function"].get("arguments") or "{}").get("url")
                    if u and u not in urls:
                        urls.append(u)
                except Exception:  # noqa: BLE001
                    pass
    return urls


def force_submit(messages: list) -> list[dict]:
    """Local models keep researching forever; make them wrap up on demand."""
    read = _read_urls(messages)
    read_list = "\n".join(f"- {u}" for u in read) or "(none)"
    messages = messages + [{"role": "user", "content":
        "Stop searching. You already read these pages:\n" + read_list +
        "\nCall submit_research now. Include EACH distinct page above that was useful "
        "(aim for 4-6 sources), each with concrete key_points drawn from what you read."}]
    try:
        resp = QWEN.chat.completions.create(
            model="local", messages=messages, tools=[SUBMIT_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_research"}},
            temperature=0.4)
        tcs = resp.choices[0].message.tool_calls or []
        if tcs:
            return json.loads(tcs[0].function.arguments or "{}").get("sources", [])
    except Exception as e:  # noqa: BLE001
        print(f"  force_submit failed: {e}")
    return []


# ----------------------------------------------------------------- orchestrator
def research(topic: dict) -> list[dict]:
    sys_msg = (
        "You are a research agent building an expert, citation-backed wiki. "
        "Use search_web and read_url to gather accurate, primary information about the "
        "given topic. Prefer papers and authoritative sources. Read at least 4 distinct "
        "pages. Do NOT repeat a search you already ran or re-read a page. As soon as you "
        "have read 4-5 good pages, call submit_research with those 4-6 sources and "
        "concrete key_points (specific facts/equations/findings, not vague summaries)."
    )
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"Topic: {topic['title']}\nScope: {topic.get('notes','')}"},
    ]
    for step in range(MAX_TOOL_STEPS):
        resp = QWEN.chat.completions.create(
            model="local", messages=messages, tools=TOOLS, temperature=0.6)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            messages.append({"role": "user",
                             "content": "Continue researching, then call submit_research."})
            continue
        messages.append(msg.model_dump(exclude_none=True))
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            if name == "submit_research":
                print(f"  submit_research: {len(args.get('sources', []))} sources")
                return args.get("sources", [])
            print(f"  [{step}] {name}({str(args)[:70]})")
            try:
                result = DISPATCH[name](**args)
            except Exception as e:  # noqa: BLE001
                result = f"[tool error: {e}]"
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result)[:PAGE_CHARS]})
    print("  ! hit MAX_TOOL_STEPS -> forcing submit_research")
    return force_submit(messages)


# ----------------------------------------------------------------- writer
def write_article(topic: dict, sources: list[dict]) -> str:
    refs = "\n".join(
        f"[S{i+1}] {s['title']} — {s['url']}\n" +
        "\n".join(f"    - {p}" for p in s.get("key_points", []))
        for i, s in enumerate(sources))
    prompt = (
        f"Write an expert-level wiki article on: {topic['title']}\n"
        f"Scope: {topic.get('notes','')}\n\n"
        "Use ONLY the sources below. Cite inline like [S1], [S2] wherever a claim comes "
        "from a source. Structure: a 2-sentence intro, then clear ## sections, then a "
        "short 'Key takeaways' list. Be precise and technical. Do not invent citations.\n\n"
        f"SOURCES:\n{refs}"
    )
    resp = GEMMA.chat.completions.create(
        model="local",
        messages=[{"role": "system", "content": "You are a precise technical writer."},
                  {"role": "user", "content": prompt}],
        temperature=0.7)
    return resp.choices[0].message.content


# ----------------------------------------------------------------- persistence
def fm(d: dict) -> str:
    return "---\n" + yaml.safe_dump(d, sort_keys=False).strip() + "\n---\n\n"


def save(topic: dict, sources: list[dict], article: str) -> None:
    SOURCES_DIR.mkdir(exist_ok=True)
    TOPICS_DIR.mkdir(exist_ok=True)
    today = dt.date.today().isoformat()

    for i, s in enumerate(sources):
        slug = re.sub(r"[^a-z0-9]+", "-", s["title"].lower()).strip("-")[:60] or f"source-{i}"
        (SOURCES_DIR / f"{slug}.md").write_text(
            fm({"title": s["title"], "url": s["url"], "retrieved": today,
                "topic": topic["slug"]}) +
            "\n".join(f"- {p}" for p in s.get("key_points", [])) + "\n")

    ref_list = "\n".join(f"- [S{i+1}] [{s['title']}]({s['url']})"
                         for i, s in enumerate(sources))
    (TOPICS_DIR / f"{topic['slug']}.md").write_text(
        fm({"title": topic["title"], "updated": today,
            "sources": [s["url"] for s in sources]}) +
        article.strip() + "\n\n## References\n" + ref_list + "\n")


def regenerate_index(tax: dict) -> None:
    lines = [f"# {tax['domain']} — Wiki\n",
             f"_Auto-generated {dt.date.today().isoformat()}._\n"]
    for t in tax["topics"]:
        mark = {"done": "✅", "draft": "📝", "empty": "⬜"}.get(t["status"], "⬜")
        link = f"[{t['title']}](topics/{t['slug']}.md)" if t["status"] != "empty" else t["title"]
        lines.append(f"- {mark} {link}")
    (ROOT / "README.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------- main
def pick_topic(tax: dict, forced: str | None):
    if forced:
        return next(t for t in tax["topics"] if t["slug"] == forced)
    order = {"empty": 0, "draft": 1, "done": 2}
    return min(tax["topics"], key=lambda t: order.get(t["status"], 3))


def main() -> None:
    tax = yaml.safe_load(TAXONOMY.read_text())
    forced = sys.argv[1] if len(sys.argv) > 1 else None
    topic = pick_topic(tax, forced)
    print(f"== topic: {topic['slug']} ({topic['title']})")

    t0 = time.time()
    sources = research(topic)
    if not sources:
        print("no sources gathered; aborting"); return
    print(f"== writing article from {len(sources)} sources...")
    article = write_article(topic, sources)
    save(topic, sources, article)

    topic["status"] = "done"
    TAXONOMY.write_text(yaml.safe_dump(tax, sort_keys=False))
    regenerate_index(tax)
    print(f"== done in {time.time()-t0:.0f}s -> topics/{topic['slug']}.md")


if __name__ == "__main__":
    main()
