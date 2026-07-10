#!/usr/bin/env python3
"""
Two-stage research cycle for the local wiki (mirrors the rl-llm-wiki method).

  Stage 1  Qwen (orchestrator, :8181)  gathers 6-8 vetted sources for a topic.
  Stage 2  Qwen distills EACH source (full text, arXiv HTML preferred) into a
           deep, faithful sources/<id>.md summary (formulas, numbers, method).
  Stage 3  Gemma (writer, :8182) synthesizes the article from those summaries,
           under the rl-llm-wiki rubric: inline [source:<id>] citations, a
           current-status+trajectory section, hedged claims, surfaced
           disagreement, and open_questions.

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
from urllib.parse import urlparse

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

MAX_TOOL_STEPS = 16      # cap orchestrator tool calls in stage 1
SCAN_CHARS = 5000        # truncation while the orchestrator is scanning pages
FULLTEXT_CHARS = 24000   # much larger read for the distillation stage
TARGET_SOURCES = 8       # how many sources to aim for


# ----------------------------------------------------------------- fetching
def arxiv_id(url: str) -> str | None:
    m = re.search(r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5})", url) \
        or re.search(r"ar5iv[^/]*/html/([0-9]{4}\.[0-9]{4,5})", url)
    return m.group(1) if m else None


def _extract(url: str, cap: int) -> str:
    try:
        html = requests.get(url, timeout=40,
                            headers={"User-Agent": "research-wiki/1.0"}).text
        text = trafilatura.extract(html, include_comments=False,
                                   include_tables=True) or ""
    except Exception as e:  # noqa: BLE001
        return f"[fetch failed: {e}]"
    return text[:cap] if text else ""


def fetch_fulltext(url: str) -> str:
    """Full source text; for arXiv prefer HTML (real LaTeX) over the PDF/abs page."""
    aid = arxiv_id(url)
    if aid:
        for cand in (f"https://arxiv.org/html/{aid}",
                     f"https://ar5iv.labs.arxiv.org/html/{aid}"):
            t = _extract(cand, FULLTEXT_CHARS)
            if len(t) > 1500:
                return t
    return _extract(url, FULLTEXT_CHARS) or "[no extractable text]"


def source_id(url: str, title: str) -> str:
    aid = arxiv_id(url)
    if aid:
        return f"arxiv:{aid}"
    host = urlparse(url).netloc.replace("www.", "").split(".")[0] or "web"
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "source"
    return f"{host}:{slug}"


def id_to_filename(sid: str) -> str:
    return sid.replace(":", "-") + ".md"


# ----------------------------------------------------------------- stage 1: gather
def search_web(query: str, max_results: int = 6) -> list[dict]:
    r = requests.get(f"{SEARXNG}/search",
                     params={"q": query, "format": "json", "categories": "general"},
                     timeout=30)
    r.raise_for_status()
    return [{"title": i.get("title"), "url": i.get("url"),
             "snippet": (i.get("content") or "")[:300]}
            for i in r.json().get("results", [])[:max_results]]


def read_url(url: str) -> str:
    return _extract(url, SCAN_CHARS) or "[no extractable text]"


TOOLS = [
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the web. Returns [{title, url, snippet}].",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 6}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_url",
        "description": "Fetch a URL and return cleaned main text (skim).",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "submit_sources",
        "description": "Call ONCE when done, with the primary sources to distill & cite.",
        "parameters": {"type": "object", "properties": {
            "sources": {"type": "array", "items": {"type": "object", "properties": {
                "title": {"type": "string"},
                "url": {"type": "string"}},
                "required": ["title", "url"]}}},
            "required": ["sources"]}}},
]
DISPATCH = {"search_web": search_web, "read_url": read_url}
SUBMIT_TOOL = TOOLS[-1]


def _read_urls(messages: list) -> list[str]:
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
    read = _read_urls(messages)
    msg = ("Stop searching. You read these pages:\n" +
           ("\n".join(f"- {u}" for u in read) or "(none)") +
           f"\nCall submit_sources now with the {TARGET_SOURCES} most authoritative "
           "(prefer arXiv papers over blogs).")
    try:
        resp = QWEN.chat.completions.create(
            model="local", messages=messages + [{"role": "user", "content": msg}],
            tools=[SUBMIT_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_sources"}},
            temperature=0.4)
        tcs = resp.choices[0].message.tool_calls or []
        if tcs:
            return json.loads(tcs[0].function.arguments or "{}").get("sources", [])
    except Exception as e:  # noqa: BLE001
        print(f"  force_submit failed: {e}")
    return []


def gather_sources(topic: dict) -> list[dict]:
    sys_msg = (
        "You are a research agent for an expert, citation-backed wiki. Use search_web "
        "and read_url to find the PRIMARY sources on the topic — prefer original papers "
        "(arXiv) over blogs, and include the seminal ones plus recent work that agrees "
        f"or DISAGREES. Do not repeat searches. Aim for {TARGET_SOURCES} sources. When "
        "you have them, call submit_sources with their titles and URLs."
    )
    messages = [{"role": "system", "content": sys_msg},
                {"role": "user",
                 "content": f"Topic: {topic['title']}\nScope: {topic.get('notes','')}"}]
    for step in range(MAX_TOOL_STEPS):
        resp = QWEN.chat.completions.create(
            model="local", messages=messages, tools=TOOLS, temperature=0.6)
        m = resp.choices[0].message
        if not m.tool_calls:
            messages.append({"role": "user", "content": "Continue, then submit_sources."})
            continue
        messages.append(m.model_dump(exclude_none=True))
        for tc in m.tool_calls:
            name, args = tc.function.name, json.loads(tc.function.arguments or "{}")
            if name == "submit_sources":
                srcs = args.get("sources", [])
                print(f"  submit_sources: {len(srcs)} sources")
                return srcs
            print(f"  [{step}] {name}({str(args)[:70]})")
            try:
                result = DISPATCH[name](**args)
            except Exception as e:  # noqa: BLE001
                result = f"[tool error: {e}]"
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result)[:SCAN_CHARS]})
    print("  ! hit MAX_TOOL_STEPS -> forcing submit_sources")
    return force_submit(messages)


# ----------------------------------------------------------------- stage 2: distill
DISTILL_SYS = (
    "You distill ONE source into a faithful, thorough summary for a citation-backed "
    "research wiki. Capture, precisely and technically: the core problem; the "
    "method/recipe step by step; the key formulas in LaTeX; the key quantitative "
    "results and numbers; and the stated limitations. 400-800 words of markdown prose. "
    "Use ONLY what the source says — never invent results or numbers."
)


def distill(topic: dict, src: dict) -> dict | None:
    text = fetch_fulltext(src["url"])
    if len(text) < 800:
        print(f"    skip (thin fetch): {src['url']}")
        return None
    sid = source_id(src["url"], src["title"])
    resp = QWEN.chat.completions.create(
        model="local", temperature=0.3,
        messages=[{"role": "system", "content": DISTILL_SYS},
                  {"role": "user", "content":
                   f"Source: {src['title']}\nURL: {src['url']}\n\nTEXT:\n{text}"}])
    summary = resp.choices[0].message.content.strip()
    print(f"    distilled [{sid}] ({len(summary)} chars)")
    return {"id": sid, "title": src["title"], "url": src["url"], "summary": summary}


# ----------------------------------------------------------------- stage 3: write
WRITE_SYS = "You are a rigorous technical writer for an expert research wiki."
WRITE_RUBRIC = (
    "Write an expert-level deep dive — enough that an expert could learn the topic "
    "WITHOUT reading the papers. Hard rules (enforced):\n"
    "1. Cite EVERY non-obvious claim inline, exactly as [source:<id>], using the ids "
    "given below. Under-citing is the cardinal failure.\n"
    "2. Include a '## Current status and trajectory' section: is this technique rising, "
    "default, or fading? Ground it in the sources and HEDGE — say 'not widely reported' "
    "rather than 'the field abandoned it'.\n"
    "3. Write DISAGREEMENT in where sources differ: 'A reports X; B contradicts under "
    "condition Y; Z would settle it.' Do not smooth it over.\n"
    "4. Use inline LaTeX for math and markdown tables where useful. No length cap; "
    "depth and precision are the bar. Do not invent citations or numbers.\n"
    "Structure: 2-sentence intro, then ## sections, then '## Key takeaways' (bullets).\n"
    "FINALLY, after the article, output a line 'OPEN_QUESTIONS:' followed by 2-4 bullet "
    "questions that the sources leave genuinely unresolved."
)


def normalize_citations(text: str) -> str:
    """`[source:a, source:b]` -> `[source:a][source:b]` (their machine-read format)."""
    def split(m):
        ids = [i.strip() for i in m.group(1).split(",") if i.strip()]
        return "".join(f"[source:{i.replace('source:', '').strip()}]" for i in ids)
    return re.sub(r"\[(source:[^\]]+)\]", split, text)


def write_article(topic: dict, distilled: list[dict]) -> tuple[str, list[str]]:
    corpus = "\n\n".join(f"[source:{d['id']}] {d['title']}\n{d['summary']}"
                         for d in distilled)
    prompt = (f"Topic: {topic['title']}\nScope: {topic.get('notes','')}\n\n"
              f"{WRITE_RUBRIC}\n\nSOURCE SUMMARIES (cite by their [source:<id>]):\n\n{corpus}")
    resp = GEMMA.chat.completions.create(
        model="local", temperature=0.7,
        messages=[{"role": "system", "content": WRITE_SYS},
                  {"role": "user", "content": prompt}])
    body = normalize_citations(resp.choices[0].message.content.strip())

    open_qs: list[str] = []
    if "OPEN_QUESTIONS:" in body:
        body, tail = body.split("OPEN_QUESTIONS:", 1)
        open_qs = [re.sub(r"^[-*\d.\s]+", "", ln).strip()
                   for ln in tail.splitlines() if ln.strip()][:4]
    return body.strip(), open_qs


# ----------------------------------------------------------------- persistence
def fm(d: dict) -> str:
    return "---\n" + yaml.safe_dump(d, sort_keys=False, allow_unicode=True).strip() + "\n---\n\n"


def save(topic: dict, distilled: list[dict], article: str, open_qs: list[str]) -> None:
    SOURCES_DIR.mkdir(exist_ok=True)
    TOPICS_DIR.mkdir(exist_ok=True)
    today = dt.date.today().isoformat()

    for d in distilled:
        meta = {"id": d["id"], "type": "paper" if d["id"].startswith("arxiv") else "web",
                "title": d["title"], "url": d["url"], "retrieved": today,
                "maturity": "comprehensive", "topic": topic["slug"]}
        (SOURCES_DIR / id_to_filename(d["id"])).write_text(fm(meta) + d["summary"] + "\n")

    front = {"title": topic["title"], "maturity": "developing", "updated": today,
             "sources": [d["id"] for d in distilled]}
    if open_qs:
        front["open_questions"] = open_qs
    refs = "\n".join(f"- [source:{d['id']}] [{d['title']}]({d['url']})" for d in distilled)
    (TOPICS_DIR / f"{topic['slug']}.md").write_text(
        fm(front) + article + "\n\n## References\n" + refs + "\n")


def regenerate_index(tax: dict) -> None:
    lines = [f"# {tax['domain']} — Wiki\n", f"_Auto-generated {dt.date.today().isoformat()}._\n"]
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

    srcs = gather_sources(topic)
    if not srcs:
        print("no sources gathered; aborting"); return

    print(f"== distilling {len(srcs)} sources...")
    distilled = [d for d in (distill(topic, s) for s in srcs) if d]
    if not distilled:
        print("no sources distilled; aborting"); return

    print(f"== writing article from {len(distilled)} distilled sources...")
    article, open_qs = write_article(topic, distilled)
    save(topic, distilled, article, open_qs)

    topic["status"] = "done"
    TAXONOMY.write_text(yaml.safe_dump(tax, sort_keys=False, allow_unicode=True))
    regenerate_index(tax)
    print(f"== done in {time.time()-t0:.0f}s -> topics/{topic['slug']}.md "
          f"({len(distilled)} sources, {len(open_qs)} open questions)")


if __name__ == "__main__":
    main()
