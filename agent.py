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
import itertools
import json
import os
import re
import sys
import time
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
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

def _load_env() -> None:
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))


_load_env()

# Local Qwen endpoints — round-robined across GPUs (5090:8181, 3090:8183).
_QWEN_URLS = [u.strip() for u in os.getenv("QWEN_URLS",
             os.getenv("QWEN_URL", "http://localhost:8181/v1")).split(",") if u.strip()]
_QWEN_CLIENTS = [OpenAI(base_url=u, api_key="local") for u in _QWEN_URLS]
_qwen_cycle = itertools.cycle(_QWEN_CLIENTS)
QWEN = _QWEN_CLIENTS[0]                        # default/first (experiment compatibility)
GEMMA = OpenAI(base_url=os.getenv("GEMMA_URL", "http://localhost:8182/v1"), api_key="local")
SEARXNG = os.getenv("SEARXNG_URL", "http://localhost:8888")


def qwen_client() -> OpenAI:
    return next(_qwen_cycle)                   # spread calls across the GPU endpoints

_GKEY = os.getenv("GOOGLE_API_KEY")
_GOOGLE_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
# One or more Mistral keys, round-robined to multiply the (low) per-key rate limit.
_MKEYS = [k.strip() for k in os.getenv("MISTRAL_API_KEYS",
          os.getenv("MISTRAL_API_KEY", "")).split(",") if k.strip()]
_MKEY = _MKEYS[0] if _MKEYS else None          # single key for OCR (limit is generous)
_mkey_cycle = itertools.cycle(_MKEYS) if _MKEYS else None


def google_client() -> OpenAI:
    return OpenAI(base_url=_GOOGLE_BASE, api_key=_GKEY)


def mistral_client() -> OpenAI:
    key = next(_mkey_cycle) if _mkey_cycle else _MKEY   # rotate keys across calls
    return OpenAI(base_url="https://api.mistral.ai/v1", api_key=key)


_ORK = os.getenv("OPENROUTER_API_KEY")


def openrouter_client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=_ORK)


def content_text(msg) -> str:
    """Read message text across backends. Mistral reasoning models (magistral)
    return content as a list of {'type': 'thinking'|'text', ...} blocks."""
    c = getattr(msg, "content", None)
    if isinstance(c, list):
        out = []
        for b in c:
            d = b if isinstance(b, dict) else (b.model_dump() if hasattr(b, "model_dump") else {})
            if isinstance(d, dict) and d.get("type") == "text":
                out.append(d.get("text", ""))
        return "\n".join(out)
    return c or ""


# Role backends = (client, model), overridable at runtime (the A/B experiment sets these).
#   WRITER   -> write_article + revise
#   REVIEWER -> review + fact_check
# Defaults: writer = Gemma 4 31B via API if a key is present else local Gemma;
#           reviewer = local Qwen (its tool loop / reasoning).
WRITER_MODEL = os.getenv("WRITER_MODEL", "gemma-4-31b-it" if _GKEY else "local")
if "/" in WRITER_MODEL and _ORK:                       # OpenRouter models: provider/name:tag
    WRITER = openrouter_client()
elif WRITER_MODEL.startswith(("mistral", "magistral")) and _MKEYS:
    WRITER = mistral_client()
elif WRITER_MODEL.startswith(("gemma", "gemini")) and _GKEY:
    WRITER = google_client()
else:
    WRITER, WRITER_MODEL = GEMMA, "local"
REVIEWER, REVIEWER_MODEL = QWEN, "local"


def writer_client() -> OpenAI:
    """Writer client for a single call — rotates Mistral keys; routes OpenRouter models."""
    if "/" in WRITER_MODEL and _ORK:
        return openrouter_client()
    if WRITER_MODEL.startswith(("mistral", "magistral")) and _MKEYS:
        return mistral_client()   # next key in the round-robin
    return WRITER


def reviewer_client() -> OpenAI:
    """Reviewer client — round-robins the local Qwen GPUs, else the override backend."""
    return qwen_client() if REVIEWER_MODEL == "local" else REVIEWER


def _chat(client: OpenAI, **kw):
    """chat.completions.create with retry/backoff (for API rate limits under concurrency)."""
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kw)
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def strip_thoughts(text: str) -> str:
    """Gemma 4 (API) emits inline <thought>...</thought>; drop it."""
    return re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()

MAX_TOOL_STEPS = 16      # cap orchestrator tool calls in stage 1
SCAN_CHARS = 5000        # truncation while the orchestrator is scanning pages
FULLTEXT_CHARS = 24000   # HTML/trafilatura read for the distillation stage (fallback)
OCR_CHARS = 90000        # Mistral OCR full-paper read (fits local Qwen 65k ctx)
TARGET_SOURCES = 8       # how many sources to aim for
MAX_REVIEW_ROUNDS = 2    # write -> review -> revise iterations before accepting
DISTILL_WORKERS = 4      # concurrent source distillations (needs server --parallel >= this)
OCR_CACHE = ROOT / "ocr_cache"


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


def ocr_pdf(pdf_url: str, cache_key: str) -> str | None:
    """Mistral OCR -> full-paper markdown (faithful, complete). Disk-cached by
    cache_key so re-runs don't re-OCR. Returns None if no key or on failure."""
    if not _MKEY:
        return None
    OCR_CACHE.mkdir(exist_ok=True)
    cached = OCR_CACHE / (re.sub(r"[^a-z0-9.]+", "-", cache_key.lower()) + ".md")
    if cached.exists():
        return cached.read_text()
    try:
        r = requests.post("https://api.mistral.ai/v1/ocr",
            headers={"Authorization": f"Bearer {_MKEY}", "Content-Type": "application/json"},
            json={"model": "mistral-ocr-latest",
                  "document": {"type": "document_url", "document_url": pdf_url}},
            timeout=180)
        r.raise_for_status()
        text = "\n\n".join(p.get("markdown", "") for p in r.json().get("pages", []))
    except Exception as e:  # noqa: BLE001
        print(f"    OCR failed ({pdf_url}): {str(e)[:100]}")
        return None
    if len(text) > 2000:
        cached.write_text(text)
        return text
    return None


def fetch_fulltext(url: str) -> str:
    """Full source text. Prefer Mistral OCR of the complete PDF (faithful, no
    truncation of the paper); fall back to arXiv HTML, then trafilatura."""
    aid = arxiv_id(url)
    # 1) Mistral OCR of the full paper — the highest-fidelity path.
    if _MKEY and (aid or url.lower().endswith(".pdf") or "/pdf/" in url):
        pdf_url = f"https://arxiv.org/pdf/{aid}" if aid else url
        text = ocr_pdf(pdf_url, cache_key=aid or urlparse(url).path)
        if text:
            return text[:OCR_CHARS]
    # 2) arXiv HTML (real LaTeX) / 3) trafilatura fallback.
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
        resp = qwen_client().chat.completions.create(
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
        "You are a research agent for an expert, citation-backed wiki. Find the PRIMARY "
        "sources on the topic — prefer original papers (arXiv) over blogs, include the "
        "seminal ones plus recent work that agrees or DISAGREES.\n"
        "Be efficient: run at most 2-3 searches, then READ candidate pages with read_url "
        "to confirm them. Never repeat a search. Once you have identified "
        f"{TARGET_SOURCES} good sources (you do NOT need to read all of them), call "
        "submit_sources immediately with their titles and URLs."
    )
    messages = [{"role": "system", "content": sys_msg},
                {"role": "user",
                 "content": f"Topic: {topic['title']}\nScope: {topic.get('notes','')}"}]
    for step in range(MAX_TOOL_STEPS):
        resp = qwen_client().chat.completions.create(
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
    try:
        resp = qwen_client().chat.completions.create(
            model="local", temperature=0.3,
            messages=[{"role": "system", "content": DISTILL_SYS},
                      {"role": "user", "content":
                       f"Source: {src['title']}\nURL: {src['url']}\n\nTEXT:\n{text}"}])
    except Exception as e:  # noqa: BLE001
        print(f"    distill failed ({sid}): {e}")
        return None
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
    "5. MATH DELIMITERS: use ONLY $...$ for inline and $$...$$ for display math. NEVER "
    "use \\[ \\] or \\( \\) (they do not render). Balance every brace and \\left/\\right.\n"
    "Structure: 2-sentence intro, then ## sections, then '## Key takeaways' (bullets), "
    "then '## Related topics' linking genuinely-related sibling articles (only from the "
    "provided list) as markdown links.\n"
    "FINALLY, after the article, output a line 'OPEN_QUESTIONS:' followed by 2-4 bullet "
    "questions that the sources leave genuinely unresolved."
)


def normalize_citations(text: str) -> str:
    """`[source:a, source:b]` -> `[source:a][source:b]` (their machine-read format)."""
    def split(m):
        ids = [i.strip() for i in m.group(1).split(",") if i.strip()]
        return "".join(f"[source:{i.replace('source:', '').strip()}]" for i in ids)
    return re.sub(r"\[(source:[^\]]+)\]", split, text)


def strip_preamble(text: str) -> str:
    """Remove the reviser's conversational wrapper that leaks into articles, e.g.
    'Here is the revised wiki article ...' + a change-list, before the real content."""
    text = text.strip()
    if re.match(r"(here is|below is|sure|certainly|i have|i've|the following|here'?s)\b"
                r".{0,90}?\b(article|wiki|revised|version|updated|deep dive)\b",
                text[:180], flags=re.I):
        m = re.search(r"^#{1,4} ", text, flags=re.M)   # cut to the first markdown heading
        if m:
            text = text[m.start():]
    text = re.sub(r"\n+\s*(let me know|i hope this|feel free|note:)\b[^\n]*$", "",
                  text, flags=re.I)
    return text.strip()


def fix_latex(text: str) -> str:
    """Normalize LaTeX delimiters to what GitHub/KaTeX markdown renders ($ and $$).
    `\\[ ... \\]` -> `$$...$$`, `\\( ... \\)` -> `$...$`. Leaves \\left[ / \\right] alone."""
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n",
                  text, flags=re.DOTALL)
    text = re.sub(r"\\\((.+?)\\\)", lambda m: f"${m.group(1).strip()}$",
                  text, flags=re.DOTALL)
    # fallback for any leftover UNPAIRED delimiters (malformed/truncated formulas)
    text = text.replace(r"\[", "$$").replace(r"\]", "$$").replace(r"\(", "$").replace(r"\)", "$")
    # a bare % inside a math span is a LaTeX comment (eats the line) -> escape it
    def esc_pct(m):
        return m.group(1) + re.sub(r"(?<!\\)%", r"\\%", m.group(2)) + m.group(1)
    text = re.sub(r"(\${1,2})(.+?)\1", esc_pct, text, flags=re.DOTALL)
    # GitHub renders $$...$$ display math ONLY as a standalone block with BLANK lines
    # around it — otherwise it shows raw LaTeX. Normalize every display block.
    text = re.sub(r"[ \t]*\${2}[ \t]*\n?(.+?)\n?[ \t]*\${2}[ \t]*",
                  lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n", text, flags=re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def check_formulas(text: str) -> list[str]:
    """Deterministic formula validator -> list of issues to fix (fed to revise)."""
    issues = []
    if re.search(r"\\\[|\\\]|\\\(|\\\)", text):
        issues.append("raw \\[ \\] or \\( \\) math delimiters — use $$ or $ (GitHub won't render \\[ \\])")
    if text.replace("$$", "").count("$") % 2 != 0:
        issues.append("unbalanced $ (odd count) — an inline-math delimiter is missing")
    for m in re.finditer(r"\$\$?(.+?)\$\$?", text, flags=re.DOTALL):
        f = m.group(1)
        if f.count("{") != f.count("}"):
            issues.append(f"unbalanced braces {{}} in formula: {f.strip()[:60]}")
        # count real \left/\right delimiters only (exclude \leftarrow, \rightarrow, ...)
        if len(re.findall(r"\\left(?![a-zA-Z])", f)) != len(re.findall(r"\\right(?![a-zA-Z])", f)):
            issues.append(f"\\left/\\right mismatch in formula: {f.strip()[:60]}")
    return issues[:8]


def _siblings_block(topic: dict, tax: dict) -> str:
    sibs = [f"- [{t['title']}]({t['slug']}.md)" for t in tax["topics"]
            if t["slug"] != topic["slug"]]
    return "RELATED SIBLING TOPICS (link only the relevant ones):\n" + "\n".join(sibs)


def _split_open_qs(body: str) -> tuple[str, list[str]]:
    open_qs: list[str] = []
    if "OPEN_QUESTIONS:" in body:
        body, tail = body.split("OPEN_QUESTIONS:", 1)
        open_qs = [re.sub(r"^[-*\d.\s]+", "", ln).strip()
                   for ln in tail.splitlines() if ln.strip()][:4]
    return body.strip(), open_qs


def write_article(topic: dict, distilled: list[dict], tax: dict) -> tuple[str, list[str]]:
    corpus = "\n\n".join(f"[source:{d['id']}] {d['title']}\n{d['summary']}"
                         for d in distilled)
    prompt = (f"Topic: {topic['title']}\nScope: {topic.get('notes','')}\n\n"
              f"{WRITE_RUBRIC}\n\n{_siblings_block(topic, tax)}\n\n"
              f"SOURCE SUMMARIES (cite by their [source:<id>]):\n\n{corpus}")
    resp = _chat(writer_client(),
        model=WRITER_MODEL, temperature=0.7,
        messages=[{"role": "system", "content": WRITE_SYS},
                  {"role": "user", "content": prompt}])
    body = strip_thoughts(content_text(resp.choices[0].message))
    return _split_open_qs(strip_preamble(fix_latex(normalize_citations(body))))


# ----------------------------------------------------------------- review gate
REVIEW_SYS = (
    "You are a strict reviewer for an expert research wiki. Approve an article ONLY if it "
    "meets ALL of these; otherwise request_changes with specific, actionable issues:\n"
    "1. Every non-obvious claim is cited inline as [source:<id>] using ONLY the valid ids.\n"
    "2. It has a '## Current status and trajectory' section, hedged (no ungrounded "
    "'the field abandoned X').\n"
    "3. Disagreement between sources is written in, not smoothed over.\n"
    "4. It is deep and precise (not a thin stub): key formulas, numbers, and method.\n"
    "5. It has '## Key takeaways'. Open questions are stored in FRONTMATTER, not the "
    "body — treat rule 5 as satisfied when the 'Open questions present' count below is "
    ">= 2. Do NOT ask for an open-questions heading in the body.\n"
    "Reason briefly, THEN on the last lines write 'VERDICT: approve' or "
    "'VERDICT: request_changes', and if requesting changes an 'ISSUES:' block with one "
    "concrete fix per bullet ('add the PPO clip equation', 'cite the AIME number')."
)


def review(topic: dict, article: str, open_qs: list[str], distilled: list[dict]) -> dict:
    ids = ", ".join(d["id"] for d in distilled)
    content = (f"Topic: {topic['title']}\nValid source ids: {ids}\n"
               f"Open questions present: {len(open_qs)}\n\nARTICLE:\n{article}")
    try:
        resp = reviewer_client().chat.completions.create(
            model=REVIEWER_MODEL, temperature=0.2, max_tokens=4000,
            messages=[{"role": "system", "content": REVIEW_SYS},
                      {"role": "user", "content": content}])
        msg = resp.choices[0].message
        out = strip_thoughts(content_text(msg))
        if "VERDICT:" not in out:
            out = (getattr(msg, "reasoning_content", "") or "") + "\n" + out
    except Exception as e:  # noqa: BLE001
        print(f"  review failed: {e}")
        return {"verdict": "approve", "issues": []}
    verdict = "request_changes" if re.search(r"VERDICT:\s*request", out, re.I) else "approve"
    issues: list[str] = []
    if verdict == "request_changes" and "ISSUES:" in out:
        tail = out.rsplit("ISSUES:", 1)[1]
        issues = [re.sub(r"^[-*\d.\s]+", "", ln).strip()
                  for ln in tail.splitlines() if ln.strip()][:8]
    return {"verdict": verdict, "issues": issues}


# ----------------------------------------------------------------- fact-check gate
FACTCHECK_SYS = (
    "You are a strict fact-checker for a citation-backed wiki. You are given an ARTICLE "
    "and the SOURCE SUMMARIES it cites. Audit every cited claim: is it actually supported "
    "by the summary of the source it cites? Flag fabricated numbers/equations/dates/"
    "findings, and claims attributed to the wrong source. Also flag any claim that is "
    "clearly false on its face.\n"
    "Work through the article claim by claim and reason explicitly. THEN, on the very "
    "last lines, write 'ISSUES:' followed by one bullet per problem — each stating the "
    "claim, the id it cites, and the fix (correct/remove/re-cite). If every cited claim "
    "is supported, write exactly 'ISSUES: NONE'."
)


def fact_check(topic: dict, article: str, distilled: list[dict]) -> list[str]:
    corpus = "\n\n".join(f"[source:{d['id']}] {d['title']}\n{d['summary']}"
                         for d in distilled)
    content = f"ARTICLE:\n{article}\n\nSOURCE SUMMARIES:\n{corpus}"
    try:
        resp = reviewer_client().chat.completions.create(
            model=REVIEWER_MODEL, temperature=0.1, max_tokens=8000,
            messages=[{"role": "system", "content": FACTCHECK_SYS},
                      {"role": "user", "content": content}])
        msg = resp.choices[0].message
        out = strip_thoughts(content_text(msg))
        if "ISSUES:" not in out:  # reasoning model may leave answer only in reasoning
            out = (getattr(msg, "reasoning_content", "") or "") + "\n" + out
    except Exception as e:  # noqa: BLE001
        print(f"  fact_check failed: {e}")
        return []
    if "ISSUES:" not in out:
        return []
    tail = out.rsplit("ISSUES:", 1)[1].strip()
    if tail.upper().startswith("NONE") or not tail:
        return []
    return [re.sub(r"^[-*\d.\s]+", "", ln).strip()
            for ln in tail.splitlines() if ln.strip() and ln.strip().upper() != "NONE"]


def revise(topic: dict, article: str, issues: list[str], distilled: list[dict],
           tax: dict) -> tuple[str, list[str]]:
    corpus = "\n\n".join(f"[source:{d['id']}] {d['title']}\n{d['summary']}"
                         for d in distilled)
    fixes = "\n".join(f"- {i}" for i in issues)
    prompt = (f"Revise this wiki article to fix the issues below. Keep what is good; "
              f"output the FULL improved article (same format: [source:<id>] citations, "
              f"'## Current status and trajectory', '## Key takeaways', '## Related "
              f"topics', then an 'OPEN_QUESTIONS:' trailer).\n"
              f"Output ONLY the article itself — start directly with the content. Do NOT "
              f"add any preamble, greeting, or list of changes you made.\n"
              f"For any issue marked [grounding], the claim is NOT supported by its cited "
              f"source: correct it to match the source, or REMOVE it — never keep a "
              f"fabricated number, equation, or misattribution.\n\n"
              f"ISSUES:\n{fixes}\n\n{_siblings_block(topic, tax)}\n\n"
              f"SOURCE SUMMARIES:\n{corpus}\n\nCURRENT ARTICLE:\n{article}")
    resp = _chat(writer_client(),
        model=WRITER_MODEL, temperature=0.6,
        messages=[{"role": "system", "content": WRITE_SYS},
                  {"role": "user", "content": prompt}])
    body = strip_thoughts(content_text(resp.choices[0].message))
    return _split_open_qs(strip_preamble(fix_latex(normalize_citations(body))))


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


def gather_and_distill(topic: dict) -> list[dict]:
    srcs = gather_sources(topic)
    if not srcs:
        return []
    print(f"== distilling {len(srcs)} sources ({DISTILL_WORKERS}-way concurrent)...")
    with ThreadPoolExecutor(max_workers=DISTILL_WORKERS) as ex:
        return [d for d in ex.map(lambda s: distill(topic, s), srcs) if d]


def compose(topic: dict, distilled: list[dict], tax: dict, verbose: bool = True):
    """Write -> review + fact-check -> revise, using the current WRITER/REVIEWER
    backends. Returns (article, open_qs, meta) where meta records the gate activity."""
    article, open_qs = write_article(topic, distilled, tax)
    rounds = []
    for rnd in range(MAX_REVIEW_ROUNDS):
        # review and fact_check are independent reads of the article -> run concurrently
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_rev = ex.submit(review, topic, article, open_qs, distilled)
            f_fc = ex.submit(fact_check, topic, article, distilled)
            verdict, grounding = f_rev.result(), f_fc.result()
        struct = verdict.get("issues") or [] if verdict.get("verdict") != "approve" else []
        latex = check_formulas(article)          # deterministic LaTeX validation
        issues = struct + [f"[grounding] {g}" for g in grounding] + [f"[latex] {x}" for x in latex]
        rounds.append({"structural": len(struct), "grounding": len(grounding), "latex": len(latex)})
        if not issues:
            if verbose:
                print(f"  round {rnd+1}: APPROVE ({len(distilled)} sources grounded)")
            break
        if verbose:
            print(f"  round {rnd+1}: revise — {len(struct)} structural, "
                  f"{len(grounding)} grounding, {len(latex)} latex")
        new_article, new_qs = revise(topic, article, issues, distilled, tax)
        article, open_qs = new_article, (new_qs or open_qs)
    return article, open_qs, {"rounds": rounds}


def main() -> None:
    tax = yaml.safe_load(TAXONOMY.read_text())
    forced = sys.argv[1] if len(sys.argv) > 1 else None
    topic = pick_topic(tax, forced)
    print(f"== topic: {topic['slug']} ({topic['title']})")
    t0 = time.time()

    distilled = gather_and_distill(topic)
    if not distilled:
        print("no sources; aborting"); return

    print(f"== writing article from {len(distilled)} distilled sources...")
    article, open_qs, _ = compose(topic, distilled, tax)
    save(topic, distilled, article, open_qs)

    topic["status"] = "done"
    TAXONOMY.write_text(yaml.safe_dump(tax, sort_keys=False, allow_unicode=True))
    regenerate_index(tax)
    print(f"== done in {time.time()-t0:.0f}s -> topics/{topic['slug']}.md "
          f"({len(distilled)} sources, {len(open_qs)} open questions)")


if __name__ == "__main__":
    main()
