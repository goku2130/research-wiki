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
import threading
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
_NVK = os.getenv("NVIDIA_API_KEY")
_NV_PREFIXES = ("nvidia/", "meta/", "deepseek-ai/", "qwen/", "mistralai/", "microsoft/")


def openrouter_client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=_ORK)


def nvidia_client() -> OpenAI:
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=_NVK)


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


def route(model: str) -> OpenAI:
    """Backend client for a model id — used by every LLM role. NVIDIA-hosted
    (nvidia/, meta/, deepseek-ai/, qwen/...) => NVIDIA build API (free); other '/'
    (google/, openai/...) => OpenRouter; mistral* => Mistral; else local Qwen GPU."""
    if model.startswith(_NV_PREFIXES) and _NVK:
        return nvidia_client()
    if "/" in model and _ORK:
        return openrouter_client()
    if model.startswith(("mistral", "magistral")) and _MKEYS:
        return mistral_client()
    if model.startswith(("gemma", "gemini")) and _GKEY:
        return google_client()
    return qwen_client()


# LLM role models — CLOUD by default now (env-overridable; set any to "local" for GPU).
ORCH_MODEL = os.getenv("ORCH_MODEL", "google/gemini-2.5-flash")
DISTILL_MODEL = os.getenv("DISTILL_MODEL", "google/gemini-2.5-flash")
REVIEWER, REVIEWER_MODEL = QWEN, os.getenv("REVIEWER_MODEL", "google/gemini-2.5-flash")


def writer_client() -> OpenAI:
    if WRITER_MODEL.startswith(("mistral", "magistral")) and _MKEYS:
        return mistral_client()   # next key in the round-robin
    return route(WRITER_MODEL)


def reviewer_client() -> OpenAI:
    return route(REVIEWER_MODEL)


# --- Google free-tier rate limiter -------------------------------------------
# The free Gemini tier allows only GOOGLE_RPM requests/minute (5 by default).
# A global sliding-window throttle serializes Google calls to stay under it so
# concurrent distill/orch/review calls don't trip 429s.
_GOOGLE_RPM = int(os.getenv("GOOGLE_RPM", "5"))
_google_lock = threading.Lock()
_google_hits: list[float] = []


def _google_throttle() -> None:
    """Block until issuing one more Google call keeps us under _GOOGLE_RPM/60s."""
    with _google_lock:
        now = time.time()
        while _google_hits and now - _google_hits[0] >= 60:
            _google_hits.pop(0)
        if len(_google_hits) >= _GOOGLE_RPM:
            wait = 60 - (now - _google_hits[0]) + 0.5
            time.sleep(max(0.0, wait))
            now = time.time()
            while _google_hits and now - _google_hits[0] >= 60:
                _google_hits.pop(0)
        _google_hits.append(time.time())


def _is_google(client: OpenAI) -> bool:
    return "generativelanguage" in str(client.base_url)


def _chat(client: OpenAI, **kw):
    """chat.completions.create with rate-limit throttle + retry/backoff."""
    google = _is_google(client)
    for attempt in range(5):
        if google:
            _google_throttle()          # stay under the free-tier RPM cap
        try:
            return client.chat.completions.create(**kw)
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            # 429/RESOURCE_EXHAUSTED needs a ~40s cooldown; other errors back off fast.
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                time.sleep(30)
            else:
                time.sleep(2 * (attempt + 1))


def strip_thoughts(text: str) -> str:
    """Gemma 4 (API) emits inline <thought>...</thought>; drop it."""
    return re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()


VERBOSE = os.getenv("VERBOSE", "").lower() in ("1", "true", "yes")


def log(tag: str, msg: str) -> None:
    """Timestamped, topic-tagged progress line (flushed so it streams live)."""
    print(f"{time.strftime('%H:%M:%S')} │ {tag[:24]:24} │ {msg}", flush=True)

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


_FCK = os.getenv("FIRECRAWL_API_KEY")
_FC = "https://api.firecrawl.dev/v1"


def firecrawl_search(query: str, n: int) -> list[dict] | None:
    if not _FCK:
        return None
    try:
        r = requests.post(f"{_FC}/search",
            headers={"Authorization": f"Bearer {_FCK}", "Content-Type": "application/json"},
            json={"query": query, "limit": n}, timeout=60)
        if r.status_code == 200:
            return [{"title": x.get("title"), "url": x.get("url"),
                     "snippet": (x.get("description") or "")[:300]}
                    for x in r.json().get("data", [])[:n]] or None
    except Exception:  # noqa: BLE001
        pass
    return None


def firecrawl_scrape(url: str, cap: int) -> str | None:
    if not _FCK:
        return None
    try:
        r = requests.post(f"{_FC}/scrape",
            headers={"Authorization": f"Bearer {_FCK}", "Content-Type": "application/json"},
            json={"url": url, "formats": ["markdown"]}, timeout=180)
        if r.status_code == 200:
            md = r.json().get("data", {}).get("markdown", "")
            return md[:cap] if md and len(md) > 200 else None
    except Exception:  # noqa: BLE001
        pass
    return None


def _cache_path(key: str):
    return OCR_CACHE / (re.sub(r"[^a-z0-9.]+", "-", key.lower()) + ".md")


def fetch_fulltext(url: str) -> str:
    """Full source text, equations preserved. Firecrawl PDF scrape first (free
    credits, matches OCR), then Mistral OCR, then Firecrawl/HTML. Disk-cached."""
    aid = arxiv_id(url)
    key = aid or urlparse(url).path
    OCR_CACHE.mkdir(exist_ok=True)
    cached = _cache_path(key)
    if cached.exists():
        return cached.read_text()[:OCR_CHARS]

    pdf_url = (f"https://arxiv.org/pdf/{aid}" if aid else
               (url if (url.lower().endswith(".pdf") or "/pdf/" in url) else None))
    text = None
    if pdf_url:
        text = firecrawl_scrape(pdf_url, OCR_CHARS)      # 1) Firecrawl PDF (equations OK)
        if not text:
            text = ocr_pdf(pdf_url, cache_key=key)        # 2) Mistral OCR fallback
    if not text:
        text = firecrawl_scrape(url, FULLTEXT_CHARS)      # 3) Firecrawl scrape of the URL
    if not text and aid:                                  # 4) HTML/trafilatura fallback
        for cand in (f"https://arxiv.org/html/{aid}",
                     f"https://ar5iv.labs.arxiv.org/html/{aid}"):
            t = _extract(cand, FULLTEXT_CHARS)
            if len(t) > 1500:
                text = t
                break
    text = text or _extract(url, FULLTEXT_CHARS) or "[no extractable text]"
    if len(text) > 2000:
        cached.write_text(text)
    return text[:OCR_CHARS]


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
    fc = firecrawl_search(query, max_results)              # cloud search (Firecrawl)
    if fc:
        return fc
    try:                                                  # fallback: local SearXNG
        r = requests.get(f"{SEARXNG}/search",
                         params={"q": query, "format": "json", "categories": "general"},
                         timeout=30)
        r.raise_for_status()
        return [{"title": i.get("title"), "url": i.get("url"),
                 "snippet": (i.get("content") or "")[:300]}
                for i in r.json().get("results", [])[:max_results]]
    except Exception:  # noqa: BLE001
        return []


def _arxiv_query(search_query: str, n: int, sort: str) -> list[dict]:
    from urllib.parse import quote
    url = (f"http://export.arxiv.org/api/query?search_query={quote(search_query)}"
           f"&sortBy={sort}&sortOrder=descending&max_results={n}")
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "research-wiki/1.0"})
        r.raise_for_status()
    except Exception:  # noqa: BLE001
        return []
    out = []
    for entry in re.findall(r"<entry>(.*?)</entry>", r.text, re.DOTALL):
        tid = re.search(r"<id>([^<]+)</id>", entry)
        title = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
        summ = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
        aid = arxiv_id(tid.group(1)) if tid else None
        if not (aid and title):
            continue
        out.append({"title": re.sub(r"\s+", " ", title.group(1)).strip(),
                    "url": f"https://arxiv.org/abs/{aid}",
                    "snippet": re.sub(r"\s+", " ", summ.group(1) if summ else "")[:300]})
    return out


def arxiv_search(query: str, n: int = 8, sort: str = "relevance",
                 loose: bool = True) -> list[dict]:
    """Query the arXiv API directly — primary literature, incl. the newest papers that
    general web search under-indexes. Returns [{title,url,snippet}]. sort='relevance'
    for on-topic seminal work, 'submittedDate' for the newest variants/follow-ups.
    Exact-phrase match first; if `loose`, fall back to an all-terms match when the phrase
    is too rare (good for variant-name lists, but risks common-word noise on vague text)."""
    # arXiv needs a clean phrase; punctuation/parens break its boolean syntax.
    phrase = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", query)).strip()
    if not phrase:
        return []
    sort = sort if sort in ("relevance", "submittedDate") else "relevance"
    res = _arxiv_query(f'all:"{phrase}"', n, sort)         # precise phrase match
    if loose and len(res) < 2:                             # rare phrase -> loosen
        res = _arxiv_query(f"all:{phrase}", n, sort)        # all-terms fallback
    return res


def arxiv_related(aid: str, n: int = 6) -> list[dict]:
    """Papers that CITE a seed arXiv paper (Semantic Scholar), ranked by their OWN
    citation count so influential descendant variants surface above recent noise.
    arXiv-only, best-effort — S2 rate-limits without a key, so this fails silently."""
    try:
        r = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{aid}/citations",
            params={"fields": "title,externalIds,citationCount", "limit": 500}, timeout=40)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception:  # noqa: BLE001
        return []
    papers = [c.get("citingPaper") or {} for c in data]
    arx = [p for p in papers if (p.get("externalIds") or {}).get("ArXiv") and p.get("title")]
    arx.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
    return [{"title": p["title"].strip(),
             "url": f"https://arxiv.org/abs/{(p['externalIds'])['ArXiv']}", "snippet": ""}
            for p in arx[:n]]


def arxiv_seed(topic: dict) -> list[dict]:
    """arXiv-native candidate set: recency-sorted search on the topic + a variant-hunting
    expansion, then citation-traversal from the top hits to catch descendant variants."""
    seeds: list[dict] = []
    seen: set[str] = set()
    title = topic["title"]
    # Notes first (they hold the real technical terms / variant names) with loose
    # matching; title only as an EXACT phrase (loose=False) to avoid common-word noise
    # like "deep dive". relevance = seminal; submittedDate = newest follow-ups.
    passes = [arxiv_search(title, 6, "relevance", loose=False)]        # exact title = seminal
    if topic.get("notes"):
        passes.append(arxiv_search(topic["notes"], 8, "relevance", loose=True))
    passes.append(arxiv_search(title, 4, "submittedDate", loose=False))  # newest follow-ups
    # Interleave passes round-robin so title-matches and notes-matches both reach the top
    # (concatenation would bury good exact-title hits under vague-notes noise, or vice-versa).
    from itertools import zip_longest
    for row in zip_longest(*passes):
        for s in row:
            if s and s["url"] not in seen:
                seen.add(s["url"]); seeds.append(s)
    for s in list(seeds[:3]):                       # follow citations -> descendant variants
        aid = arxiv_id(s["url"])
        for c in (arxiv_related(aid, 4) if aid else []):
            if c["url"] not in seen:
                seen.add(c["url"]); seeds.append(c)
    return seeds[:20]


def read_url(url: str) -> str:
    return (firecrawl_scrape(url, SCAN_CHARS) or _extract(url, SCAN_CHARS)
            or "[no extractable text]")


TOOLS = [
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the web. Returns [{title, url, snippet}].",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 6}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "arxiv_search",
        "description": "Search arXiv directly, newest-first. Best for primary papers and "
                       "recent variants/follow-ups. Use variant-hunting queries too.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "n": {"type": "integer", "default": 8}},
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
DISPATCH = {"search_web": search_web, "arxiv_search": arxiv_search, "read_url": read_url}
SUBMIT_TOOL = TOOLS[-1]


def _candidate_urls(messages: list) -> list[str]:
    """Every arXiv URL seen so far — from read_url calls AND search results —
    so we can still submit sources even if the orchestrator only searched."""
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
        # scan any string content (tool results AND the injected arXiv seed list)
        if isinstance(m.get("content"), str):
            for u in re.findall(r"https?://arxiv\.org/(?:abs|pdf|html)/[0-9.]+\w*", m["content"]):
                u = u.rstrip('.",\')')
                if u not in urls:
                    urls.append(u)
    return urls[:15]


def force_submit(messages: list) -> list[dict]:
    cands = _candidate_urls(messages)
    msg = ("Stop searching. arXiv sources seen so far (from your searches/reads):\n" +
           ("\n".join(f"- {u}" for u in cands) or "(none)") +
           f"\nCall submit_sources now with the {TARGET_SOURCES} most authoritative of "
           "these (prefer arXiv papers). Pick from the list above.")
    try:
        resp = _chat(route(ORCH_MODEL),
            model=ORCH_MODEL, messages=messages + [{"role": "user", "content": msg}],
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
        "sources on the topic — STRONGLY prefer original arXiv papers over blogs/videos, "
        "include the seminal ones PLUS recent variants/follow-ups that agree or DISAGREE.\n"
        "A pre-fetched arXiv candidate list (recency-sorted + citation-linked) is provided "
        "below — favor those. Use arxiv_search for more primary papers (and variant-hunting "
        "queries like '<topic> variants/improvements'); use search_web only to fill gaps.\n"
        "Be efficient: a couple of searches, optionally READ a few candidates with read_url "
        "to confirm. Never repeat a search. Once you have "
        f"{TARGET_SOURCES} good sources (mostly arXiv; you need NOT read them all), call "
        "submit_sources immediately with their titles and URLs."
    )
    seeds = arxiv_seed(topic)
    seed_block = ("\n".join(f"- {s['title']} — {s['url']}" for s in seeds)
                  or "(none — use arxiv_search)")
    log(topic["slug"], f"arxiv seed: {len(seeds)} candidate papers")
    messages = [{"role": "system", "content": sys_msg},
                {"role": "user",
                 "content": f"Topic: {topic['title']}\nScope: {topic.get('notes','')}\n\n"
                            f"Pre-fetched arXiv candidates (prefer these):\n{seed_block}"}]
    for step in range(MAX_TOOL_STEPS):
        resp = _chat(route(ORCH_MODEL),
            model=ORCH_MODEL, messages=messages, tools=TOOLS, temperature=0.6)
        m = resp.choices[0].message
        if not m.tool_calls:
            messages.append({"role": "user", "content": "Continue, then submit_sources."})
            continue
        messages.append(m.model_dump(exclude_none=True))
        for tc in m.tool_calls:
            name, args = tc.function.name, json.loads(tc.function.arguments or "{}")
            if name == "submit_sources":
                return args.get("sources", [])
            if VERBOSE:
                log(topic["slug"], f"  [{step}] {name}({str(args)[:60]})")
            try:
                result = DISPATCH[name](**args)
            except Exception as e:  # noqa: BLE001
                result = f"[tool error: {e}]"
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result)[:SCAN_CHARS]})
    log(topic["slug"], "  (hit step cap → forcing source list)")
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
        if VERBOSE:
            log(topic["slug"], f"    skip (thin fetch): {src['url'][:50]}")
        return None
    sid = source_id(src["url"], src["title"])
    try:
        resp = _chat(route(DISTILL_MODEL),
            model=DISTILL_MODEL, temperature=0.3,
            messages=[{"role": "system", "content": DISTILL_SYS},
                      {"role": "user", "content":
                       f"Source: {src['title']}\nURL: {src['url']}\n\nTEXT:\n{text}"}])
    except Exception as e:  # noqa: BLE001
        log(topic["slug"], f"    distill failed ({sid}): {str(e)[:60]}")
        return None
    summary = strip_thoughts(content_text(resp.choices[0].message)).strip()
    if VERBOSE:
        log(topic["slug"], f"    distilled [{sid}] ({len(summary)} chars)")
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
    """Normalize LaTeX to what GitHub's MathJax renders: $ / $$ delimiters, blank
    lines around display blocks, and only allowed macros."""
    # GitHub's MathJax rejects some macros -> map to allowed equivalents.
    text = re.sub(r"\\operatorname\*?", r"\\text", text)          # \operatorname{clip} -> \text{clip}
    text = re.sub(r"\\tag\*?\s*\{[^}]*\}", "", text)              # \tag{3} -> (removed)
    text = re.sub(r"\\bm\s*(?=\{)", r"\\mathbf", text)            # \bm{x} -> \mathbf{x}
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
    bad = sorted(set(re.findall(r"\\(?:operatorname|tag|bm|def|newcommand|label|require|cssId)\b", text)))
    if bad:
        issues.append("GitHub-disallowed macros: " + ", ".join(bad))
    if re.search(r"\\\[|\\\]|\\\(|\\\)", text):
        issues.append("raw \\[ \\] or \\( \\) math delimiters — use $$ or $ (GitHub won't render \\[ \\])")
    t = re.sub(r"```.*?```", "", text, flags=re.DOTALL)   # ignore code fences
    t = re.sub(r"`[^`\n]*`", "", t)                        # ignore inline code
    t = t.replace(r"\$", "").replace("$$", "")             # ignore escaped/display $
    if t.count("$") % 2 != 0:
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
        resp = _chat(reviewer_client(),
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
        resp = _chat(reviewer_client(),
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
           tax: dict, preserve: bool = False) -> tuple[str, list[str]]:
    corpus = "\n\n".join(f"[source:{d['id']}] {d['title']}\n{d['summary']}"
                         for d in distilled)
    fixes = "\n".join(f"- {i}" for i in issues)
    keep = ("PRESERVE MODE: fix ONLY the specific issues listed below (broken LaTeX and "
            "ungrounded claims). Preserve every other sentence, section, table, equation, "
            "and [source:<id>] citation VERBATIM. Do NOT shorten, summarize, merge, drop, "
            "or restructure any content that is not named in an issue.\n"
            if preserve else
            "Keep what is good; output the FULL improved article.\n")
    prompt = (f"Revise this wiki article to fix the issues below. {keep}"
              f"Output the article in the same format: [source:<id>] citations, "
              f"'## Current status and trajectory', '## Key takeaways', '## Related "
              f"topics', then an 'OPEN_QUESTIONS:' trailer.\n"
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


# ----------------------------------------------------------------- enrichment
DEEPEN_INSTR = (
    "You are DEEPENING an existing wiki article using NEW sources not yet in it. "
    "Integrate the new findings into the relevant sections (or add sections), cite them "
    "inline as [source:<id>], and surface any NEW disagreement. CRITICAL: preserve ALL "
    "existing content and its existing [source:...] citations — you ADD, never rewrite or "
    "drop. Keep the format ('## Current status and trajectory', '## Key takeaways', "
    "'## Related topics', then an 'OPEN_QUESTIONS:' trailer). Output ONLY the article."
)


def deepen(topic: dict, existing_body: str, new_distilled: list[dict], tax: dict):
    corpus = "\n\n".join(f"[source:{d['id']}] {d['title']}\n{d['summary']}"
                         for d in new_distilled)
    prompt = (f"Topic: {topic['title']}\n\n{DEEPEN_INSTR}\n\n{_siblings_block(topic, tax)}\n\n"
              f"NEW SOURCE SUMMARIES:\n{corpus}\n\nEXISTING ARTICLE:\n{existing_body}")
    resp = _chat(writer_client(), model=WRITER_MODEL, temperature=0.6,
                 messages=[{"role": "system", "content": WRITE_SYS},
                           {"role": "user", "content": prompt}])
    body = strip_thoughts(content_text(resp.choices[0].message))
    return _split_open_qs(strip_preamble(fix_latex(normalize_citations(body))))


def _load_source(sid: str) -> dict | None:
    """Reload a distilled source from disk (to reassemble the full source set)."""
    p = SOURCES_DIR / id_to_filename(sid)
    if not p.exists():
        return None
    parts = p.read_text().split("---", 2)
    if len(parts) < 3:
        return None
    meta = yaml.safe_load(parts[1]) or {}
    return {"id": meta.get("id", sid), "title": meta.get("title", ""),
            "url": meta.get("url", ""), "summary": parts[2].strip()}


_MATURITY_NEXT = {"stub": "developing", "developing": "comprehensive",
                  "comprehensive": "comprehensive"}


# ----------------------------------------------------------------- persistence
def fm(d: dict) -> str:
    return "---\n" + yaml.safe_dump(d, sort_keys=False, allow_unicode=True).strip() + "\n---\n\n"


def save(topic: dict, distilled: list[dict], article: str, open_qs: list[str],
         maturity: str = "developing") -> None:
    SOURCES_DIR.mkdir(exist_ok=True)
    TOPICS_DIR.mkdir(exist_ok=True)
    today = dt.date.today().isoformat()

    for d in distilled:
        meta = {"id": d["id"], "type": "paper" if d["id"].startswith("arxiv") else "web",
                "title": d["title"], "url": d["url"], "retrieved": today,
                "maturity": "comprehensive", "topic": topic["slug"]}
        (SOURCES_DIR / id_to_filename(d["id"])).write_text(fm(meta) + fix_latex(d["summary"]) + "\n")

    front = {"title": topic["title"], "maturity": maturity, "updated": today,
             "sources": [d["id"] for d in distilled]}
    if open_qs:
        front["open_questions"] = open_qs
    refs = "\n".join(f"- [source:{d['id']}] [{d['title']}]({d['url']})" for d in distilled)
    # Strip any References section the body already carries (deepen() preserves the
    # existing one) so we emit exactly one, regenerated from the full source set.
    article = re.sub(r"\n#{1,6}[ \t]*References[ \t]*\n.*$", "", article,
                     flags=re.DOTALL | re.IGNORECASE).rstrip()
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
    slug = topic["slug"]
    t0 = time.time()
    srcs = gather_sources(topic)
    log(slug, f"gather: {len(srcs)} sources ({time.time()-t0:.0f}s)")
    if not srcs:
        return []
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=DISTILL_WORKERS) as ex:
        distilled = [d for d in ex.map(lambda s: distill(topic, s), srcs) if d]
    log(slug, f"distill: {len(distilled)}/{len(srcs)} kept ({time.time()-t1:.0f}s)")
    return distilled


def compose(topic: dict, distilled: list[dict], tax: dict, verbose: bool = True,
            seed_body: str | None = None):
    """Write (or DEEPEN an existing article if seed_body) -> review + fact-check ->
    revise. Returns (article, open_qs, meta)."""
    slug = topic["slug"]
    preserve = seed_body is not None     # enrichment: ADD only, never shrink the base article
    tw = time.time()
    if seed_body:
        article, open_qs = deepen(topic, seed_body, distilled, tax)
    else:
        article, open_qs = write_article(topic, distilled, tax)
    log(slug, f"{'deepen' if seed_body else 'write'}: {len(article.split())}w ({time.time()-tw:.0f}s)")
    rounds = []
    for rnd in range(MAX_REVIEW_ROUNDS):
        tr = time.time()
        # review and fact_check are independent reads of the article -> run concurrently
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_rev = ex.submit(review, topic, article, open_qs, distilled)
            f_fc = ex.submit(fact_check, topic, article, distilled)
            verdict, grounding = f_rev.result(), f_fc.result()
        # In enrichment (preserve) mode the base article was already approved at
        # generation; deepen only ADDS. Acting on the reviewer's structural notes
        # rewrites (and shrinks) the longer draft, so ignore them and keep only the
        # grounding + latex guards that protect the NEW material.
        struct = [] if preserve else (verdict.get("issues") or []
                                      if verdict.get("verdict") != "approve" else [])
        latex = check_formulas(article)          # deterministic LaTeX validation
        issues = struct + [f"[grounding] {g}" for g in grounding] + [f"[latex] {x}" for x in latex]
        rounds.append({"structural": len(struct), "grounding": len(grounding), "latex": len(latex)})
        if not issues:
            log(slug, f"review r{rnd+1}: APPROVE ({len(distilled)} sources grounded, {time.time()-tr:.0f}s)")
            break
        log(slug, f"review r{rnd+1}: revise — {len(struct)} structural, {len(grounding)} "
                  f"grounding, {len(latex)} latex ({time.time()-tr:.0f}s)")
        new_article, new_qs = revise(topic, article, issues, distilled, tax, preserve=preserve)
        article, open_qs = new_article, (new_qs or open_qs)
    return article, open_qs, {"rounds": rounds}


def enrich(topic: dict, tax: dict) -> bool:
    """Re-visit a done topic: gather NEW sources (not already cited), DEEPEN the
    existing article to fold them in, advance maturity. Returns True if changed."""
    slug = topic["slug"]
    p = TOPICS_DIR / f"{slug}.md"
    if not p.exists():
        print(f"  {slug} not generated yet — nothing to enrich"); return False
    parts = p.read_text().split("---", 2)
    front = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()
    cited = set(front.get("sources") or [])
    print(f"== enrich {slug}: {len(cited)} sources, maturity={front.get('maturity', '?')}")
    srcs = gather_sources(topic)
    new = [s for s in srcs if source_id(s["url"], s["title"]) not in cited][:TARGET_SOURCES]
    if not new:
        print("  no new sources — already current"); return False
    print(f"== distilling {len(new)} NEW sources ({DISTILL_WORKERS}-way)...")
    with ThreadPoolExecutor(max_workers=DISTILL_WORKERS) as ex:
        new_distilled = [d for d in ex.map(lambda s: distill(topic, s), new)
                         if d and d["id"] not in cited]
    if not new_distilled:
        print("  nothing new distilled"); return False
    print(f"== deepening with {len(new_distilled)} new sources...")
    article, open_qs, _ = compose(topic, new_distilled, tax, seed_body=body)
    old = [d for sid in cited if (d := _load_source(sid))]
    all_distilled = old + new_distilled
    maturity = _MATURITY_NEXT.get(front.get("maturity", "developing"), "comprehensive")
    save(topic, all_distilled, article, open_qs, maturity=maturity)
    print(f"== enriched {slug}: +{len(new_distilled)} -> {len(all_distilled)} sources, "
          f"maturity={maturity}")
    return True


def _stalest_done(tax: dict):
    done = [t for t in tax["topics"] if t.get("status") == "done"]
    if not done:
        return None

    def updated(t):
        p = TOPICS_DIR / f"{t['slug']}.md"
        if not p.exists():
            return ""
        return str((yaml.safe_load(p.read_text().split("---", 2)[1]) or {}).get("updated", ""))
    return min(done, key=updated)


def main() -> None:
    argv = sys.argv[1:]
    enrich_mode = "--enrich" in argv
    rest = [a for a in argv if a != "--enrich"]
    slug = rest[0] if rest else None
    tax = yaml.safe_load(TAXONOMY.read_text())
    t0 = time.time()

    if enrich_mode:
        topic = (next(t for t in tax["topics"] if t["slug"] == slug) if slug
                 else _stalest_done(tax))
        if not topic:
            print("nothing to enrich"); return
        print(f"== ENRICH: {topic['slug']} ({topic['title']})")
        enrich(topic, tax)
        regenerate_index(tax)
        print(f"== enrich tick done in {time.time()-t0:.0f}s")
        return

    topic = pick_topic(tax, slug)
    log(topic["slug"], f"START — {topic['title']}")
    distilled = gather_and_distill(topic)
    if not distilled:
        log(topic["slug"], "no sources; aborting"); return
    article, open_qs, _ = compose(topic, distilled, tax)
    save(topic, distilled, article, open_qs)
    topic["status"] = "done"
    TAXONOMY.write_text(yaml.safe_dump(tax, sort_keys=False, allow_unicode=True))
    regenerate_index(tax)
    log(topic["slug"], f"DONE ({time.time()-t0:.0f}s, {len(distilled)} sources, "
                       f"{len(open_qs)} open questions)")


if __name__ == "__main__":
    main()
    # Force a clean exit: cloud HTTP clients / thread pools can keep the process
    # alive after the work is done, which would hang a cron/Actions job.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
