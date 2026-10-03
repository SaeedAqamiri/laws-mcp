"""Iranian laws MCP server — index, read and download laws from ekhtebar.ir.

Scrapes the laws hub (https://www.ekhtebar.ir/قوانین/) into a local index,
renders law pages to markdown files, downloads the official PDFs and extracts
their text with pdftotext. Scanned PDFs fall back to VLM OCR (GLM vision via
the Z.AI API). No external service is required at runtime; if a self-hosted
firecrawl is available, set FIRECRAWL_URL and page fetching goes through it.

Data layout (override dir with LAWS_DATA_DIR):
    data/index.json          scraped hub index (~650 laws)
    data/laws/<slug>.md      law page text
    data/pdf/<slug>.pdf      downloaded PDFs
    data/pdf-text/<slug>.txt extracted PDF text
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

BASE = "https://www.ekhtebar.ir"
HUB = BASE + "/%d9%82%d9%88%d8%a7%d9%86%db%8c%d9%86/"  # /قوانین/
UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LAWS_DATA_DIR", ROOT / "data"))
LAWS_DIR = DATA_DIR / "laws"
PDF_DIR = DATA_DIR / "pdf"
TXT_DIR = DATA_DIR / "pdf-text"
INDEX_FILE = DATA_DIR / "index.json"
STATUS_FILE = DATA_DIR / "sync_status.json"

ZAI_BASE = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
ZAI_KEY = os.environ.get("ZAI_API_KEY", "")
VLM_MODEL = os.environ.get("LAWS_VLM_MODEL", "glm-4.6v")
FIRECRAWL_URL = os.environ.get("FIRECRAWL_URL", "").rstrip("/")
DELAY = float(os.environ.get("LAWS_DELAY", "0.4"))

OCR_PROMPT = (
    "متن این صفحه از یک سند حقوقی فارسی را دقیقاً و کامل استخراج کن. "
    "فقط خود متن را به‌صورت مارک‌داون بنویس؛ هیچ توضیح، مقدمه یا جمع‌بندی اضافه نکن. "
    "شماره‌گذاری مواد، تبصره‌ها و اعداد را دقیقاً همان‌طور که هست حفظ کن."
)

mcp = FastMCP("iran-laws")

# ---------------------------------------------------------------- helpers

_AR = {"ي": "ی", "ك": "ک", "ۀ": "ه", "أ": "ا", "إ": "ا"}
_FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def norm(s: str) -> str:
    """Normalize Persian text for matching (alef, keheh, ZWNJ, digits)."""
    s = s or ""
    for a, b in _AR.items():
        s = s.replace(a, b)
    s = s.replace("\u200c", " ").replace("\u200f", "").replace("\u200e", "")
    s = s.translate(_FA_DIGITS)
    return re.sub(r"\s+", " ", s).strip().lower()


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def slugify(url: str) -> str:
    p = urlparse(url)
    if p.query.startswith("p="):
        return f"p{p.query.split('=')[1].split('&')[0]}"
    segs = [s for s in p.path.split("/") if s]
    slug = unquote(segs[-1]) if segs else "index"
    slug = re.sub(r'[\\/:*?"<>|]+', "-", slug).strip(" .-")
    return (slug or "law")[:90]


def _get(url: str, tries: int = 3) -> httpx.Response:
    last = "unknown"
    with httpx.Client(headers=UA, timeout=30, follow_redirects=True) as c:
        for i in range(tries):
            try:
                r = c.get(url)
                if r.status_code == 200:
                    return r
                last = f"HTTP {r.status_code}"
                if r.status_code in (429, 500, 502, 503):
                    time.sleep(2 * (i + 1))
                    continue
                break
            except httpx.HTTPError as e:
                last = repr(e)
                time.sleep(2 * (i + 1))
    raise RuntimeError(f"fetch failed ({url}): {last}")


def firecrawl_markdown(url: str) -> str | None:
    """Optional: fetch page markdown through a self-hosted firecrawl."""
    if not FIRECRAWL_URL:
        return None
    try:
        r = httpx.post(
            f"{FIRECRAWL_URL}/v1/scrape",
            json={"url": url, "formats": ["markdown"]},
            timeout=120,
        )
        if r.status_code == 200:
            return (r.json().get("data") or {}).get("markdown") or None
    except Exception:
        return None
    return None


# ---------------------------------------------------------------- index

def scrape_index() -> list[dict]:
    r = _get(HUB)
    soup = BeautifulSoup(r.text, "lxml")
    items: list[dict] = []
    seen: set[str] = set()
    # two tables on the hub page: main laws (aligncenter) + full tablepress table
    for tr in soup.select("table.aligncenter tr, table.tablepress tbody tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        a = tds[0].find("a", href=True)
        if not a:
            continue
        url = a["href"].split("#")[0]
        if "ekhtebar.ir" not in url:
            continue
        title = _ws(a.get_text(strip=True))
        if not title:
            continue
        key = url if "?p=" in url else url
        if key in seen:
            continue
        seen.add(key)
        year = _ws(tds[1].get_text(strip=True)) if len(tds) > 1 else ""
        pdf = None
        if ".pdf" in url.lower():
            # rare rows where the title itself links straight to the PDF (no post page)
            url, pdf = "", url
        for t in tds[1:]:
            pa = t.find("a", href=True)
            if pa and ".pdf" in pa["href"].lower() and not pdf:
                pdf = pa["href"]
        items.append({"title": title, "url": url, "year": year, "pdf": pdf})
    return items


def load_index(refresh: bool = False) -> list[dict]:
    if not refresh and INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    items = scrape_index()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(
        json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return items


def resolve(ref: str) -> dict:
    """Resolve a law reference (URL / slug / title fragment) to an index item."""
    ref = (ref or "").strip()
    if not ref:
        raise ValueError("empty reference")
    if ref.startswith("http"):
        for it in load_index():
            if it["url"].split("#")[0] == ref.split("#")[0]:
                return it
        return {"title": slugify(ref), "url": ref, "year": "", "pdf": None}
    items = load_index()
    n = norm(ref)
    for it in items:
        if norm(it["title"]) == n or slugify(it["url"]) == ref or it["url"].rstrip("/").endswith(ref.rstrip("/")):
            return it
    scored = sorted(
        items,
        key=lambda it: SequenceMatcher(None, norm(it["title"]), n).ratio(),
        reverse=True,
    )
    if scored and SequenceMatcher(None, norm(scored[0]["title"]), n).ratio() >= 0.55:
        return scored[0]
    raise LookupError(f"no law matches {ref!r} in the index; use law_search first")


# ---------------------------------------------------------------- page -> markdown

HEADINGS = {"h1": 2, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 5}


def _render_block(el) -> str | None:
    name = el.name
    if name == "tr":
        cells = [_ws(c.get_text(" ", strip=True)) for c in el.find_all(["td", "th"])]
        return "| " + " | ".join(cells) + " |" if any(cells) else None
    if name in HEADINGS:
        txt = _ws(el.get_text(" ", strip=True))
        return f"{'#' * HEADINGS[name]} {txt}" if txt else None
    if name == "li":
        for sub in el.find_all(["ul", "ol"]):
            sub.extract()
        txt = _ws(el.get_text(" ", strip=True))
        return f"- {txt}" if txt else None
    txt = _ws(el.get_text(" ", strip=True))
    return txt or None


def render_entry(html: str) -> tuple[str, str, list[str]]:
    """Extract (title, markdown, more_page_urls) from a law page."""
    soup = BeautifulSoup(html, "lxml")
    title = ""
    if soup.title:
        title = re.sub(r"\s*[-–|]\s*پایگاه خبری اختبار\s*$", "", soup.title.get_text(strip=True))
    ec = soup.select_one(".entry-content") or soup.select_one("article") or soup.body
    for t in ec.select("script, style, nav, form, iframe, [class*='addtoany']"):
        t.decompose()
    blocks = []
    for el in ec.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "blockquote"]):
        if el.name == "p" and el.find_parent(["li", "td", "blockquote"]):
            continue
        b = _render_block(el)
        if b:
            blocks.append(b)
    text = "\n\n".join(blocks)
    more = []
    for a in soup.select('[class*="page-nav"] a, [class*="pages-nav"] a, .post-page-numbers a'):
        t = a.get_text(strip=True)
        href = a.get("href") or ""
        if t.isdigit() and href.startswith("http") and href not in more:
            more.append(href)
    return title, text, more[:20]


def fetch_law_markdown(url: str) -> tuple[str, str]:
    """Fetch a law post (following post pagination) -> (title, markdown)."""
    return _fetch_entry(url)[:2]


def _fetch_entry(url: str) -> tuple[str, str, list[str]]:
    """Fetch a post -> (title, markdown, pdf links found in the content)."""
    if ".pdf" in url.lower():
        raise RuntimeError("no post page for this law; use iran-laws_pdf")
    md = firecrawl_markdown(url)
    if md:
        return "", md, []
    r = _get(url)
    soup = BeautifulSoup(r.text, "lxml")
    ec = soup.select_one(".entry-content") or soup
    pdfs = [
        a["href"].split("#")[0]
        for a in ec.select("a[href]")
        if ".pdf" in (a.get("href") or "").lower()
    ]
    title, text, more = render_entry(r.text)
    seen = {str(r.url)}
    for href in more:
        if href in seen:
            continue
        seen.add(href)
        try:
            _, t2, more2 = render_entry(_get(href).text)
            text += "\n\n" + t2
            more.extend(more2[:5])
            time.sleep(DELAY)
        except Exception:
            continue
        if len(seen) > 25:
            break
    return title, text, list(dict.fromkeys(pdfs))


# ---------------------------------------------------------------- pdf + ocr

def _pdf_quality(txt: str) -> tuple[int, float, float]:
    t = norm(txt)
    letters = [ch for ch in t if ch.isalpha()]
    n = max(len(letters), 1)
    fa = sum(1 for ch in letters if "\u0600" <= ch <= "\u06ff") / n
    la = sum(1 for ch in letters if ch.isascii()) / n
    return len(t), fa, la


def _needs_ocr(txt: str) -> bool:
    n, fa, la = _pdf_quality(txt)
    if n >= 200 and la >= 0.4:
        return False  # real (possibly latin-script) text layer
    return n < 200 or fa < 0.4


def _vlm_ocr_page(png: Path) -> str:
    b64 = base64.b64encode(png.read_bytes()).decode()
    body = {
        "model": VLM_MODEL,
        "thinking": {"type": "disabled"},
        "temperature": 0.1,
        "max_tokens": 8192,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
    }
    r = httpx.post(
        f"{ZAI_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {ZAI_KEY}"},
        json=body,
        timeout=600,
    )
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()


def _ocr_pdf(pdf: Path, txt_path: Path, start_page: int, max_pages: int) -> dict:
    if not ZAI_KEY:
        raise RuntimeError("ZAI_API_KEY is not set; cannot OCR scanned PDFs")
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            ["pdftoppm", "-r", "110", "-jpeg", "-jpegopt", "quality=85",
             "-f", str(start_page), "-l",
             str(start_page + max_pages - 1), str(pdf), str(Path(td) / "pg")],
            check=True, timeout=600,
        )
        pages = sorted(Path(td).glob("pg-*.jpg"))
        if not pages:
            raise RuntimeError("pdftoppm produced no pages")
        mode = "a" if start_page > 1 and txt_path.exists() else "w"
        parts = []
        with open(txt_path, mode, encoding="utf-8") as f:
            for p in pages:
                out = _vlm_ocr_page(p)
                parts.append(out)
                f.write(out + "\n\n")
                time.sleep(0.5)
        return {"ocr_pages": len(parts), "ocr_start_page": start_page,
                "preview": parts[0][:400] if parts else ""}


def pdf_pipeline(url: str, slug: str, ocr: bool, ocr_start_page: int,
                 max_ocr_pages: int, refresh: bool) -> dict:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = PDF_DIR / f"{slug}.pdf"
    txt = TXT_DIR / f"{slug}.txt"
    if refresh or not pdf.exists():
        with httpx.Client(headers=UA, timeout=120, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
            pdf.write_bytes(r.content)
    info: dict = {"pdf": str(pdf), "bytes": pdf.stat().st_size}
    if refresh or not txt.exists() or txt.stat().st_size == 0:
        subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf), str(txt)], check=True, timeout=300)
    body = txt.read_text(encoding="utf-8", errors="replace")
    n, fa, la = _pdf_quality(body)
    info.update({"text_file": str(txt), "text_chars": n, "via": "pdftotext"})
    if "اسالمی" in body or "اوالد" in body:
        # some site PDFs map the lam-alef ligature to a swapped «ال» in their
        # text layer; not safely repairable — flag it so the agent can OCR
        info["artifact"] = ("lam-alef ligature broken in the PDF text layer "
                            "(e.g. «اسالمی»); text usable, but OCR gives cleaner text")
    if _needs_ocr(body):
        if not ocr:
            info["warning"] = ("PDF looks scanned; re-call with ocr=true "
                               f"(ZAI key {'set' if ZAI_KEY else 'MISSING'})")
        else:
            npages = 0
            try:
                out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, timeout=60)
                m = re.search(r"Pages:\s+(\d+)", out.stdout)
                npages = int(m.group(1)) if m else 0
            except Exception:
                pass
            res = _ocr_pdf(pdf, txt, ocr_start_page, max_ocr_pages)
            body = txt.read_text(encoding="utf-8", errors="replace")
            n, fa, la = _pdf_quality(body)
            info.update({"text_chars": n, "via": f"vlm-ocr:{VLM_MODEL}", **res,
                         "total_pages": npages,
                         "pages_remaining": max(0, npages - (ocr_start_page + res["ocr_pages"] - 1))})
    info["preview"] = body.strip()[:400]
    return info


# ---------------------------------------------------------------- background sync

_sync: dict = {"running": False, "done": 0, "total": 0, "failed": 0,
               "current": "", "started": "", "finished": "", "errors": [], "stop": False,
               "kind": ""}
_sync_lock = threading.Lock()


def _write_status():
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(json.dumps(_sync, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def _sync_worker(items: list[dict], include_pdfs: bool, refresh: bool):
    try:
        for it in items:
            if _sync["stop"]:
                break
            slug = slugify(it["url"])
            _sync["current"] = it["title"]
            md = LAWS_DIR / f"{slug}.md"
            if refresh or not md.exists():
                try:
                    _, text = fetch_law_markdown(it["url"])
                    if not text.strip():
                        raise RuntimeError("empty page")
                    LAWS_DIR.mkdir(parents=True, exist_ok=True)
                    md.write_text(
                        f"# {it['title']}\n\n> منبع: {it['url']}\n\n{text}", encoding="utf-8"
                    )
                    time.sleep(DELAY)
                except Exception as e:
                    _sync["failed"] += 1
                    if len(_sync["errors"]) < 50:
                        _sync["errors"].append(f"{it['title']}: {e}")
            if include_pdfs and it.get("pdf"):
                try:
                    pdf_pipeline(it["pdf"], slug, ocr=False, ocr_start_page=1,
                                 max_ocr_pages=0, refresh=refresh)
                    time.sleep(DELAY)
                except Exception as e:
                    _sync["failed"] += 1
                    if len(_sync["errors"]) < 50:
                        _sync["errors"].append(f"pdf {it['title']}: {e}")
            _sync["done"] += 1
            if _sync["done"] % 5 == 0:
                _write_status()
    finally:
        _sync["running"] = False
        _sync["finished"] = datetime.now().isoformat(timespec="seconds")
        _write_status()


# ---------------------------------------------------------------- tools

def _out(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


@mcp.tool(name="index")
def laws_index(refresh: bool = False) -> str:
    """Index of Iranian laws on ekhtebar.ir (قوانین hub page).

    First call scrapes and caches ~650 laws (title / url / year / pdf link).
    Args:
        refresh: re-scrape the hub page even if a cached index exists.
    """
    try:
        items = load_index(refresh)
        years = sorted({i["year"] for i in items if i["year"]})
        return _out({
            "count": len(items),
            "with_pdf": sum(1 for i in items if i["pdf"]),
            "years": [years[0], years[-1]] if years else [],
            "index_file": str(INDEX_FILE),
            "sample": [i["title"] for i in items[:5]],
        })
    except Exception as e:
        return _out({"error": str(e)})


@mcp.tool(name="list")
def laws_list(query: str = "", year: str = "", offset: int = 0, limit: int = 20) -> str:
    """Page through the cached laws index, optionally filtered.

    Args:
        query: substring of the law title (Persian-friendly matching).
        year: approval year, e.g. 1403 or ۱۴۰۳.
        offset / limit: pagination over matched entries.
    """
    try:
        items = load_index()
        nq, ny = norm(query), norm(year)
        hits = [
            i for i in items
            if (not nq or nq in norm(i["title"]))
            and (not ny or norm(i["year"]) == ny)
        ]
        return _out({
            "matched": len(hits), "offset": offset,
            "returned": len(hits[offset:offset + limit]),
            "items": [
                {"title": i["title"], "year": i["year"], "url": i["url"],
                 "slug": slugify(i["url"]), "pdf": bool(i["pdf"])}
                for i in hits[offset:offset + limit]
            ],
        })
    except Exception as e:
        return _out({"error": str(e)})


@mcp.tool(name="search")
def law_search(query: str, limit: int = 8) -> str:
    """Full-text search across the whole ekhtebar.ir site (site search, not just the laws hub).

    Args:
        query: Persian search phrase.
        limit: max results.
    """
    try:
        r = _get(BASE + "/?s=" + quote(query))
        soup = BeautifulSoup(r.text, "lxml")
        out, seen = [], set()
        for a in soup.select(".post-title a[href]"):
            href = a["href"].split("#")[0]
            title = _ws(a.get_text(strip=True))
            if not title or href in seen or "ekhtebar.ir" not in href:
                continue
            seen.add(href)
            out.append({"title": title, "url": href, "slug": slugify(href)})
            if len(out) >= limit:
                break
        return _out({"query": query, "results": out})
    except Exception as e:
        return _out({"error": str(e)})


@mcp.tool(name="get")
def law_get(ref: str, refresh: bool = False) -> str:
    """Fetch a law page from ekhtebar.ir and save it as a markdown file.

    Use this for the full text of a law. ref can be a law URL, its slug from
    the index, or a title fragment (fuzzy-matched). Returns the file path plus
    a short preview — read the file for the full text.

    Args:
        ref: URL, slug or title fragment of the law.
        refresh: re-download even if a local copy exists.
    """
    try:
        item = resolve(ref)
        if not item["url"] or ".pdf" in item["url"].lower():
            return _out({"error": "این قانون صفحه پست ندارد؛ فقط PDF رسمی دارد",
                         "title": item["title"], "pdf": item["pdf"],
                         "hint": "use iran-laws_pdf with this ref"})
        slug = slugify(item["url"])
        md = LAWS_DIR / f"{slug}.md"
        if refresh or not md.exists():
            title, text = fetch_law_markdown(item["url"])
            if not text.strip():
                raise RuntimeError("empty page")
            LAWS_DIR.mkdir(parents=True, exist_ok=True)
            md.write_text(
                f"# {item['title']}\n\n> منبع: {item['url']}\n\n{text}", encoding="utf-8"
            )
        body = md.read_text(encoding="utf-8")
        return _out({
            "title": item["title"], "url": item["url"], "file": str(md),
            "chars": len(body), "year": item["year"], "pdf": item["pdf"],
            "preview": body[:600],
        })
    except Exception as e:
        return _out({"error": str(e), "ref": ref})


@mcp.tool(name="pdf")
def law_pdf(ref: str, ocr: bool = True, ocr_start_page: int = 1,
            max_ocr_pages: int = 6, refresh: bool = False) -> str:
    """Download a law's official PDF and extract its text.

    Uses pdftotext first; if the PDF is scanned (no text layer), falls back to
    VLM OCR (GLM vision). ref can be a direct .pdf URL or a law reference that
    has a PDF in the index.

    Args:
        ref: .pdf URL or law URL/slug/title.
        ocr: allow VLM OCR when pdftotext yields no usable text.
        ocr_start_page: first page to OCR (for continuing long scans).
        max_ocr_pages: pages per OCR call (long PDFs need repeated calls).
        refresh: re-download the PDF.
    """
    try:
        if ref.lower().endswith(".pdf") or "/wp-content/uploads/" in ref:
            url, item = ref, None
        else:
            item = resolve(ref)
            url = item.get("pdf")
            if not url:
                # fall back to a PDF link embedded in the law page itself
                r = _get(item["url"])
                soup = BeautifulSoup(r.text, "lxml")
                ec = soup.select_one(".entry-content") or soup
                a = ec.select_one("a[href$='.pdf'], a[href*='.pdf']")
                if not a:
                    return _out({"error": "no PDF known for this law", "title": item["title"],
                                 "hint": "the page text may already be enough; use iran-laws_get"})
                url = a["href"]
        slug = slugify(url) if not item else slugify(item["url"] or item["title"])
        return _out(pdf_pipeline(url, slug, ocr, ocr_start_page, max_ocr_pages, refresh))
    except Exception as e:
        return _out({"error": str(e), "ref": ref})


@mcp.tool(name="sync")
def laws_sync(limit: int = 0, years: str = "", pdf_only: bool = False,
              with_pdfs: bool = False, refresh: bool = False, stop: bool = False) -> str:
    """Bulk-download law pages (and optionally their PDFs) in the background.

    Skips laws already on disk, so re-running resumes where it stopped.
    Returns immediately; poll iran-laws_sync_status for progress.

    Args:
        limit: max laws this run (0 = all matching).
        years: comma-separated approval years to restrict to, e.g. "1403,1404".
        pdf_only: only laws that have an official PDF link.
        with_pdfs: also download PDFs (pdftotext only, no OCR).
        refresh: re-download even if files exist.
        stop: ask a running sync to stop after the current law.
    """
    try:
        with _sync_lock:
            if stop:
                _sync["stop"] = True
                return _out({"stopping": True})
            if _sync["running"]:
                return _out({"error": "sync already running", "status": _sync})
            items = load_index()
            ny = {norm(y) for y in years.split(",") if y.strip()}
            if ny:
                items = [i for i in items if norm(i["year"]) in ny]
            if pdf_only:
                items = [i for i in items if i["pdf"]]
            if limit and limit > 0:
                items = items[:limit]
            _sync.update({"running": True, "done": 0, "failed": 0, "total": len(items),
                          "current": "", "started": datetime.now().isoformat(timespec="seconds"),
                          "finished": "", "errors": [], "stop": False, "kind": "laws"})
            threading.Thread(target=_sync_worker, args=(items, with_pdfs, refresh), daemon=True).start()
            return _out({"started": True, "total": len(items), "with_pdfs": with_pdfs,
                         "hint": "poll iran-laws_sync_status"})
    except Exception as e:
        return _out({"error": str(e)})


@mcp.tool(name="sync_status")
def laws_sync_status() -> str:
    """Progress of the background bulk download (iran-laws_sync)."""
    return _out(_sync | {"status_file": str(STATUS_FILE)})


@mcp.tool(name="local")
def laws_local(query: str = "", limit: int = 30) -> str:
    """List laws already downloaded to disk (markdown files).

    Args:
        query: optional substring filter on file (title) names.
        limit: max entries returned.
    """
    try:
        nq = norm(query)
        files = sorted(LAWS_DIR.glob("*.md"), key=lambda p: p.stat().st_size) if LAWS_DIR.exists() else []
        hits = []
        for p in files:
            if nq and nq not in norm(p.stem):
                continue
            hits.append({"slug": p.stem, "chars": p.stat().st_size, "file": str(p)})
        pdfs = sorted(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []
        txts = sorted(TXT_DIR.glob("*.txt")) if TXT_DIR.exists() else []
        return _out({
            "matched": len(hits), "laws_total": len(files),
            "pdfs_total": len(pdfs), "pdf_texts_total": len(txts),
            "items": hits[:limit],
            "note": "smaller files first; read any file directly for full text",
        })
    except Exception as e:
        return _out({"error": str(e)})


# ---------------------------------------------------------------- categories

CATS_FILE = DATA_DIR / "index-cats.json"
CATS_DIR = DATA_DIR / "cats"
OCR_TODO = DATA_DIR / "ocr-remaining.json"

_BASE_CATS = BASE + "/category/%d9%82%d9%88%d8%a7%d9%86%db%8c%d9%86-%d9%88-%d9%85%d8%b5%d9%88%d8%a8%d8%a7%d8%aa"
CATEGORY_URLS = {
    "آرا وحدت رویه": _BASE_CATS + "/%d8%a2%d8%b1%d8%a7-%d9%88%d8%ad%d8%af%d8%aa-%d8%b1%d9%88%db%8c%d9%87/",
    "نظریه‌های مشورتی": _BASE_CATS + "/%d9%86%d8%b8%d8%b1%db%8c%d9%87-%d9%87%d8%a7%db%8c-%d9%85%d8%b4%d9%88%d8%b1%d8%aa%db%8c/",
    "مصوبات": _BASE_CATS + "/%d9%85%d8%b5%d9%88%d8%a8%d8%a7%d8%aa/",
    "طرح و لایحه": _BASE_CATS + "/%d8%b7%d8%b1%d8%ad-%d9%88-%d9%84%d8%a7%db%8c%d8%ad%d9%87/",
    "نظریه‌های رئیس مجلس": _BASE_CATS + "/%d9%86%d8%b8%d8%b1%db%8c%d9%87%d9%87%d8%a7%db%8c-%d8%b1%d8%a6%db%8c%d8%b3-%d9%85%d8%ac%d9%84%d8%b3-%d8%b4%d9%88%d8%b1%d8%a7%db%8c-%d8%a7%d8%b3%d9%84%d8%a7%d9%85/",
    "نشست‌های قضایی": _BASE_CATS + "/%d9%86%d8%b4%d8%b3%d8%aa%d9%87%d8%a7%db%8c-%d9%82%d8%b6%d8%a7%db%8c%db%8c/",
    "آیین‌نامه‌ها": _BASE_CATS + "/%d8%a2%db%8c%db%8c%d9%86-%d9%86%d8%a7%d9%85%d9%87-%d9%87%d8%a7/",
    "بخشنامه‌ها": _BASE_CATS + "/%d8%a8%d8%ae%d8%b4%d9%86%d8%a7%d9%85%d9%87-%d9%87%d8%a7/",
    "آرای دیوان عدالت اداری": _BASE_CATS + "/%d8%b1%d8%a7%db%8c-%d9%87%db%8c%d8%a7%d8%aa-%d8%b9%d9%85%d9%88%d9%85%db%8c-%d8%af%db%8c%d9%88%d8%a7%d9%86-%d8%b9%d8%af%d8%a7%d9%84%d8%aa-%d8%a7%d8%af%d8%a7%d8%b1%db%8c/",
    "رویه قضایی": _BASE_CATS + "/%d8%b1%d9%88%db%8c%d9%87-%d9%82%d8%b6%d8%a7%db%8c%db%8c/",
    "سیاست‌های کلی": _BASE_CATS + "/%d8%b3%db%8c%d8%a7%d8%b3%d8%aa-%d9%87%d8%a7%db%8c-%da%a9%d9%84%db%8c/",
}


def scrape_category(cat: str, url: str, cap_pages: int = 800) -> list[dict]:
    """Walk a WordPress category archive page by page, collecting posts."""
    items, seen = [], set()
    for n in range(1, cap_pages + 1):
        page = url if n == 1 else f"{url.rstrip('/')}/page/{n}/"
        try:
            r = _get(page)
        except Exception:
            break
        soup = BeautifulSoup(r.text, "lxml")
        main = soup.select_one("#main-content") or soup
        new = 0
        for a in main.select(".post-title a[href]"):
            href = a["href"].split("#")[0]
            title = _ws(a.get_text(strip=True))
            if not title or "ekhtebar.ir" not in href or href in seen:
                continue
            seen.add(href)
            items.append({"title": title, "url": href, "cat": cat})
            new += 1
        if new == 0:
            break
        time.sleep(DELAY)
    return items


def _md_pdfs(md: Path) -> list[str]:
    """Resume support: pdf links stored in the markdown header."""
    try:
        head = md.read_text(encoding="utf-8")[:600]
        return re.findall(r"> پی‌دی‌اف: (\S+)", head)
    except Exception:
        return []


def _cat_worker(items: list[dict], ocr: bool, ocr_max_pages: int, refresh: bool):
    try:
        for it in items:
            if _sync["stop"]:
                break
            cat, slug = it["cat"], slugify(it["url"])
            _sync["current"] = f"[{cat}] {it['title']}"
            d = CATS_DIR / cat
            md = d / f"{slug}.md"
            pdfs: list[str] = []
            if refresh or not md.exists():
                try:
                    _, text, pdfs = _fetch_entry(it["url"])
                    if not text.strip():
                        raise RuntimeError("empty page")
                    d.mkdir(parents=True, exist_ok=True)
                    head = f"# {it['title']}\n\n> منبع: {it['url']}\n"
                    head += "".join(f"> پی‌دی‌اف: {p}\n" for p in pdfs)
                    md.write_text(head + f"\n{text}", encoding="utf-8")
                    time.sleep(DELAY)
                except Exception as e:
                    _sync["failed"] += 1
                    if len(_sync["errors"]) < 100:
                        _sync["errors"].append(f"[{cat}] {it['title']}: {e}")
            if md.exists() and not pdfs:
                pdfs = _md_pdfs(md)
            for i, p in enumerate(pdfs[:3], 1):
                pslug = f"{cat}--{slug}" + (f"-{i}" if len(pdfs) > 1 else "")
                try:
                    info = pdf_pipeline(p, pslug, ocr=ocr, ocr_start_page=1,
                                        max_ocr_pages=ocr_max_pages, refresh=refresh)
                    if info.get("pages_remaining"):
                        todo = json.loads(OCR_TODO.read_text(encoding="utf-8")) if OCR_TODO.exists() else []
                        todo.append({"pdf": info["pdf"], "text_file": info["text_file"],
                                     "next_page": ocr_max_pages + 1,
                                     "pages_remaining": info["pages_remaining"]})
                        OCR_TODO.parent.mkdir(parents=True, exist_ok=True)
                        OCR_TODO.write_text(json.dumps(todo, ensure_ascii=False, indent=1), encoding="utf-8")
                    time.sleep(DELAY)
                except Exception as e:
                    _sync["failed"] += 1
                    if len(_sync["errors"]) < 100:
                        _sync["errors"].append(f"pdf [{cat}] {it['title']}: {e}")
            _sync["done"] += 1
            if _sync["done"] % 5 == 0:
                _write_status()
    finally:
        _sync["running"] = False
        _sync["finished"] = datetime.now().isoformat(timespec="seconds")
        _write_status()


@mcp.tool(name="cats_index")
def cats_index(cats: str = "all", refresh: bool = False) -> str:
    """Index the ekhtebar category archives (rulings, advisory opinions, bylaws, ...).

    Walks every /page/N/ of each category and collects post links.
    Args:
        cats: "all" or comma-separated category names, e.g. "آرا وحدت رویه,مصوبات".
        refresh: re-walk even if a cached index exists.
    """
    try:
        if not refresh and CATS_FILE.exists():
            idx = json.loads(CATS_FILE.read_text(encoding="utf-8"))
        else:
            names = list(CATEGORY_URLS) if cats.strip().lower() == "all" \
                else [c.strip() for c in cats.split(",") if c.strip()]
            idx = {}
            for name in names:
                url = CATEGORY_URLS.get(name)
                if not url:
                    raise LookupError(f"unknown category {name!r}; known: {list(CATEGORY_URLS)}")
                idx[name] = scrape_category(name, url)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            CATS_FILE.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
        return _out({"categories": {k: len(v) for k, v in idx.items()},
                     "total": sum(len(v) for v in idx.values()),
                     "index_file": str(CATS_FILE)})
    except Exception as e:
        return _out({"error": str(e)})


@mcp.tool(name="cats_sync")
def cats_sync(cats: str = "all", limit: int = 0, ocr: bool = True,
              ocr_max_pages: int = 50, refresh: bool = False, stop: bool = False) -> str:
    """Bulk-download category posts (rulings, advisory opinions, ...) in the background.

    Saves each post as markdown (data/cats/<cat>/), downloads PDFs linked in the
    posts, extracts text with pdftotext and OCRs scanned ones with the VLM.
    Long scans beyond ocr_max_pages are queued in data/ocr-remaining.json.
    Resumable: skips posts already on disk. Returns immediately; poll iran-laws_sync_status.

    Args:
        cats: "all" or comma-separated category names.
        limit: max posts this run (0 = all).
        ocr: OCR scanned PDFs.
        ocr_max_pages: max OCR pages per PDF (rest is queued in ocr-remaining.json).
        refresh: re-download even if files exist.
        stop: ask a running sync to stop after the current post.
    """
    try:
        with _sync_lock:
            if stop:
                _sync["stop"] = True
                return _out({"stopping": True})
            if _sync["running"]:
                return _out({"error": "sync already running", "status": _sync})
            if not CATS_FILE.exists():
                return _out({"error": "no category index yet; call iran-laws_cats_index first"})
            idx = json.loads(CATS_FILE.read_text(encoding="utf-8"))
            names = list(idx) if cats.strip().lower() == "all" \
                else [c.strip() for c in cats.split(",") if c.strip()]
            items: list[dict] = []
            for n in names:
                items.extend(idx.get(n, []))
            if limit and limit > 0:
                items = items[:limit]
            _sync.update({"running": True, "done": 0, "failed": 0, "total": len(items),
                          "current": "", "started": datetime.now().isoformat(timespec="seconds"),
                          "finished": "", "errors": [], "stop": False, "kind": "categories"})
            threading.Thread(target=_cat_worker, args=(items, ocr, ocr_max_pages, refresh),
                             daemon=True).start()
            return _out({"started": True, "total": len(items), "ocr": ocr,
                         "hint": "poll iran-laws_sync_status"})
    except Exception as e:
        return _out({"error": str(e)})


@mcp.tool(name="cats_local")
def cats_local(query: str = "", limit: int = 30) -> str:
    """List category posts already downloaded (data/cats/<cat>/*.md)."""
    try:
        nq = norm(query)
        per_cat: dict = {}
        items = []
        if CATS_DIR.exists():
            for d in sorted(CATS_DIR.iterdir()):
                if not d.is_dir():
                    continue
                files = sorted(d.glob("*.md"))
                per_cat[d.name] = len(files)
                for p in files:
                    if nq and nq not in norm(p.stem):
                        continue
                    items.append({"cat": d.name, "slug": p.stem,
                                  "chars": p.stat().st_size, "file": str(p)})
        return _out({"per_cat": per_cat, "total": sum(per_cat.values()),
                     "matched": len(items), "items": items[:limit],
                     "ocr_queue": str(OCR_TODO)})
    except Exception as e:
        return _out({"error": str(e)})


if __name__ == "__main__":
    mcp.run(transport="stdio")
