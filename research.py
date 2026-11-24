"""Historical grounding: Wikipedia excerpts -> fact sheet -> accurate narration and image prompts.

The LLM never writes about a period from memory alone. It first reads source excerpts (Wikipedia, plus optional
notes you provide with --notes), distils a structured fact sheet (what people did, wore, ate, used, and which
things did NOT exist yet), and every narration and image prompt is built from that sheet. Network access is
optional: if offline, the app continues with the model's own knowledge and says so.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple

API = "https://en.wikipedia.org/w/api.php"
UA = "SleepDocGenerator/1.0 (local educational video tool)"

# Words that are anachronistic for essentially every pre-modern period (the fact sheet adds period-specific ones).
STATIC_MODERN = ["o'clock", "okay", "plastic", "electric", "electricity", "battery", "telephone", "phone", "television",
                 "photograph", "internet", "computer", "automobile", "minutes", "weekend", "gasoline", "airplane",
                 "aeroplane", "light bulb", "lightbulb", "zipper", "cellphone", "smartphone", "supermarket"]


# --------------------------------------------------------------------------- Wikipedia
def _get(params: dict, timeout: int = 15) -> dict:
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def search_titles(query: str, n: int = 2) -> List[str]:
    d = _get({"action": "query", "list": "search", "srsearch": query, "srlimit": n, "format": "json"})
    return [x["title"] for x in d.get("query", {}).get("search", [])]


def fetch_extract(title: str, chars: int = 2400) -> str:
    d = _get({"action": "query", "prop": "extracts", "explaintext": 1, "exchars": chars, "redirects": 1,
              "titles": title, "format": "json"})
    page = next(iter(d.get("query", {}).get("pages", {}).values()), {})
    return re.sub(r"\n{3,}", "\n\n", page.get("extract", "") or "").strip()


def gather(queries: List[str], per_query: int = 2, max_pages: int = 5, chars: int = 2400) -> Tuple[str, List[Dict]]:
    """Return (combined source text, [{'title','url'}]). Raises on network failure for the first request only."""
    titles: List[str] = []
    for q in queries:
        for t in search_titles(q, per_query):
            if t not in titles:
                titles.append(t)
        if len(titles) >= max_pages:
            break
    blocks, sources = [], []
    for t in titles[:max_pages]:
        try:
            txt = fetch_extract(t, chars)
        except Exception:
            continue
        if len(txt) > 200:
            blocks.append(f"### {t}\n{txt}")
            sources.append({"title": t, "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(t.replace(" ", "_"))})
    return "\n\n".join(blocks), sources


# --------------------------------------------------------------------------- fact sheet helpers
LIST_KEYS = ["daily_life", "clothing", "architecture_objects", "food_drink", "sensory", "anachronisms", "uncertain"]


def normalize_sheet(d: dict, topic: str) -> dict:
    def lst(k, n):
        v = d.get(k, [])
        v = [re.sub(r"\s+", " ", str(x)).strip() for x in (v if isinstance(v, list) else [v]) if str(x).strip()]
        return v[:n]

    sheet = {
        "period": str(d.get("period", "")).strip(),
        "region": str(d.get("region", "")).strip(),
        "visual_anchor": " ".join(str(d.get("visual_anchor", "")).split()[:10]) or topic,
        "setting": str(d.get("setting", "")).strip(),
    }
    caps = {"daily_life": 12, "clothing": 6, "architecture_objects": 8, "food_drink": 7, "sensory": 6,
            "anachronisms": 18, "uncertain": 4}
    for k in LIST_KEYS:
        sheet[k] = lst(k, caps[k])
    return sheet


def sheet_text(sheet: dict) -> str:
    if not sheet:
        return "(no fact sheet available)"
    def bl(k):
        return "; ".join(sheet.get(k, []))
    return (f"Period: {sheet.get('period', '')} | Region: {sheet.get('region', '')}\n"
            f"Setting: {sheet.get('setting', '')}\n"
            f"What people did: {bl('daily_life')}\n"
            f"Clothing: {bl('clothing')}\n"
            f"Buildings, objects, tools, light: {bl('architecture_objects')}\n"
            f"Food and drink: {bl('food_drink')}\n"
            f"Sensory details: {bl('sensory')}\n"
            f"Uncertain or debated: {bl('uncertain')}\n"
            f"MUST NOT APPEAR (did not exist or not used): {bl('anachronisms')}")


def banned_terms(sheet: dict) -> List[str]:
    terms = list(STATIC_MODERN)
    for t in (sheet or {}).get("anachronisms", []):
        t = t.strip().lower()
        if 2 < len(t) <= 30:
            terms.append(t)
    return sorted(set(terms), key=len, reverse=True)


def _banned_regex(terms: List[str]):
    if not terms:
        return None
    return re.compile(r"(?<![\w'])(" + "|".join(re.escape(t) for t in terms) + r")(?:s|es)?(?![\w])", re.I)


def find_anachronisms(text: str, terms: List[str]) -> List[str]:
    rx = _banned_regex(terms)
    return sorted({m.group(1).lower() for m in rx.finditer(text)}) if rx else []


def drop_anachronisms(text: str, terms: List[str]) -> str:
    """Remove sentences that mention a banned term (keeps the text if that would delete too much)."""
    rx = _banned_regex(terms)
    if not rx or not rx.search(text):
        return text
    paras_out = []
    for para in text.split("\n\n"):
        sents = re.split(r"(?<=[.!?])\s+", para)
        kept = [s for s in sents if not rx.search(s)]
        if kept:
            paras_out.append(" ".join(kept))
    out = "\n\n".join(paras_out)
    return out if len(out.split()) >= 0.7 * len(text.split()) else text


def scrub_prompt(text: str, terms: List[str]) -> str:
    rx = _banned_regex(terms)
    return re.sub(r"\s{2,}", " ", rx.sub("", text)).strip(" ,") if rx else text


def compose_image_prompt(sheet: dict, scene: str) -> str:
    """Period anchor FIRST (so even 77-token CLIP models keep it), then the concrete scene, then period details."""
    anchor = (sheet or {}).get("visual_anchor", "")
    scene = scrub_prompt(re.sub(r"\s+", " ", scene).strip().rstrip("."), banned_terms(sheet))
    extras = ", ".join(((sheet or {}).get("clothing", []))[:2] + ((sheet or {}).get("architecture_objects", []))[:2])
    return ", ".join(x for x in (anchor, scene, extras) if x)
