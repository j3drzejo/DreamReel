"""Duplicate prevention: find existing video projects and pick fresh topics."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import config
from config import BUILD_CACHE, read_json

DUPLICATE_THRESHOLD = 0.55

FALLBACK_TOPICS: List[str] = [
    "Life as a Roman Legionary on Hadrian's Wall",
    "A Night Inside a 14th-Century Medieval Monastery",
    "How Ancient Babylonians Lived",
    "Life Aboard an 18th-Century Sailing Ship",
    "What Was Life Actually Like in the Middle Ages",
    "A Winter Evening in a Viking Longhouse",
    "Daily Life Along the Nile in Ancient Egypt",
    "A Night in a Victorian London Townhouse",
    "Life in a Medieval Castle Kitchen",
    "Surviving the Winter in a Stone Age Settlement",
    "An Ordinary Day in Ancient Athens",
    "A Caravan Journey Along the Silk Road",
    "Life in a Samurai-Era Japanese Village",
    "An Evening in the Bazaars of Ottoman Istanbul",
    "A Night in a Pompeii Household Before Vesuvius",
    "Life in a Pioneer Cabin on the American Frontier",
    "A Night in a Tudor England Manor House",
    "Life Inside a Renaissance Florence Artist's Workshop",
    "Life in a Medieval Highland Scottish Village",
    "A Night in an Ancient Roman Bathhouse",
]

_STOP = set(
    "a an the of in on at to for and or with from by is was were are be been it its this that these those "
    "what how why when who did do does really actually like life living live lived lives night day days "
    "inside through among about into over under during before after your you people person world old era "
    "evening ordinary journey".split()
)


def _tokens(text: str) -> set:
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower().replace("'", "")):
        if w in _STOP or len(w) < 3:
            continue
        if w.endswith("s") and len(w) > 4:
            w = w[:-1]
        out.add(w)
    return out


def similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return max(inter / len(ta | tb), 0.8 * inter / min(len(ta), len(tb)))


@dataclass
class Existing:
    title: str
    topic: str
    path: Path
    complete: bool


def _parse_info(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    t = re.search(r'^title:\s*"(.*)"\s*$', text, re.M)
    p = re.search(r'^topic:\s*"(.*)"\s*$', text, re.M)
    return (t.group(1) if t else path.parent.name), (p.group(1) if p else "")


def scan_existing(roots: Iterable[Path], exclude: Optional[Path] = None) -> List[Existing]:
    """Look at every immediate sub-folder (and the root itself) of each root for info.txt
    or an unfinished project (.build_cache/meta.json)."""
    seen, found = set(), []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        candidates = [root] + [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        for d in candidates:
            try:
                key = d.resolve()
            except OSError:
                continue
            if key in seen or (exclude and key == Path(exclude).resolve()):
                continue
            info = d / "info.txt"
            meta = d / BUILD_CACHE / "meta.json"
            if info.is_file():
                title, topic = _parse_info(info)
                found.append(Existing(title, topic, d, True))
                seen.add(key)
            elif meta.is_file():
                try:
                    m = read_json(meta)
                    found.append(Existing(m.get("title", d.name), m.get("topic", ""), d, False))
                    seen.add(key)
                except Exception:
                    pass
    return found


def covered_strings(existing: List[Existing]) -> List[str]:
    out: List[str] = []
    for e in existing:
        for s in (e.title, e.topic):
            if s and s not in out:
                out.append(s)
    return out


def pick_seed_candidates(existing: List[Existing], k: int = 6) -> List[str]:
    covered = covered_strings(existing)
    fresh = [t for t in FALLBACK_TOPICS if all(similarity(t, c) < DUPLICATE_THRESHOLD for c in covered)]
    random.shuffle(fresh)
    return fresh[:k]


def find_incomplete_project(roots: Iterable[Path], topic: Optional[str]) -> Optional[Path]:
    """Return the newest unfinished project (has an outline, no info.txt) to resume.
    If a topic was requested, only a matching project qualifies."""
    best: Optional[tuple[float, Path]] = None
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for d in root.iterdir():
            outline = d / BUILD_CACHE / "outline.json"
            if not (d.is_dir() and outline.is_file()) or (d / "info.txt").exists():
                continue
            if topic:
                try:
                    m = read_json(d / BUILD_CACHE / "meta.json")
                except Exception:
                    continue
                if similarity(topic, m.get("topic", "") + " " + m.get("title", "")) < 0.5:
                    continue
            mt = outline.stat().st_mtime
            if best is None or mt > best[0]:
                best = (mt, d)
    return best[1] if best else None
