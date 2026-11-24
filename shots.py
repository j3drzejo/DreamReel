"""Narration -> sentences -> image units -> timed image plan.

Every image is generated from the narration passage being spoken at that moment, so the picture always matches the
words. Passages are ~`interval` seconds of speech (default 6 s); their on-screen times come from the real TTS timing.
"""
from __future__ import annotations

import re
from typing import Dict, List

from config import SPEECH_WPS


def split_sentences(narration: str, max_words: int = 19) -> List[Dict]:
    """[{'text','para'}] in reading order. Very long sentences are split at commas/semicolons."""
    out: List[Dict] = []
    paras = [p for p in re.split(r"\n\s*\n", narration) if p.strip()]   # same split as tts_gen
    for pi, para in enumerate(paras):
        para = re.sub(r"\s+", " ", para).strip()
        for s in re.split(r'(?<=[.!?])["\')\]]*\s+(?=[A-Z"\'(])', para):
            s = s.strip()
            if not s:
                continue
            if len(s.split()) > max_words:
                clauses = [c.strip() for c in re.split(r"(?<=[,;:])\s+", s) if c.strip()]
                merged: List[str] = []
                for c in clauses:
                    if merged and len(merged[-1].split()) < 7:
                        merged[-1] += " " + c
                    else:
                        merged.append(c)
                if len(merged) > 1 and len(merged[-1].split()) < 5:
                    merged[-2] += " " + merged.pop()
                out += [{"text": c, "para": pi} for c in merged]
            else:
                out.append({"text": s, "para": pi})
    return out


def make_units(sents: List[Dict], target_words: int) -> List[Dict]:
    """Group consecutive sentences into units whose word count is as close to target_words as possible
    (never much above 1.4x, so an image stays on screen for roughly 4-8 s): [{'first','last'}] (last exclusive)."""
    units, first, words = [], 0, 0
    for i, s in enumerate(sents):
        w = len(s["text"].split())
        if words > 0:
            further = abs(words + w - target_words) > abs(words - target_words)
            too_long = words + w > 1.3 * target_words and words >= 0.6 * target_words
            new_para = s["para"] != sents[first]["para"] and words >= 0.5 * target_words   # don't straddle a pause
            if further or too_long or new_para:
                units.append({"first": first, "last": i})
                first, words = i, 0
        words += w
    if first < len(sents):
        if units and words < 0.5 * target_words and sents[first]["para"] == sents[units[-1]["last"] - 1]["para"]:
            units[-1]["last"] = len(sents)       # a tiny tail joins the previous image
        else:
            units.append({"first": first, "last": len(sents)})
    return units


def unit_text(sents: List[Dict], u: Dict) -> str:
    return " ".join(s["text"] for s in sents[u["first"]:u["last"]])


def sentence_times(sents: List[Dict], ch_tl: Dict) -> List[tuple]:
    """Absolute (t0, t1) per sentence: proportional to characters inside each paragraph's real audio span."""
    paras = ch_tl.get("paragraphs")
    npara = (max((s["para"] for s in sents), default=-1) + 1)
    if not paras or len(paras) < npara:   # no TTS marks (old cache): spread over the whole chapter
        span = [ch_tl["voice_start"], ch_tl["voice_end"]]
        paras = [span] * npara
        whole = True
    else:
        whole = False
    times: List[tuple] = [(0.0, 0.0)] * len(sents)
    groups: Dict[int, List[int]] = {}
    for i, s in enumerate(sents):
        groups.setdefault(0 if whole else s["para"], []).append(i)
    for p, idxs in groups.items():
        a, b = paras[p]
        tot = sum(len(sents[i]["text"]) for i in idxs) or 1
        pos = 0
        for i in idxs:
            n = len(sents[i]["text"])
            times[i] = (a + (b - a) * pos / tot, a + (b - a) * (pos + n) / tot)
            pos += n
    return times


def build_unit_plan(script: Dict, timeline: Dict) -> List[Dict]:
    """One image segment per unit: start = when its first sentence is spoken, end = start of the next image."""
    segs: List[Dict] = []
    for ci, (ch, tl) in enumerate(zip(script["chapters"], timeline["chapters"])):
        st = sentence_times(ch["sentences"], tl)
        for ui, u in enumerate(ch["units"]):
            segs.append({"chapter": ci, "sub": ui, "start": st[u["first"]][0], "prompt": u.get("prompt", ""),
                         "file": f"ch{ci + 1:02d}_{ui + 1:03d}.png"})
    segs[0]["start"] = 0.0
    for k in range(1, len(segs)):
        segs[k]["start"] = max(segs[k]["start"], segs[k - 1]["start"] + 0.5)   # strictly increasing
    for k in range(len(segs) - 1):
        segs[k]["end"] = segs[k + 1]["start"]
    segs[-1]["end"] = timeline["total_duration"]
    return segs


def target_words_for(interval_s: float) -> int:
    return max(6, int(round(interval_s * SPEECH_WPS)))
