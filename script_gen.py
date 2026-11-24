"""Topic -> title/description/chapters -> narration, using a local instruction-tuned LLM.

Two-phase generation keeps a 7B model reliable:
  1. one small JSON call for the outline (title, description, tags, chapter plan, image prompts)
  2. one plain-text call per chapter for the narration (never asks the model for 2,000 words of JSON)
Every chapter is cached on disk, so an interrupted run resumes at the next missing chapter.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import config
from config import free_memory, log, read_json, write_json
import research
from scanner import DUPLICATE_THRESHOLD, covered_strings, similarity

SYSTEM_OUTLINE = (
    "You are an award-winning documentary writer who creates calm, immersive, historically grounded "
    "bedtime documentaries for YouTube sleep channels. You answer with a single valid JSON object and nothing else."
)
SYSTEM_HISTORIAN = (
    "You are a meticulous historian. You rely on the supplied source excerpts and on well-established scholarship, "
    "you never invent specifics, and you answer with a single valid JSON object and nothing else."
)
SYSTEM_NARRATOR = (
    "You are a gentle, soothing narrator-writer of bedtime history documentaries. You write in the second "
    "person, present tense, slow and sensory, in plain prose with no formatting of any kind."
)


# --------------------------------------------------------------------------- LLM wrapper
class LLM:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        log(f"Loading LLM {model_id} ...")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        device = config.get_device()
        # Put the WHOLE model on the accelerator. device_map="auto" may offload layers to disk on Macs, which is
        # many times slower. Fall back to "auto" only if the model really does not fit.
        maps = [{"": device}, "auto"] if device in ("cuda", "mps") else ["auto"]
        last = None
        for dm in maps:
            for kw in ("dtype", "torch_dtype"):
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=dm, **{kw: "auto"})
                    break
                except TypeError:
                    continue
                except Exception as exc:
                    last = exc
                    free_memory()
                    log(f"  loading with device_map={dm!r} failed ({type(exc).__name__}); trying the next option")
                    break
            else:
                continue
            if hasattr(self, "model"):
                break
        if not hasattr(self, "model"):
            raise last or RuntimeError("could not load the LLM")
        self.model.eval()

    def chat(self, system: str, user: str, max_new_tokens: int, temperature: float = 0.7) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,
                pad_token_id=self.tok.eos_token_id,
            )
        text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    def unload(self) -> None:
        try:
            del self.model
            del self.tok
        except AttributeError:
            pass
        free_memory()


def _load_llm(cfg: config.Settings) -> LLM:
    device = config.get_device()
    model_id = config.pick_llm(cfg, device)
    try:
        return LLM(model_id)
    except Exception as exc:
        free_memory()
        if model_id == config.LLM_SMALL or cfg.llm_model != "auto":
            raise
        log(f"Could not load {model_id} ({type(exc).__name__}: {exc}). Falling back to {config.LLM_SMALL}.")
        return LLM(config.LLM_SMALL)


# --------------------------------------------------------------------------- JSON helpers
def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        chunk = text[m.start():]
        for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
            try:
                obj, _ = dec.raw_decode(candidate)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in model output")


def _clean_line(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().strip('"“”')


def _validate_outline(d: dict, topic_hint: Optional[str]) -> dict:
    title = _clean_line(d.get("title", ""))
    if len(title) < 8:
        raise ValueError("title missing")
    if len(title) > 100:
        title = title[:100].rsplit(" ", 1)[0]
    chapters = d.get("chapters")
    if not isinstance(chapters, list) or len(chapters) < 8:
        raise ValueError("need at least 8 chapters")
    chapters = chapters[:12]
    out: List[dict] = []
    for i, ch in enumerate(chapters, 1):
        if not isinstance(ch, dict):
            raise ValueError("chapter is not an object")
        ct = _clean_line(ch.get("title", ""))
        scene = _clean_line(ch.get("scene", "") or ch.get("image_prompt", ""))
        if not ct or not scene:
            raise ValueError(f"chapter {i} missing title/scene")
        subs = [_clean_line(x) for x in ch.get("sub_scenes", []) if isinstance(x, str) and x.strip()][:4]
        out.append({"title": ct, "summary": _clean_line(ch.get("summary", "")),
                    "activity": _clean_line(ch.get("activity", "")) or _clean_line(ch.get("summary", "")),
                    "scene": scene, "sub_scenes": subs})
    tags = [re.sub(r"[^A-Za-z0-9 ]", "", t).strip() for t in d.get("tags", []) if isinstance(t, str)]
    tags = [t for t in tags if t][:15]
    desc = str(d.get("description", "")).strip() or f"Drift off to sleep as we travel back in time: {title}."
    return {
        "topic": _clean_line(topic_hint or d.get("topic") or title),
        "title": title,
        "description": desc,
        "tags": tags,
        "chapters": out,
    }

def _fallback_outline(topic: str) -> dict:
    """Used only if the LLM fails to produce valid JSON several times in a row."""
    shots = ["wide view of the whole scene", "close detail of everyday objects", "people quietly at work", "a doorway at dusk"]
    stages = [
        ("Settling In", "arriving and getting your bearings"),
        ("First Light", "the quiet hour before dawn and the start of the day"),
        ("The Daily Work", "the labor that fills the hours"),
        ("Food and Fire", "meals, hearths and the warmth of shared food"),
        ("The Faces Around You", "family, neighbors and the social world"),
        ("Belief and Ritual", "faith, customs and festivals"),
        ("Hardship and Comfort", "the dangers and small comforts of the era"),
        ("Evening Gathering", "the community comes together as light fades"),
        ("Stories by the Fire", "songs, stories and shared memory"),
        ("The Long Night", "candles are snuffed and sleep settles in"),
    ]
    chapters = [{"title": t, "summary": f"{x} ({topic})", "activity": x, "scene": f"{x}, warm firelight",
                 "sub_scenes": [f"{x}, {sh}" for sh in shots[:3]]} for t, x in stages]
    return {
        "topic": topic,
        "title": topic,
        "description": f"Settle in and drift off to sleep as we explore {topic}. A calm, relaxing history story to help you unwind.",
        "tags": ["sleep story", "history", "relaxing", "documentary", "bedtime story"],
        "chapters": chapters,
    }

# --------------------------------------------------------------------------- topic + research
def _pick_topic(llm: LLM, cfg: config.Settings, existing, seeds: List[str]) -> str:
    covered = covered_strings(existing)
    seed_txt = "\n".join(f"- {x}" for x in seeds) if seeds else "(none)"
    avoid = "\n".join(f"- {c}" for c in covered[:60]) or "(none)"
    extra = ""
    for attempt in range(3):
        prompt = f"""Choose ONE specific, well-documented historical setting for a calm bedtime documentary about what ordinary life was really like.
Pick one of these seed ideas, or a closely related one (a precise place and period, a real, well-documented society):
{seed_txt}
Never repeat or closely resemble anything already covered:
{avoid}
{extra}
Return ONLY JSON: {{"topic": "specific scenario in one short phrase, max 12 words"}}"""
        try:
            topic = _clean_line(extract_json(llm.chat(SYSTEM_OUTLINE, prompt, max_new_tokens=120, temperature=0.8))["topic"])
        except Exception:
            continue
        if len(topic) < 8:
            continue
        if max((similarity(topic, c) for c in covered), default=0.0) < DUPLICATE_THRESHOLD:
            return topic
        extra = f'"{topic}" is too similar to an existing video. Choose a clearly different setting.'
    return seeds[0] if seeds else "What Life Was Really Like in the Middle Ages"


def _fact_sheet(llm: LLM, topic: str, source_text: str, notes: str) -> dict:
    src = ""
    if notes:
        src += f"USER NOTES (highest priority, always trust over everything else):\n{notes[:5000]}\n\n"
    src += source_text[:11000] if source_text else "(no source excerpts available: rely only on well-established scholarship and list anything uncertain)"
    prompt = f"""Topic: "{topic}"

SOURCE EXCERPTS (prefer these over your memory):
{src}

Build a historical FACT SHEET for writing an accurate documentary and drawing accurate pictures of this exact time and place.
Return ONLY a JSON object with exactly these keys:
{{
  "period": "specific dates, e.g. 'c. 793-1066 CE (Viking Age)'",
  "region": "country/region and kind of community",
  "visual_anchor": "max 8 words naming the period and place for image prompts, e.g. 'Viking Age Scandinavia, tenth century'",
  "setting": "one sentence: the place and who lives there",
  "daily_life": ["10 to 12 short statements of what ordinary people of this period actually did: work, chores, meals, rituals, leisure, by time of day and season"],
  "clothing": ["4 to 6 short items: typical dress, fabrics, footwear, jewelry"],
  "architecture_objects": ["6 to 8 short items: buildings, furniture, tools, containers, light sources and materials typical of the period"],
  "food_drink": ["5 to 7 foods and drinks actually available in this period and region"],
  "sensory": ["4 to 6 sounds, smells and textures true to the setting"],
  "anachronisms": ["12 to 18 things that did NOT exist or were NOT in use here and then, each 1 to 2 words, singular, such as 'clock', 'potato', 'glass window', 'fork'"],
  "uncertain": ["2 to 4 points that are debated or poorly documented"]
}}
Use only facts supported by the excerpts or universally established. Do not invent names, dates or numbers."""
    for _ in range(3):
        try:
            return research.normalize_sheet(extract_json(llm.chat(SYSTEM_HISTORIAN, prompt, max_new_tokens=1500, temperature=0.3)), topic)
        except Exception as exc:
            log(f"Fact sheet attempt unusable ({exc}); retrying")
    return research.normalize_sheet({}, topic)


def _research(llm: LLM, cfg: config.Settings, topic: str) -> dict:
    log(f"Researching: {topic}")
    queries: List[str] = []
    try:
        q = extract_json(llm.chat(SYSTEM_HISTORIAN,
            f'List 4 English Wikipedia article titles that best cover the everyday life, buildings, clothing and food of: "{topic}". '
            'Return ONLY JSON: {"articles": ["Title 1", "Title 2", "Title 3", "Title 4"]}', max_new_tokens=150, temperature=0.2))
        queries = [str(x) for x in q.get("articles", []) if str(x).strip()][:4]
    except Exception:
        pass
    queries = queries + [topic]
    text, sources = "", []
    try:
        text, sources = research.gather(queries)
        log(f"  read {len(sources)} Wikipedia articles: " + ", ".join(x["title"] for x in sources))
    except Exception as exc:
        log(f"  WARNING: could not reach Wikipedia ({type(exc).__name__}). Continuing from the model's own knowledge; "
            "facts will be less reliable. Use --notes FILE to supply your own sources.")
    notes = ""
    if getattr(cfg, "notes", None):
        try:
            notes = open(cfg.notes, encoding="utf-8", errors="ignore").read()
            log(f"  using your notes from {cfg.notes} ({len(notes.split())} words)")
        except OSError as exc:
            log(f"  could not read --notes file: {exc}")
    sheet = _fact_sheet(llm, topic, text, notes)
    log(f"  fact sheet: {sheet['period'] or 'period n/a'} | {len(sheet['daily_life'])} daily-life facts, "
        f"{len(sheet['anachronisms'])} banned anachronisms")
    return {"sheet": sheet, "sources": sources, "grounded": bool(text or notes)}


# --------------------------------------------------------------------------- outline
def _outline_prompt(topic: str, sheet: dict, n: int, extra: str, covered: List[str]) -> str:
    avoid = ("ALREADY COVERED - never resemble:\n" + "\n".join(f"- {c}" for c in covered[:40])) if covered else ""
    return f"""Plan a long, calming bedtime history documentary narrated in second person.
TOPIC (use exactly this subject): "{topic}"
{avoid}
{extra}

HISTORICAL FACT SHEET (every chapter must be built from this; do not contradict it):
{research.sheet_text(sheet)}

Return ONLY a JSON object with exactly this shape:
{{
  "title": "Engaging YouTube-style title that starts with the main keyword, max 90 characters",
  "description": "SEO-friendly YouTube description, 2 short paragraphs, mentions it is a relaxing sleep story with real history. No timestamps.",
  "tags": ["10 to 14 short search tags"],
  "chapters": [
    {{
      "title": "short chapter title",
      "summary": "one sentence on what this chapter covers",
      "activity": "what people of this period are doing in this chapter, taken from the fact sheet",
      "scene": "ONE concrete picture, 18 to 25 words: who (period dress), doing what, where, with which period objects and light. Plain visual description only.",
      "sub_scenes": ["four DIFFERENT concrete pictures from this chapter, 15 to 22 words each: vary place, activity, close-up of objects, wide view"]
    }}
  ]
}}

Rules: exactly {n} chapters that progress gently (arrival -> morning -> work -> food -> people -> beliefs -> hardships -> evening -> night -> sleep).
Each chapter shows DIFFERENT activities drawn from the fact sheet's daily-life list. Scenes must only contain people, clothing, buildings, objects, food and tools that existed in this period and region.
Scenes contain no text or writing, no modern objects, and nothing from the MUST NOT APPEAR list."""

def _make_outline(llm: LLM, cfg: config.Settings, existing, seeds: List[str], topic: str, res: dict) -> dict:
    covered = covered_strings(existing)
    sheet = res["sheet"]
    extra = ""
    for attempt in range(1, 5):
        prompt = _outline_prompt(topic, sheet, config.NUM_CHAPTERS, extra, covered)
        raw = llm.chat(SYSTEM_OUTLINE, prompt, max_new_tokens=3200, temperature=0.7 if attempt < 3 else 0.4)
        try:
            data = _validate_outline(extract_json(raw), topic)
        except Exception as exc:
            log(f"Outline attempt {attempt} unusable ({exc}); retrying")
            continue
        break
    else:
        log("LLM never produced a valid outline; using the built-in generic outline.")
        data = _fallback_outline(topic)
    # build image prompts in code: period anchor + concrete scene + period details, with anachronisms scrubbed
    for ch in data["chapters"]:
        ch["image_prompt"] = research.compose_image_prompt(sheet, ch["scene"])
        ch["sub_scenes"] = [research.compose_image_prompt(sheet, x) for x in ch["sub_scenes"]]
    data["sheet"], data["sources"], data["grounded"] = sheet, res["sources"], res["grounded"]
    return data

# --------------------------------------------------------------------------- narration
def _clean_for(outline: dict, text: str) -> str:
    """Clean LLM output, then drop any sentence that mentions a banned (anachronistic) term."""
    out = clean_narration(text)
    terms = research.banned_terms(outline.get("sheet", {}))
    flagged = research.find_anachronisms(out, terms)
    if flagged:
        log(f"  removed sentences with anachronisms: {', '.join(flagged)}")
        out = research.drop_anachronisms(out, terms)
    return out


def _word_count(t: str) -> int:
    return len(t.split())


def clean_narration(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = text.replace("—", ", ").replace("–", ", ").replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", text)  # stage directions
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if re.match(r"^(#+\s|chapter\s+\w+\s*[:.-]|\*\*[^*]+\*\*\s*$|[-*•]\s|\d+[.)]\s)", s, re.I) and len(s.split()) < 14:
            continue
        lines.append(re.sub(r"[*_`>#]", "", ln))
    text = "\n".join(lines).strip()
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if paras and re.match(r"^(sure|certainly|here is|here's|of course)\b", paras[0], re.I):
        paras = paras[1:]
    if len(paras) == 1 and _word_count(paras[0]) > 80:  # model wrote one block: re-split into paragraphs
        sents = re.split(r"(?<=[.!?])\s+", paras[0])
        paras = [" ".join(sents[i:i + 4]) for i in range(0, len(sents), 4)]
    if paras and paras[-1] and paras[-1][-1] not in '.!?"':  # cut off by the token limit
        cut = max(paras[-1].rfind(". "), paras[-1].rfind("? "), paras[-1].rfind("! "))
        paras[-1] = paras[-1][: cut + 1] if cut > 0 else paras[-1] + "."
    return "\n\n".join(p for p in paras if p)


def _role(i: int, n: int) -> str:
    if i == 0:
        return ("This is the OPENING chapter. Begin with a single intriguing hook sentence (a surprising, true detail or question about "
                "daily life in this world) within the first ten words. Then, in two or three gentle sentences, welcome the listener, "
                "invite them to dim the screen and get comfortable, and mention they can say in the comments where they are listening "
                "from tonight. Then let their breathing slow and gently begin the journey into the past.")
    if i == n - 1:
        return ("This is the FINAL chapter: slow the pace even more, soften every image, let the world fall quiet, and end with a "
                "tender goodnight that carries the listener into sleep.")
    return "This is a MIDDLE chapter: continue seamlessly from the previous one. Do not greet the listener and do not wrap up the story."


def _chapter_prompt(outline: dict, i: int, words: int, prev_tail: str) -> str:
    chs = outline["chapters"]
    plan = "\n".join(f"{k + 1}. {c['title']} - {c['summary']}" for k, c in enumerate(chs))
    prev = f'\nThe previous chapter ended with: "{prev_tail}"\n' if prev_tail else ""
    return f"""Documentary: "{outline['title']}" ({outline['topic']})
Full chapter plan:
{plan}

HISTORICAL FACT SHEET (what people really did and used; stay consistent with it):
{research.sheet_text(outline.get('sheet', {}))}

Write the narration for chapter {i + 1}: "{chs[i]['title']}" - {chs[i]['summary']}
In this chapter people are shown: {chs[i].get('activity', '')}
{_role(i, len(chs))}{prev}
Requirements:
- About {words} words (between {int(words * 0.9)} and {int(words * 1.15)}).
- Second person, present tense ("You step through the doorway..."), calm, slow, deeply sensory: warm light, soft sounds, smells, textures, temperature.
- 2 to 4 paragraphs separated by a blank line. Short and medium sentences. Gentle rhythm, no cliffhangers, nothing startling.
- ACCURACY: use only details consistent with the fact sheet. Never mention anything from the MUST NOT APPEAR list. Do not invent names, exact dates, numbers, statistics or quotations. For uncertain points use hedges such as "often", "most likely" or "historians believe". Teach one or two real facts about how people lived in this chapter.
- No graphic violence. Write all numbers and years in words. No modern references, and no modern units such as minutes.
- Output ONLY the narration text: no title, no headings, no lists, no markdown, no stage directions, no sound-effect notes."""

def _extend_prompt(outline: dict, i: int, so_far: str, need: int) -> str:
    closing = "" if i < len(outline["chapters"]) - 1 else " You may end with a gentle goodnight."
    return f"""Here is the narration so far for chapter "{outline['chapters'][i]['title']}" of "{outline['title']}":

{so_far}

Fact sheet to stay consistent with:
{research.sheet_text(outline.get('sheet', {}))}

Continue directly from where it stops with roughly {need} more words in exactly the same calm second-person present-tense voice.
Add NEW sensory detail and real historical texture from the fact sheet; do not repeat anything already said, invent nothing, and never mention anything from the MUST NOT APPEAR list. Do not conclude the chapter or say goodnight.{closing}
Output ONLY the new continuation text, plain prose, paragraphs separated by a blank line."""

def _narrate_chapter(llm: LLM, outline: dict, i: int, words: int, prev_tail: str) -> str:
    raw = llm.chat(SYSTEM_NARRATOR, _chapter_prompt(outline, i, words, prev_tail), max_new_tokens=int(words * 2.0) + 150, temperature=0.7)
    text = _clean_for(outline, raw)
    for _ in range(2):
        if _word_count(text) >= 0.75 * words:
            break
        more = _clean_for(outline, llm.chat(SYSTEM_NARRATOR, _extend_prompt(outline, i, text, words - _word_count(text)),
                                        max_new_tokens=int(words * 1.4) + 100, temperature=0.7))
        more = _cap_words(more, int((words - _word_count(text)) * 1.25) + 10)
        text = (text + "\n\n" + more).strip()
    if not text:
        text = f"{outline['chapters'][i]['summary']}. Take a slow breath, and let the scene settle around you."
    return text


def _cap_words(text: str, max_words: int) -> str:
    """Trim to at most max_words, ending on a sentence boundary."""
    words = text.split()
    if len(words) <= max_words:
        return text
    cut = " ".join(words[:max_words])
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "), cut.rfind('." '))
    return cut[: end + 1] if end > len(cut) * 0.5 else cut


def _tail(text: str, n: int = 45) -> str:
    return " ".join(text.split()[-n:])


def _write_narration(llm: LLM, cfg: config.Settings, outline: dict, cache) -> dict:
    chs = outline["chapters"]
    n = len(chs)
    per = cfg.target_words // n
    parts = cache / "script_parts"
    parts.mkdir(parents=True, exist_ok=True)
    texts: List[str] = []
    for i in range(n):
        f = parts / f"ch{i + 1:02d}.txt"
        if f.exists() and f.read_text(encoding="utf-8").strip():
            texts.append(f.read_text(encoding="utf-8"))
            log(f"Chapter {i + 1}/{n}: loaded from cache ({_word_count(texts[-1])} words)")
            continue
        log(f"Writing chapter {i + 1}/{n}: {chs[i]['title']}")
        text = _narrate_chapter(llm, outline, i, per, _tail(texts[-1]) if texts else "")
        f.write_text(text, encoding="utf-8")
        texts.append(text)
        log(f"  -> {_word_count(text)} words")

    min_words = max(1500, int(cfg.target_words * 0.92))
    for rnd in range(3):
        total = sum(_word_count(t) for t in texts)
        if total >= min_words:
            break
        log(f"Narration is {total} words (< {min_words}); extending the shortest chapters (round {rnd + 1})")
        order = sorted(range(n - 1), key=lambda k: _word_count(texts[k]))
        for k in order[:4]:
            need = max(60, per - _word_count(texts[k]) + 40)
            more = _clean_for(outline, llm.chat(SYSTEM_NARRATOR, _extend_prompt(outline, k, texts[k], need),
                                            max_new_tokens=int(need * 1.6) + 100))
            more = _cap_words(more, int(need * 1.25))
            if more:
                texts[k] = texts[k] + "\n\n" + more
                (parts / f"ch{k + 1:02d}.txt").write_text(texts[k], encoding="utf-8")

    unit_data = _write_units(llm, cfg, outline, texts, cache)
    return {
        "topic": outline["topic"],
        "title": outline["title"],
        "description": outline["description"],
        "tags": outline["tags"],
        "sheet": outline.get("sheet", {}),
        "sources": outline.get("sources", []),
        "grounded": outline.get("grounded", False),
        "chapters": [
            {"title": c["title"], "narration": t, "image_prompt": c["image_prompt"], "sub_scenes": c["sub_scenes"],
             "sentences": u["sentences"], "units": u["units"]}
            for c, t, u in zip(chs, texts, unit_data)
        ],
        "word_count": sum(_word_count(t) for t in texts),
    }


# --------------------------------------------------------------------------- per-passage image prompts
SYSTEM_ILLUSTRATOR = (
    "You are an art director for a historical documentary. You write precise, period-accurate image prompts and "
    "answer with a single valid JSON object and nothing else."
)
VARIANTS = ["wide view of the whole scene", "medium shot of people at their task", "close-up of hands and tools",
            "detail of food and vessels", "interior seen through a doorway", "view across the settlement at dusk"]


def _unit_request(outline: dict, ci: int, passages: List[str], prev: str) -> str:
    ch, sheet = outline["chapters"][ci], outline.get("sheet", {})
    j = lambda k, n: "; ".join(sheet.get(k, [])[:n])
    items = "\n".join(f"{k + 1}. {p}" for k, p in enumerate(passages))
    prev_line = f"Previous image: {prev}\n" if prev else ""
    return f"""Illustrating the documentary "{outline['title']}" ({sheet.get('period', '')}, {sheet.get('region', '')}).
Chapter: "{ch['title']}" - {ch.get('activity', '')}
Period-accurate details you may use. Clothing: {j('clothing', 4)}. Buildings, objects, tools, light: {j('architecture_objects', 6)}. Food: {j('food_drink', 4)}.
NEVER show: {j('anachronisms', 14)}.
{prev_line}
Write ONE image prompt for each narration passage below. Each prompt must show exactly what its passage describes (place, people and what they are doing, objects, light) using only period-accurate clothing, buildings, tools and food. 16 to 22 words, plain visual description, nothing written in the picture, no modern objects, people seen from behind or at a distance. Consecutive prompts must look clearly different: vary wide view, medium view and close-up of objects, hands, food or tools.

Passages:
{items}

Return ONLY JSON: {{"prompts": ["prompt for passage 1", "... exactly {len(passages)} items"]}}"""


def _write_units(llm: LLM, cfg: config.Settings, outline: dict, texts: List[str], cache) -> List[dict]:
    import shots

    target = shots.target_words_for(cfg.image_interval)
    sheet = outline.get("sheet", {})
    res: List[dict] = []
    total = 0
    for ci, text in enumerate(texts):
        f = cache / "script_parts" / f"units_{ci + 1:02d}.json"
        sents = shots.split_sentences(text)
        cached = None
        if f.exists():
            try:
                cached = read_json(f)
                if cached["target"] != target or cached["sentences"] != sents:
                    cached = None
            except Exception:
                cached = None
        if cached:
            res.append({"sentences": sents, "units": cached["units"]})
            total += len(cached["units"])
            continue
        units = shots.make_units(sents, target)
        passages = [shots.unit_text(sents, u) for u in units]
        scenes: List[str] = []
        for b in range(0, len(passages), 6):
            batch = passages[b:b + 6]
            got: List[str] = []
            for _ in range(2):
                try:
                    arr = extract_json(llm.chat(SYSTEM_ILLUSTRATOR, _unit_request(outline, ci, batch, scenes[-1] if scenes else ""),
                                                max_new_tokens=130 * len(batch) + 100, temperature=0.6)).get("prompts", [])
                    arr = [_clean_line(x) for x in arr if isinstance(x, str) and len(x.split()) >= 5]
                    if len(arr) == len(batch):
                        got = arr
                        break
                except Exception:
                    pass
            if not got:   # model failed twice: fall back to the chapter scene with a varied framing
                base = outline["chapters"][ci]["scene"]
                got = [f"{base}, {VARIANTS[(b + k) % len(VARIANTS)]}" for k in range(len(batch))]
                log(f"  chapter {ci + 1}: fell back to generic scenes for passages {b + 1}-{b + len(batch)}")
            scenes += got
        for u, sc in zip(units, scenes):
            u["prompt"] = research.compose_image_prompt(sheet, sc)
        write_json(f, {"target": target, "sentences": sents, "units": units})
        res.append({"sentences": sents, "units": units})
        total += len(units)
        log(f"Image prompts: chapter {ci + 1}/{len(texts)} -> {len(units)} images")
    log(f"{total} narration-matched images planned (one about every {cfg.image_interval:.0f}s)")
    return res


# --------------------------------------------------------------------------- public entry
def generate_script(cfg: config.Settings, existing, seeds: List[str], project) -> Dict:
    """Return the full script dict. `project` creates the output folder as soon as a title exists."""
    if project.folder and (project.cache / "script.json").exists():
        log("Script found in cache - skipping LLM stage")
        return read_json(project.cache / "script.json")

    llm = _load_llm(cfg)
    try:
        outline = None
        if project.folder and (project.cache / "outline.json").exists():
            outline = read_json(project.cache / "outline.json")
            log(f"Resuming outline: {outline['title']}")
        if outline is None:
            topic = cfg.topic or _pick_topic(llm, cfg, existing, seeds)
            outline = _make_outline(llm, cfg, existing, seeds, topic, _research(llm, cfg, topic))
            project.create(cfg.output_dir, outline["title"], outline["topic"])
            write_json(project.cache / "outline.json", outline)
            log(f"Title: {outline['title']}")
        script = _write_narration(llm, cfg, outline, project.cache)
    finally:
        llm.unload()
        del llm
        free_memory()
    write_json(project.cache / "script.json", script)
    log(f"Script complete: {script['word_count']} words, {len(script['chapters'])} chapters")
    return script
