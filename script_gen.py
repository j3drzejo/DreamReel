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
from scanner import DUPLICATE_THRESHOLD, covered_strings, similarity

SYSTEM_OUTLINE = (
    "You are an award-winning documentary writer who creates calm, immersive, historically grounded "
    "bedtime documentaries for YouTube sleep channels. You answer with a single valid JSON object and nothing else."
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
        ct, ip = _clean_line(ch.get("title", "")), _clean_line(ch.get("image_prompt", ""))
        if not ct or not ip:
            raise ValueError(f"chapter {i} missing title/image_prompt")
        subs = [_clean_line(s) for s in ch.get("sub_scenes", []) if isinstance(s, str) and s.strip()][:4]
        out.append({"title": ct, "summary": _clean_line(ch.get("summary", "")), "image_prompt": ip, "sub_scenes": subs})
    tags = [re.sub(r"[^A-Za-z0-9 ]", "", t).strip() for t in d.get("tags", []) if isinstance(t, str)]
    tags = [t for t in tags if t][:15]
    desc = str(d.get("description", "")).strip() or f"Drift off to sleep as we travel back in time: {title}."
    return {
        "topic": _clean_line(d.get("topic") or topic_hint or title),
        "title": title,
        "description": desc,
        "tags": tags,
        "chapters": out,
    }


def _fallback_outline(topic: str) -> dict:
    """Used only if the LLM fails to produce valid JSON several times in a row."""
    shots = ["wide establishing view", "close detail of everyday objects", "people quietly at work", "a doorway at dusk"]
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
    chapters = []
    for t, s in stages:
        base = f"Cinematic oil painting, {topic}, {s}, warm candlelight, historically accurate, highly detailed, 16:9"
        chapters.append(
            {"title": t, "summary": f"{s} ({topic})", "image_prompt": base, "sub_scenes": [f"{base}, {x}" for x in shots[:3]]}
        )
    return {
        "topic": topic,
        "title": topic,
        "description": f"Settle in and drift off to sleep as we explore {topic}. A calm, relaxing history story to help you unwind.",
        "tags": ["sleep story", "history", "relaxing", "documentary", "bedtime story"],
        "chapters": chapters,
    }


# --------------------------------------------------------------------------- outline
def _outline_prompt(topic: Optional[str], seeds: List[str], covered: List[str], n: int, extra: str) -> str:
    if topic:
        subject = f'TOPIC (use exactly this subject): "{topic}"'
    elif seeds:
        subject = (
            "TOPIC: choose ONE of these seed ideas (or invent a closely related new one in the same spirit):\n"
            + "\n".join(f"- {s}" for s in seeds)
        )
    else:
        subject = "TOPIC: invent a fresh, specific historical 'what was life really like' scenario (any era or culture)."
    avoid = ("ALREADY COVERED - never repeat or closely resemble any of these:\n" + "\n".join(f"- {c}" for c in covered[:60])) if covered else ""
    return f"""Plan a long, calming bedtime history documentary narrated in second person.
{subject}
{avoid}
{extra}

Return ONLY a JSON object with exactly this shape:
{{
  "topic": "the specific scenario in one short phrase",
  "title": "Engaging YouTube-style title, max 90 characters",
  "description": "SEO-friendly YouTube description, 2 short paragraphs, mentions it is a relaxing sleep story with history facts. No timestamps.",
  "tags": ["10 to 14 short search tags"],
  "chapters": [
    {{
      "title": "short chapter title",
      "summary": "one sentence on what this chapter covers",
      "image_prompt": "detailed, historically grounded visual prompt: cinematic oil painting style, specific scene, lighting, objects, 16:9 widescreen, highly detailed",
      "sub_scenes": ["three different visual scenes from within this chapter, each a full image prompt"]
    }}
  ]
}}

Rules: exactly {n} chapters that progress gently (arrival -> morning -> work -> food -> people -> beliefs -> hardships -> evening -> night -> sleep).
Image prompts must be historically accurate, contain no text/writing, no modern objects, and no people's faces in extreme close-up."""


def _make_outline(llm: LLM, cfg: config.Settings, existing, seeds: List[str]) -> dict:
    covered = covered_strings(existing)
    extra = ""
    for attempt in range(1, 5):
        prompt = _outline_prompt(cfg.topic, seeds, covered, config.NUM_CHAPTERS, extra)
        raw = llm.chat(SYSTEM_OUTLINE, prompt, max_new_tokens=2800, temperature=0.8 if attempt < 3 else 0.5)
        try:
            data = _validate_outline(extract_json(raw), cfg.topic)
        except Exception as exc:
            log(f"Outline attempt {attempt} unusable ({exc}); retrying")
            continue
        clash = max((similarity(data["title"] + " " + data["topic"], c) for c in covered), default=0.0)
        if clash >= DUPLICATE_THRESHOLD:
            if cfg.topic:
                log("Note: the requested topic resembles an existing video; continuing because you asked for it.")
            elif attempt < 4:
                log(f"Outline attempt {attempt} too similar to an existing video; asking for a different one")
                extra = f'Your previous idea "{data["title"]}" was too similar to an existing video. Choose a clearly different subject.'
                continue
        return data
    log("LLM never produced a valid outline; using the built-in generic outline.")
    topic = cfg.topic or (seeds[0] if seeds else "What Life Was Really Like in the Middle Ages")
    return _fallback_outline(topic)


# --------------------------------------------------------------------------- narration
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

Write the narration for chapter {i + 1}: "{chs[i]['title']}" - {chs[i]['summary']}
{_role(i, len(chs))}{prev}
Requirements:
- About {words} words (between {int(words * 0.9)} and {int(words * 1.15)}).
- Second person, present tense ("You step through the doorway..."), calm, slow, deeply sensory: warm light, soft sounds, smells, textures, temperature.
- 2 to 4 paragraphs separated by a blank line. Short and medium sentences. Gentle rhythm, no cliffhangers, nothing startling.
- Weave in genuine, plausible historical detail about daily life; stay historically grounded and avoid inventing precise statistics. No graphic violence.
- Write all numbers and years in words. No modern references.
- Output ONLY the narration text: no title, no headings, no lists, no markdown, no stage directions, no sound-effect notes."""


def _extend_prompt(outline: dict, i: int, so_far: str, need: int) -> str:
    closing = "" if i < len(outline["chapters"]) - 1 else " You may end with a gentle goodnight."
    return f"""Here is the narration so far for chapter "{outline['chapters'][i]['title']}" of "{outline['title']}":

{so_far}

Continue directly from where it stops with roughly {need} more words in exactly the same calm second-person present-tense voice.
Add NEW sensory detail and historical texture; do not repeat anything already said. Do not conclude the chapter or say goodnight.{closing}
Output ONLY the new continuation text, plain prose, paragraphs separated by a blank line."""


def _narrate_chapter(llm: LLM, outline: dict, i: int, words: int, prev_tail: str) -> str:
    raw = llm.chat(SYSTEM_NARRATOR, _chapter_prompt(outline, i, words, prev_tail), max_new_tokens=int(words * 2.0) + 150, temperature=0.7)
    text = clean_narration(raw)
    for _ in range(2):
        if _word_count(text) >= 0.75 * words:
            break
        more = clean_narration(llm.chat(SYSTEM_NARRATOR, _extend_prompt(outline, i, text, words - _word_count(text)),
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
            more = clean_narration(llm.chat(SYSTEM_NARRATOR, _extend_prompt(outline, k, texts[k], need),
                                            max_new_tokens=int(need * 1.6) + 100))
            more = _cap_words(more, int(need * 1.25))
            if more:
                texts[k] = texts[k] + "\n\n" + more
                (parts / f"ch{k + 1:02d}.txt").write_text(texts[k], encoding="utf-8")

    return {
        "topic": outline["topic"],
        "title": outline["title"],
        "description": outline["description"],
        "tags": outline["tags"],
        "chapters": [
            {"title": c["title"], "narration": t, "image_prompt": c["image_prompt"], "sub_scenes": c["sub_scenes"]}
            for c, t in zip(chs, texts)
        ],
        "word_count": sum(_word_count(t) for t in texts),
    }


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
            outline = _make_outline(llm, cfg, existing, seeds)
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
