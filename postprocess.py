from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()  # no-op si déjà chargé
import re
import srt

import os, time, math
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any, Tuple, Callable

from subtitles import read_srt
from utils import chunks_of_text  # on garde pour fallback, mais on introduit un chunker meilleur
from prompts import NATURALIZE_PROMPT, TRANSLATE_PROMPT  # on va aussi les renforcer plus bas

# ---------- Config ----------
DEFAULT_OPENAI_MODEL = os.getenv("POSTPROCESS_OPENAI_MODEL", "gpt-4o")
POSTPROCESS_BACKEND = (os.getenv("POSTPROCESS_BACKEND", "openai") or "openai").lower()

# ---------- Helpers ----------

def _safe_len(s: str) -> int:
    return 0 if s is None else len(s)

def _retry(fn: Callable[[], str], tries: int = 4, base_delay: float = 1.0) -> str:
    last_err = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(base_delay * (2 ** i))
    raise last_err  # type: ignore


def _openai_complete(prompt: str, model: Optional[str] = None, temperature: float = 0.2, api_key: Optional[str] = None) -> str:
    """
    Appel OpenAI robuste avec retry et modèle configurable.
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    use_model = model or DEFAULT_OPENAI_MODEL

    # 🔧 Certains modèles (ex: gpt-4o-mini, gpt-5-mini/nano) n'acceptent pas de temperature ≠ 1
    forced_temp = temperature
    if any(name in use_model for name in ["gpt-4o-mini", "gpt-5-mini", "gpt-5-nano"]):
        forced_temp = 1

    def _call() -> str:
        resp = client.chat.completions.create(
            model=use_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=forced_temp,
        )
        return resp.choices[0].message.content.strip()  # type: ignore

    return _retry(_call)


def _marian_translate(text: str, target_lang: str) -> str:
    """
    Basique : tente d'auto-sélectionner un modèle opus-mt. Idéal FR<->EN.
    """
    from transformers import MarianMTModel, MarianTokenizer

    # Déduction simple de la direction (améliorable selon ton use-case)
    # Si la cible est FR -> en-fr, sinon fr-en comme fallback.
    if target_lang.lower().startswith("fr"):
        model_name = "Helsinki-NLP/opus-mt-en-fr"
    elif target_lang.lower().startswith("en"):
        model_name = "Helsinki-NLP/opus-mt-fr-en"
    else:
        # Fallback générique fr->en (à adapter si multi-langues complexes)
        model_name = "Helsinki-NLP/opus-mt-fr-en"

    tok = MarianTokenizer.from_pretrained(model_name)
    mod = MarianMTModel.from_pretrained(model_name)
    batch = tok([text], return_tensors="pt", padding=True, truncation=True)
    gen = mod.generate(**batch, max_new_tokens=2000)
    return tok.batch_decode(gen, skip_special_tokens=True)[0]

def _translate_line(text: str, target_lang: str, model: Optional[str], api_key: Optional[str]) -> str:
    # garde la concision, pas de guillemets ajoutés
    if POSTPROCESS_BACKEND == "openai":
        prompt = (
            f"Traduis en {target_lang} le texte suivant, SANS ajouter ni retirer de contenu, "
            "et SANS entourer de guillemets. Garde la ponctuation naturelle et les retours à la ligne si présents.\n"
            f"---\n{text}\n---"
        )
        # certains modèles n'acceptent pas temperature!=1
        temp = 1 if (model and any(x in model for x in ["gpt-4o-mini","gpt-5-mini","gpt-5-nano"])) else 0.2
        return _openai_complete(prompt, model=model, temperature=temp, api_key=api_key)
    elif POSTPROCESS_BACKEND == "marian":
        return _marian_translate(text, target_lang)
    else:
        return text

def translate_subs_to_srt(input_srt: Path, target_lang: str, model: Optional[str], api_key: Optional[str]) -> Path:
    subs = read_srt(input_srt)
    out_subs = []
    for i, s in enumerate(subs, 1):
        src = (s.content or "").strip()
        if not src:
            out_subs.append(srt.Subtitle(index=i, start=s.start, end=s.end, content=""))
            continue
        trans = _translate_line(src, target_lang, model, api_key)
        out_subs.append(srt.Subtitle(index=i, start=s.start, end=s.end, content=trans))

    # nom : <base>[.<src>].<to>.srt  (si <src> présent à la fin du stem, on le remplace)
    stem = input_srt.stem
    m = re.search(r"\.([a-z]{2,3})$", stem, flags=re.I)
    if m:
        base_no_lang = stem[: -len(m.group(0))]
    else:
        base_no_lang = stem
    out_path = input_srt.with_name(f"{base_no_lang}.{target_lang}.srt")

    out_path.write_text(srt.compose(out_subs), encoding="utf-8")
    return out_path

# ---------- Chunking amélioré ----------

def smart_chunks(text: str, max_chars: int = 2800, overlap: int = 200) -> Iterable[str]:
    """
    Découpe par phrases approximatives (sur '. ' et saillants), avec chevauchement pour garder du contexte.
    Fallback à chunks_of_text si texte très court.
    """
    t = (text or "").strip()
    if _safe_len(t) <= max_chars:
        yield t
        return

    # split naïf sur fins de phrases
    import re
    sentences = re.split(r"(?<=[.!?])\s+", t)
    buf: List[str] = []
    size = 0
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        if size + len(s) + 1 > max_chars and buf:
            chunk = " ".join(buf).strip()
            yield chunk
            # chevauchement
            if overlap > 0:
                keep = chunk[-overlap:]
                buf = [keep, s]
                size = len(keep) + len(s) + 1
            else:
                buf = [s]
                size = len(s) + 1
        else:
            buf.append(s)
            size += len(s) + 1
    if buf:
        yield " ".join(buf).strip()


# ---------- Cœurs de tâches ----------

def _naturalize(text: str, glossary: Optional[Dict[str, str]] = None, model: Optional[str] = None, api_key: Optional[str] = None) -> str:
    parts: List[str] = []
    for ch in smart_chunks(text, max_chars=2800, overlap=200):
        g_section = ""
        if glossary:
            pairs = "\n".join([f"- {k} → {v}" for k, v in glossary.items()])
            g_section = f"\n\nContraintes terminologiques (à respecter strictement):\n{pairs}\n"

        prompt = NATURALIZE_PROMPT.format(chunk=ch) + g_section
        if POSTPROCESS_BACKEND == "openai":
            parts.append(_openai_complete(prompt, model=model, temperature=0.2, api_key=api_key))
        elif POSTPROCESS_BACKEND == "none":
            parts.append(ch)
        else:
            parts.append(ch)

    merged = "\n\n".join(parts)

    if POSTPROCESS_BACKEND == "openai" and len(parts) > 1:
        final_prompt = (
            "Tu reçois un texte composé de plusieurs morceaux déjà corrigés. "
            "Unifie le style, supprime les répétitions résiduelles, assure la cohérence des références et pronoms, "
            "et rends l'ensemble parfaitement fluide en français. Ne change pas le sens.\n\n"
            f"Texte:\n---\n{merged}\n---"
        )
        return _openai_complete(final_prompt, model=model, temperature=0.2, api_key=api_key)

    return merged


def _translate(text: str, target_lang: str, glossary: Optional[Dict[str, str]] = None, model: Optional[str] = None, api_key: Optional[str] = None) -> str:
    if POSTPROCESS_BACKEND == "openai":
        parts: List[str] = []
        for ch in smart_chunks(text, max_chars=2800, overlap=200):
            g_section = ""
            if glossary:
                pairs = "\n".join([f"- {k} → {v}" for k, v in glossary.items()])
                g_section = f"\n\nContraintes terminologiques (à respecter strictement):\n{pairs}\n"
            prompt = TRANSLATE_PROMPT.format(target_lang=target_lang, chunk=ch) + g_section
            parts.append(_openai_complete(prompt, model=model, temperature=0.2, api_key=api_key))
        merged = "\n\n".join(parts)

        if len(parts) > 1:
            final_prompt = (
                f"Relis et harmonise cette traduction en {target_lang}. "
                "Uniformise le style, corrige les micro-incohérences de contexte, garde le sens exact.\n\n"
                f"Traduction:\n---\n{merged}\n---"
            )
            return _openai_complete(final_prompt, model=model, temperature=0.2, api_key=api_key)
        return merged

    elif POSTPROCESS_BACKEND == "marian":
        return _marian_translate(text, target_lang)
    else:
        return text


# ---------- API principale ----------

def naturalize_and_translate(
    input_srt: Path,
    to_lang: Optional[str] = None,
    mode: str = "naturalize",
    backend_override: Optional[str] = None,
    glossary: Optional[Dict[str, str]] = None,
    openai_model: Optional[str] = None,
    api_key: str = None, # type: ignore
) -> Dict[str, str]:
    global POSTPROCESS_BACKEND  # doit être tout en haut
    backend = (backend_override or POSTPROCESS_BACKEND).lower()

    # Fallback .env si pas de overrides GUI
    use_model = openai_model or os.getenv("POSTPROCESS_OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    use_key = api_key or os.getenv("OPENAI_API_KEY")

    if backend == "openai" and not use_key:
        raise RuntimeError("Aucune clé OpenAI fournie. Donne une clé dans la GUI ou mets OPENAI_API_KEY dans l'env.")

    subs = read_srt(input_srt)
    raw_text = " ".join(s.content.strip() for s in subs)
    base = input_srt.parent
    outputs: Dict[str, str] = {}

    # Naturalisation
    if mode in ("naturalize", "naturalize+translate"):
        naturalized = _naturalize(raw_text, glossary=glossary, model=use_model, api_key=use_key)
        nat_path = base / f"{input_srt.stem}.naturalized.txt"
        nat_path.write_text(naturalized, encoding="utf-8")
        outputs["naturalized_txt"] = str(nat_path)
    else:
        naturalized = raw_text

    # Traduction
    if to_lang:
        translated = _translate(naturalized, to_lang, glossary=glossary, model=use_model, api_key=use_key)
        out_tr   = base / f"{input_srt.stem}.naturalized.{to_lang}.txt"
        out_tr.write_text(translated, encoding="utf-8")
        outputs["translated_txt"] = str(out_tr)
        
        # Traduction
    if to_lang:
        translated = _translate(naturalized, to_lang, glossary=glossary, model=use_model, api_key=use_key)
        out_tr = base / f"{input_srt.stem}.naturalized.{to_lang}.txt"
        out_tr.write_text(translated, encoding="utf-8")
        outputs["translated_txt"] = str(out_tr)

        # NEW: SRT traduit (aligné sur les timecodes d'origine)
        translated_srt_path = translate_subs_to_srt(input_srt, to_lang, model=use_model, api_key=use_key)
        outputs["translated_srt"] = str(translated_srt_path)

    return outputs

