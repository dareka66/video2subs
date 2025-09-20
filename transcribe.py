# transcribe.py
from typing import Union, Optional, Callable, Dict, Any
from pathlib import Path
from datetime import timedelta
import tempfile, os, ffmpeg, srt
from faster_whisper import WhisperModel
from downloader import download_media, ProgressCb

TranscribeCb = Optional[Callable[[str, float, Dict[str, Any]], None]]

def _pick_compute_type(device: str): return "float16" if device in ("cuda","gpu") else "int8"

def transcribe_video(
    input_source: Union[Path, str],
    srt_out: Path,
    raw_txt_out: Path,
    whisper_model="medium",
    language=None,
    device="cpu",
    prefer_audio_only=True,
    save_media=False,
    cb: TranscribeCb = None,
):
    cleanup = []
    meta: Dict[str, Any] = {"title": None, "thumbnail": None, "duration": None}
    downloaded = False  # 🧩 pour savoir si on peut renommer le média gardé

    # ---------- Résolution source ----------
    if isinstance(input_source, str) and input_source.startswith(("http://","https://")):
        media_path, m = download_media(
            input_source,
            audio_only=prefer_audio_only,
            dest_dir=srt_out.parent,
            keep=save_media,
            cb=cb
        )
        meta.update(m or {})
        downloaded = True
    else:
        media_path = Path(input_source)

    base_name = media_path.stem
    lang_suffix = f".{language}" if language else ""

    # 🧩 UNIFIER les sorties sur le même nom racine + langue
    #    On ignore le nom fichier passé et on garde le dossier.
    out_dir = srt_out.parent
    srt_out = out_dir / f"{base_name}{lang_suffix}.srt"
    raw_txt_out = out_dir / f"{base_name}{lang_suffix}.raw.txt"

    # ---------- Renommer le média gardé (optionnel) ----------
    # Si on a téléchargé dans out/ et qu'on veut conserver, on renomme en <basename>.<lang>.<ext>
    if save_media and downloaded and media_path.parent == out_dir:
        target_media = media_path.with_name(f"{base_name}{lang_suffix}{media_path.suffix}")
        if target_media != media_path:
            try:
                media_path.rename(target_media)
                media_path = target_media
                if cb:
                    cb("rename", 100.0, {"media": str(media_path)})
            except Exception:
                # silencieux: si le rename échoue (verrou, droits), on garde le nom d’origine
                pass
    else:
        # Si on n'a pas gardé le média, on prévois son cleanup plus tard
        if isinstance(input_source, str) and input_source.startswith(("http://","https://")) and not save_media:
            cleanup.append(media_path)

    # ---------- Extraction WAV ----------
    tmp_wav = Path(tempfile.mkstemp(prefix="v2s_", suffix=".wav")[1])
    (
        ffmpeg.input(str(media_path))
        .output(str(tmp_wav), acodec="pcm_s16le", ac=1, ar=16000, vn=None, loglevel="error")
        .overwrite_output().run()
    )

    # ---------- Whisper ----------
    compute_type = _pick_compute_type(device if device!="auto" else "cpu")
    try:
        model = WhisperModel(whisper_model, device=(device if device!="auto" else "cpu"), compute_type=compute_type)
    except ValueError:
        model = WhisperModel(whisper_model, device="cpu", compute_type="int8")

    segments_iter, info = model.transcribe(str(tmp_wav), language=language, vad_filter=True)
    if info and getattr(info, "duration", None):
        meta["duration"] = info.duration  # type: ignore
        
    detected = getattr(info, "language", None)
    if (language is None) and detected:
        out_dir = srt_out.parent
        base_name = media_path.stem
        srt_out = out_dir / f"{base_name}.{detected}.srt"
        raw_txt_out = out_dir / f"{base_name}.{detected}.raw.txt"


    subs, raw_lines, idx = [], [], 1
    last_sec = 0.0
    for seg in segments_iter:
        start = timedelta(seconds=seg.start)
        end = timedelta(seconds=seg.end)
        text = (seg.text or "").strip()
        if text:
            raw_lines.append(text)
            subs.append(srt.Subtitle(index=idx, start=start, end=end, content=text))
            idx += 1
        # progress
        if cb and meta.get("duration"):
            sec = float(seg.end or 0)
            if sec > last_sec:
                last_sec = sec
                cb("transcribe", min(100, 100*sec/float(meta["duration"])), {"lang": getattr(info,"language", None)})  # type: ignore

    # ---------- Écriture ----------
    srt_out.parent.mkdir(parents=True, exist_ok=True)
    raw_txt_out.parent.mkdir(parents=True, exist_ok=True)

    if subs:
        with open(srt_out, "w", encoding="utf-8") as f:
            f.write(srt.compose(subs)); f.flush(); os.fsync(f.fileno())
        with open(raw_txt_out, "w", encoding="utf-8") as f:
            f.write("\n".join(raw_lines)); f.flush(); os.fsync(f.fileno())

    # ---------- Cleanup ----------
    try: tmp_wav.unlink(missing_ok=True)
    except: pass
    for p in cleanup:
        try: p.unlink(missing_ok=True)
        except: pass

    # 🧩 Retourner les chemins réels, pour affichage correct en CLI/GUI
    return {
        "segments": len(subs),
        "chars": sum(len(t) for t in raw_lines),
        "language": getattr(info,"language",None),
        "meta": {**meta, "media_path": str(media_path)},
        "srt_path": str(srt_out),
        "raw_txt_path": str(raw_txt_out),
    }
