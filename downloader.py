# downloader.py
from pathlib import Path
import tempfile, mimetypes, requests
from typing import Callable, Optional, Tuple, Dict, Any
from tqdm import tqdm

ProgressCb = Optional[Callable[[str, float, Dict[str, Any]], None]]
# cb(stage, percent, meta)  stage in {"probe","download"}

def is_probably_direct_media(url: str) -> bool:
    exts = (".mp4", ".mkv", ".mov", ".mp3", ".wav", ".m4a", ".webm", ".ogg")
    return any(url.lower().endswith(ext) for ext in exts)

def _download_direct(url: str, dest_dir: Optional[Path], keep: bool, cb: ProgressCb) -> Tuple[Path, Dict[str, Any]]:
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    ext = mimetypes.guess_extension(ctype.split(";")[0].strip()) or ""
    suffixes = {".mp4",".mkv",".mov",".mp3",".wav",".m4a",".webm",".ogg"}
    suffix = ext if ext in suffixes else ".bin"

    if keep and dest_dir:
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"downloaded{suffix}"
    else:
        fd, tmpname = tempfile.mkstemp(prefix="video2subs_", suffix=suffix)
        path = Path(tmpname)

    total = int(r.headers.get("Content-Length", 0)) or None
    got = 0
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
                got += len(chunk)
                if total and cb:
                    cb("download", min(100, 100*got/total), {"source":"direct"})
    return path, {"title": path.name, "thumbnail": None, "duration": None}

def download_media(
    url: str,
    audio_only: bool = False,
    dest_dir: Optional[Path] = None,
    keep: bool = False,
    cb: ProgressCb = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Télécharge via yt-dlp ou HTTP. Retourne (filepath, info)."""
    try:
        import yt_dlp  # type: ignore
    except Exception:
        yt_dlp = None

    if yt_dlp is None or is_probably_direct_media(url):
        return _download_direct(url, dest_dir, keep, cb)

    base_dir = dest_dir if (keep and dest_dir) else Path(tempfile.mkdtemp(prefix="video2subs_"))
    base_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(base_dir / ("%(title).150s.%(ext)s"))

    def _hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes") or 0
            if total and cb:
                cb("download", min(100, 100*downloaded/total), {"source":"yt-dlp"})
        elif d.get("status") == "finished":
            if cb: cb("download", 100.0, {"source":"yt-dlp"})

    ydl_opts = {
        "outtmpl": outtmpl,
        "quiet": True,
        "noprogress": True,
        "progress_hooks": [_hook],
        "merge_output_format": "mp4",
    }
    if audio_only:
        ydl_opts.update({
            "format": "bestaudio/best",
            "postprocessors": [{"key":"FFmpegExtractAudio","preferredcodec":"m4a","preferredquality":"192"}],
            "merge_output_format": None,
        })
    else:
        ydl_opts["format"] = "bv*+ba/b"

    from yt_dlp import YoutubeDL
    with YoutubeDL(ydl_opts) as ydl: # type: ignore
        # probe metadata upfront
        info = ydl.extract_info(url, download=False)
        if cb:
            cb("probe", 0, {
                "title": info.get("title"),
                "thumbnail": info.get("thumbnail"),
                "duration": info.get("duration")
            })
        info = ydl.extract_info(url, download=True)

        filepaths = []
        req = info.get("requested_downloads") or []
        for it in req:
            fp = it.get("filepath")
            if fp: filepaths.append(Path(fp))

        meta = {
            "title": info.get("title"),
            "thumbnail": info.get("thumbnail"),
            "duration": info.get("duration"),
        }

    # choisir meilleur
    for p in filepaths:
        if p.suffix.lower() in (".mp4",".mkv",".webm",".mov",".m4a",".mp3",".wav",".ogg"):
            return p, meta
    return (filepaths[0], meta) if filepaths else (None, meta) # type: ignore
