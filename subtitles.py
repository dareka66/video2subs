import srt
from pathlib import Path

def read_srt(path: Path):
    return list(srt.parse(path.read_text(encoding="utf-8")))
