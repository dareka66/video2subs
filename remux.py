from pathlib import Path
import subprocess
import shlex

def remux_subs(input_video: Path, input_srt: Path, output_video: Path, title="Subtitles", language=None):
    # MP4 + mov_text : soft-sub; pour MKV, pas besoin de mov_text
    # On fait simple: pour MP4
    lang_flag = f"-metadata:s:s:0 language={language}" if language else ""
    cmd = f'ffmpeg -y -i {shlex.quote(str(input_video))} -i {shlex.quote(str(input_srt))} -c copy -c:s mov_text -metadata:s:s:0 title={shlex.quote(title)} {lang_flag} {shlex.quote(str(output_video))}'
    subprocess.run(cmd, shell=True, check=True)
