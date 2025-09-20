import typer
from pathlib import Path
from dotenv import load_dotenv
from transcribe import transcribe_video
from postprocess import naturalize_and_translate
from remux import remux_subs

app = typer.Typer(no_args_is_help=True)

@app.command("transcribe")
def transcribe_cmd(
    input_source: str = typer.Argument(...),
    out: Path = typer.Option(Path("out"), "--out", "-o"),
    whisper_model: str = typer.Option("medium"),
    lang: str = typer.Option(None),
    audio_only: bool = typer.Option(True, help="Télécharger uniquement l'audio (plus rapide)."),
    device: str = typer.Option("cpu"),
    save_media: bool = typer.Option(False, help="Conserver le média téléchargé dans --out"),
    download_video: bool = typer.Option(False, help="Forcer le téléchargement de la vidéo complète (ignore audio_only)"),
):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    srt_path = (out / "subtitles.srt").resolve()
    txt_path = (out / "transcript.raw.txt").resolve()

    prefer_audio_only = audio_only and (not download_video)

    info = transcribe_video(
        input_source, srt_path, txt_path,
        whisper_model=whisper_model,
        language=lang,
        prefer_audio_only=prefer_audio_only,
        device=device,
        save_media=save_media
    )

    exists_srt = srt_path.exists()
    exists_txt = txt_path.exists()
    size_srt = srt_path.stat().st_size if exists_srt else 0
    size_txt = txt_path.stat().st_size if exists_txt else 0

    typer.echo(f"SRT -> exists={exists_srt} size={size_srt} path={srt_path}")
    typer.echo(f"RAW -> exists={exists_txt} size={size_txt} path={txt_path}")
    typer.echo(f"Segments={info['segments']} chars={info['chars']} lang_detected={info.get('language')}")

@app.command("postprocess")
def postprocess_cmd(
    input_srt: Path = typer.Argument(..., exists=True, readable=True),
    to_lang: str = typer.Option(None),
    mode: str = typer.Option("naturalize"),
    backend: str = typer.Option(None),
):
    load_dotenv()
    out = naturalize_and_translate(input_srt, to_lang=to_lang, mode=mode, backend_override=backend)
    for k, v in out.items():
        typer.echo(f"{k}: {v}")

@app.command("remux")
def remux_cmd(
    input_video: Path = typer.Argument(..., exists=True, readable=True),
    input_srt: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(Path("output_with_subs.mp4"), "--out", "-o"),
    title: str = typer.Option("Subtitles"),
    language: str = typer.Option(None),
):
    remux_subs(input_video, input_srt, out, title=title, language=language)
    typer.echo(f"OK: {out}")

if __name__ == "__main__":
    app()
