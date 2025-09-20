# video2subs — URL/Fichier → Sous-titres + Texte → Naturalisation + Traduction

**video2subs** télécharge une vidéo (YouTube, etc.) ou lit un fichier local, en extrait l’audio, transcrit avec [faster-whisper], et peut naturaliser + traduire le texte (OpenAI ou Marian offline).  
Il fournit des sorties cohérentes, nommées d’après le média source :
```
<Nom>.{lang}.srt
<Nom>.{lang}.raw.txt
<Nom>.{lang}.naturalized.txt
<Nom>.{lang}.naturalized.<to>.txt
<Nom>.<to>.srt            # SRT traduit aligné (option)
```

## Fonctionnalités
- URL (YouTube via `yt-dlp`) ou fichier local
- Option vidéo complète ou audio-only
- ASR auto (détection langue) ou langue forcée
- Naturalisation du texte
- Traduction (OpenAI, ou Marian offline FR↔EN par défaut)
- SRT traduit (aligné sur timecodes)
- GUI cross-platform (PySide6)
- CLI ergonomique (Typer)

## Prérequis
- Python 3.10+ (testé avec 3.12)
- `ffmpeg` installé dans le PATH
- (Optionnel) Clé OpenAI si backend `openai`

Linux (Debian/Ubuntu) :
```bash
sudo apt update
sudo apt install -y ffmpeg
```

Windows :
- Installe ffmpeg (ex. via winget/choco) et ajoute-le au PATH.

## Installation rapide
```bash
git clone https://github.com/dareka66/video2subs.git
cd video2subs
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration (.env)
Crée un fichier `.env` à la racine :
```env
OPENAI_API_KEY=sk-xxxxx
POSTPROCESS_OPENAI_MODEL=gpt-4o
POSTPROCESS_BACKEND=openai
```

## Interface graphique (GUI)
```bash
python gui.py
```

## CLI
### Transcrire (ASR auto, audio-only)
```bash
python cli.py transcribe "https://www.youtube.com/watch?v=XXXX" --out out
```

### Transcrire vidéo complète et la garder
```bash
python cli.py transcribe "https://www.youtube.com/watch?v=XXXX" --out out --download-video --save-media
```

### Forcer la langue ASR
```bash
python cli.py transcribe input.mp4 --out out --lang en
```

### Post-process (naturaliser + traduire en FR)
```bash
python cli.py postprocess out/<Nom>.<src>.srt --backend openai --to fr
```

## Nommage des fichiers
- Base = nom du média (`<Nom>`).
- Si ASR auto, la langue détectée est ajoutée :
  - `out/<Nom>.en.srt`
  - `out/<Nom>.en.raw.txt`
- Post-process :
  - `out/<Nom>.en.naturalized.txt`
  - `out/<Nom>.fr.txt`
  - `out/<Nom>.fr.srt`

## Dépendances principales
- faster-whisper
- yt-dlp
- ffmpeg-python
- srt
- PySide6
- requests
- python-dotenv
- openai
- transformers + sentencepiece + torch

## Build exécutable
Linux :
```bash
pip install pyinstaller
pyinstaller --onefile --windowed gui.py
```

Windows :
```bash
pyinstaller --onefile --windowed gui.py
```

## Dépannage
- 403 GitHub : mauvais compte/token → reconfigurer `git remote`
- OpenAI BadRequest (temperature) : certains modèles n’acceptent que `temperature=1`
- Fichiers vides : vérifier ffmpeg
- CUDA : fallback CPU si pas dispo

## Licence
MIT
