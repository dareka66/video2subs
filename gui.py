# gui.py
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# ---- Qt / GUI ----
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QCheckBox, QComboBox,
    QTextEdit, QMessageBox, QGroupBox, QFormLayout, QProgressBar
)

# ---- Réseau pour miniature ----
import requests

# ---- Import du pipeline (supporte deux layouts: package ou fichiers à la racine)

from transcribe import transcribe_video              # si tout est à la racine
from postprocess import naturalize_and_translate  # ajoute ceci

def _stage_progress(stage: str, percent: float) -> int:
    # Répartition: probe 0–5, download 5–40, transcribe 40–90, postprocess 90–100
    bands = {"probe": (0, 5), "download": (5, 40), "transcribe": (40, 90), "postprocess": (90, 100)}
    start, end = bands.get(stage, (0, 100))
    p = max(0.0, min(100.0, float(percent or 0.0)))
    return int(round(start + (end - start) * (p / 100.0)))

# --------------------------- Modèle de données ---------------------------

@dataclass
class TranscribeParams:
    input_source: str
    out_dir: Path
    whisper_model: str
    language: Optional[str]
    device: str
    prefer_audio_only: bool
    save_media: bool
    do_naturalize: bool
    do_translate: bool
    target_lang: Optional[str]
    backend: str
    api_key: Optional[str]
    openai_model: Optional[str]


# --------------------------- Worker en thread ---------------------------

class Worker(QObject):
    started = Signal()
    log = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)
    progress = Signal(str, float, dict)   # stage, percent, meta
    thumbnail = Signal(bytes)             # image bytes

    def __init__(self, params: TranscribeParams):
        super().__init__()
        self.params = params

    # Callback passé au pipeline (download & transcription)
    def _cb(self, stage: str, percent: float, meta: Dict[str, Any]):
        # Récupérer la miniature une seule fois au stage "probe"
        if stage == "probe" and meta.get("thumbnail"):
            try:
                resp = requests.get(meta["thumbnail"], timeout=20)
                if resp.ok and resp.content:
                    self.thumbnail.emit(resp.content)
            except Exception:
                pass
        self.progress.emit(stage, percent or 0.0, meta or {})

    def run(self):
        try:
            self.started.emit()
            p = self.params

            self.log.emit("Création du dossier de sortie…")
            p.out_dir.mkdir(parents=True, exist_ok=True)

            srt_path = (p.out_dir / "subtitles.srt").resolve()
            txt_path = (p.out_dir / "transcript.raw.txt").resolve()

            self.log.emit(f"Source : {p.input_source}")
            self.log.emit(f"Sortie : {p.out_dir}")
            self.log.emit(f"Whisper : {p.whisper_model} | Langue : {p.language or 'auto'} | Device : {p.device}")
            self.log.emit(f"Téléchargement : {'VIDÉO complète' if not p.prefer_audio_only else 'audio-only'} | Conserver média : {'oui' if p.save_media else 'non'}")

            info = transcribe_video(
                input_source=p.input_source,
                srt_out=(p.out_dir / "DUMMY.srt"),        # sera ignoré/écrasé
                raw_txt_out=(p.out_dir / "DUMMY.raw.txt"),# sera ignoré/écrasé
                whisper_model=p.whisper_model,
                language=p.language,
                device=p.device,
                prefer_audio_only=p.prefer_audio_only,
                save_media=p.save_media,
                cb=self._cb,
            )

            # 🔁 NOUVEAU : récupérer les VRAIS chemins renvoyés par transcribe_video
            srt_path = Path(info.get("srt_path", p.out_dir / "subtitles.srt"))
            txt_path = Path(info.get("raw_txt_path", p.out_dir / "transcript.raw.txt"))

            exists_srt = srt_path.exists()
            exists_txt = txt_path.exists()
            size_srt = srt_path.stat().st_size if exists_srt else 0
            size_txt = txt_path.stat().st_size if exists_txt else 0

            self.log.emit(f"SRT -> exists={exists_srt} size={size_srt} path={srt_path}")
            self.log.emit(f"RAW -> exists={exists_txt} size={size_txt} path={txt_path}")
            self.log.emit(f"Segments={info.get('segments')} | chars={info.get('chars')} | lang_detected={info.get('language')}")
            
            # Post-traitement (naturaliser/traduire)
            if self.params.do_naturalize or self.params.do_translate:
                self.log.emit("Post-traitement en cours…")
                # petite animation de progression indéterminée
                self.progress.emit("postprocess", -1, {})

                mode = "naturalize"
                if self.params.do_naturalize and self.params.do_translate:
                    mode = "naturalize+translate"
                elif not self.params.do_naturalize and self.params.do_translate:
                    mode = "translate"

                self.progress.emit("postprocess", 0.0, {})
                outputs = naturalize_and_translate(
                    srt_path,  # 🔁 on passe le vrai chemin
                    to_lang=self.params.target_lang,
                    mode=mode,
                    backend_override=self.params.backend,
                    api_key=self.params.api_key,            # 🔁 clé GUI # type: ignore
                    openai_model=self.params.openai_model,  # 🔁 modèle GUI (ou None => .env)
                )
                self.progress.emit("postprocess", 100.0, {})
                tr_srt = outputs.get("translated_srt")
                if tr_srt:
                    self.log.emit(f"Traduit (srt) : {tr_srt}")


                nat_path = outputs.get("naturalized_txt")
                tr_path = outputs.get("translated_txt")
                if nat_path: self.log.emit(f"Naturalisé : {nat_path}")
                if tr_path: self.log.emit(f"Traduit   : {tr_path}")

                # fin postprocess : barre à 100%
                self.progress.emit("postprocess", 100.0, {})

            if not exists_srt or size_srt == 0:
                self.failed.emit("Sous-titres vides ou absents. Vérifie la source/langue/modèle.")
                return

            out = {
                "srt": str(srt_path),
                "raw": str(txt_path),
            }
            meta = info.get("meta") or {}
            out.update(meta)
            self.finished.emit(out)

        except Exception as e:
            tb = traceback.format_exc()
            self.log.emit(tb)
            self.failed.emit(str(e))


# --------------------------- Fenêtre principale ---------------------------

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("video2subs — GUI")
        self.setMinimumWidth(760)

        # ===================== Widgets d'entrée =====================
        # Source
        self.input_edit = QLineEdit()
        self.btn_browse_file = QPushButton("Parcourir…")
        self.btn_browse_file.clicked.connect(self.pick_file)

        # Dossier de sortie
        self.out_edit = QLineEdit()
        self.btn_browse_out = QPushButton("Choisir…")
        self.btn_browse_out.clicked.connect(self.pick_out_dir)

        # Options de téléchargement
        self.cb_download_video = QCheckBox("Télécharger la vidéo entière (sinon audio-only)")
        self.cb_save_media = QCheckBox("Conserver le média téléchargé dans le dossier de sortie")

        # Whisper
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v2"])
        self.model_combo.setCurrentText("medium")

        # Langue ASR
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("Auto", userData=None)
        for code in ["fr", "en", "es", "de", "it", "pt", "ja", "ko", "zh"]:
            self.lang_combo.addItem(code, userData=code)

        # Device
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "auto", "cuda"])
        self.device_combo.setCurrentText("cpu")

        # ===================== Post-traitement =====================
        self.cb_naturalize = QCheckBox("Naturaliser le texte (ponctuation, nettoyage)")
        self.cb_translate = QCheckBox("Traduire")

        # Backend + (API key & modèle OpenAI si backend=openai)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["openai", "marian", "none"])
        self.backend_combo.setCurrentText("openai")

        # Champ API key (doit exister AVANT le toggle)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password) # type: ignore
        self.api_key_edit.setPlaceholderText("sk-...")

        # Modèle OpenAI (postprocess)
        self.model_post_combo = QComboBox()
        self.model_post_combo.addItem("Auto (.env)", userData=None)
        for m in ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o"]:
            self.model_post_combo.addItem(m, userData=m)
        self.model_post_combo.setCurrentIndex(0)

        # Prefill depuis .env (si présent)
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            self.api_key_edit.setText(env_key)  # restera masquée
        env_model = os.getenv("POSTPROCESS_OPENAI_MODEL")
        if env_model:
            idx = self.model_post_combo.findData(env_model)
            if idx != -1:
                self.model_post_combo.setCurrentIndex(idx)

        # Toggle unique selon backend
        def toggle_backend(_=None):
            is_oa = (self.backend_combo.currentText() == "openai")
            self.api_key_edit.setEnabled(is_oa)
            self.model_post_combo.setEnabled(is_oa)
        self.backend_combo.currentIndexChanged.connect(toggle_backend)
        toggle_backend()  # init OK: widgets déjà créés

        # Langue cible (traduction)
        self.target_combo = QComboBox()
        for code in ["en", "fr", "es", "de", "it", "pt", "ja", "ko", "zh"]:
            self.target_combo.addItem(code)
        self.target_combo.setCurrentText("en")
        self.target_combo.setEnabled(False)
        self.cb_translate.toggled.connect(self.target_combo.setEnabled)

        # Action
        self.btn_start = QPushButton("Transcrire")
        self.btn_start.clicked.connect(self.on_start)

        # ===================== UI feedback =====================
        # Miniature
        self.thumb_label = QLabel("Miniature")
        self.thumb_label.setAlignment(Qt.AlignCenter) # type: ignore
        self.thumb_label.setFixedHeight(180)
        self.thumb_label.setStyleSheet("border:1px solid #444; background:#111; color:#aaa;")

        # Progression
        self.stage_label = QLabel("Prêt.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        # Logs
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        # ===================== Layout =====================
        form_box = QGroupBox("Paramètres")
        form = QFormLayout(form_box)

        # API Key (toujours visible, mais activée seulement si backend=openai)
        form.addRow(QLabel("API Key (OpenAI) :"), self.api_key_edit)

        # Source
        src_row = QHBoxLayout()
        src_row.addWidget(self.input_edit, stretch=1)
        src_row.addWidget(self.btn_browse_file)
        form.addRow(QLabel("Source (URL ou fichier) :"), self._wrap(src_row))

        # Dossier sortie
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_edit, stretch=1)
        out_row.addWidget(self.btn_browse_out)
        form.addRow(QLabel("Dossier de sortie :"), self._wrap(out_row))

        # Options download
        form.addRow(self.cb_download_video)
        form.addRow(self.cb_save_media)

        # Whisper / OpenAI / Langue / Device
        form.addRow(QLabel("Modèle Whisper :"), self.model_combo)
        form.addRow(QLabel("Modèle OpenAI :"), self.model_post_combo)
        form.addRow(QLabel("Langue ASR :"), self.lang_combo)
        form.addRow(QLabel("Device :"), self.device_combo)

        # Post-process
        form.addRow(self.cb_naturalize)
        form.addRow(self.cb_translate)
        form.addRow(QLabel("Backend post-traitement :"), self.backend_combo)
        form.addRow(QLabel("Langue cible :"), self.target_combo)

        # Main layout
        main = QVBoxLayout(self)
        main.addWidget(form_box)
        main.addWidget(self.btn_start, alignment=Qt.AlignLeft) # type: ignore
        main.addWidget(self.thumb_label)

        prog_row = QHBoxLayout()
        prog_row.addWidget(self.stage_label)
        prog_row.addWidget(self.progress)
        main.addLayout(prog_row)

        main.addWidget(QLabel("Logs :"))
        main.addWidget(self.log_view, stretch=1)

    # Helpers UI
    def _wrap(self, layout):
        w = QWidget()
        w.setLayout(layout)
        return w

    def append_log(self, text: str):
        self.log_view.append(text)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    # Actions
    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir un média local", "",
            "Médias (*.mp4 *.mkv *.mov *.mp3 *.m4a *.wav *.webm);;Tous les fichiers (*.*)"
        )
        if path:
            self.input_edit.setText(path)

    def pick_out_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie")
        if path:
            self.out_edit.setText(path)

    def on_start(self):
        source = self.input_edit.text().strip()
        out = self.out_edit.text().strip()

        if not source:
            QMessageBox.warning(self, "Manque la source", "Indique une URL ou choisis un fichier.")
            return
        if not out:
            QMessageBox.warning(self, "Manque le dossier de sortie", "Choisis un dossier de sortie.")
            return

        params = TranscribeParams(
            input_source=source,
            out_dir=Path(out),
            whisper_model=self.model_combo.currentText(),
            language=self.lang_combo.currentData(),
            device=self.device_combo.currentText(),
            prefer_audio_only=not self.cb_download_video.isChecked(),
            save_media=self.cb_save_media.isChecked(),
            do_naturalize=self.cb_naturalize.isChecked(),
            do_translate=self.cb_translate.isChecked(),
            target_lang=self.target_combo.currentText() if self.cb_translate.isChecked() else None,
            backend=self.backend_combo.currentText(),
            api_key=self.api_key_edit.text().strip() or None,
            openai_model=self.model_post_combo.currentData(),
        )


        # Reset UI
        self.btn_start.setEnabled(False)
        self.thumb_label.setPixmap(QPixmap())
        self.progress.setValue(0)
        self.stage_label.setText("Démarrage…")
        self.log_view.clear()
        self.append_log("Démarrage du traitement…")

        # Thread de travail
        self.thread = QThread(self) # type: ignore
        self.worker = Worker(params)
        self.worker.moveToThread(self.thread) # type: ignore

        # Connexions signaux
        self.thread.started.connect(self.worker.run) # type: ignore
        self.worker.started.connect(lambda: self.append_log("⏳ Travail en cours…"))
        self.worker.log.connect(self.append_log)
        self.worker.thumbnail.connect(self.on_thumbnail)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)

        # Fin de thread
        self.worker.finished.connect(self.thread.quit) # type: ignore
        self.worker.failed.connect(self.thread.quit) # type: ignore
        self.thread.finished.connect(self.thread.deleteLater) # type: ignore

        # Go!
        self.thread.start() # type: ignore

    # Slots
    def on_thumbnail(self, img_bytes: bytes):
        img = QImage.fromData(img_bytes)
        if not img.isNull():
            pix = QPixmap.fromImage(img).scaledToHeight(180, Qt.SmoothTransformation) # type: ignore
            self.thumb_label.setPixmap(pix)

    def on_progress(self, stage: str, percent: float, meta: dict):
        names = {"probe": "Analyse", "download": "Téléchargement", "transcribe": "Transcription", "postprocess": "Post-traitement"}
        self.stage_label.setText(names.get(stage, stage).capitalize())

        if stage == "postprocess" and (percent is None or percent < 0):
            self.progress.setRange(0, 0)  # indéterminé
            return

        if self.progress.minimum() == 0 and self.progress.maximum() == 0:
            self.progress.setRange(0, 100)

        try:
            overall = _stage_progress(stage, percent)
            self.progress.setValue(overall)
        except Exception:
            pass


    def on_finished(self, info: Dict[str, Any]):
        self.append_log("✅ Terminé.")
        if info.get("srt"):
            self.append_log(f"Sous-titres : {info['srt']}")
        if info.get("raw"):
            self.append_log(f"Texte brut   : {info['raw']}")
        if info.get("title"):
            self.append_log(f"Titre : {info['title']}")
        if info.get("duration") is not None:
            self.append_log(f"Durée (s) : {info['duration']}")
        self.btn_start.setEnabled(True)
        self.stage_label.setText("Terminé")
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if info else 0)
        QMessageBox.information(self, "Terminé", "Transcription terminée ✔")

    def on_failed(self, error: str):
        self.append_log("❌ Échec.")
        self.append_log(error)
        self.btn_start.setEnabled(True)
        self.stage_label.setText("Erreur")
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if error else 0)
        QMessageBox.critical(self, "Erreur", f"Une erreur est survenue :\n{error}")


# --------------------------- Entrée du programme ---------------------------

def main():
    load_dotenv()
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
