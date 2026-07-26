from __future__ import annotations

from dataclasses import asdict
import json
import os
import shutil
import tempfile
import logging
import sys
from pathlib import Path
from threading import Event
from PySide6.QtCore import Qt, QThread, Signal, QStringListModel, QTimer, QUrl, QProcess, Slot, QObject
import time
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QApplication, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QComboBox, QProgressBar, QSplitter, QTableWidget, QTextEdit,
    QTableWidgetItem, QVBoxLayout, QWidget, QCompleter, QSlider, QCheckBox)
from .asr import transcribe, logger as asr_logger
from .audio import render, render_preview, logger as audio_logger
from .matching import find_matches, suggestions
from .models import Flag, Term, Word, load_terms, load_transcript, save_terms
from .subtitles import export_subtitles

def censor_text(text: str) -> str:
    """Replace all vowels (a, e, i, o, u) with '*'."""
    import re
    return re.sub(r'[aeiou]', '*', text, flags=re.IGNORECASE)

class LogEmitter(QObject):
    logged = Signal(str)

class ConsoleHandler(logging.Handler):
    def __init__(self, emitter: LogEmitter):
        super().__init__(); self.emitter = emitter
    def emit(self, record):
        self.emitter.logged.emit(self.format(record))

class TranscriptionWorker(QThread):
    progressed = Signal(int); transcript = Signal(str); stage = Signal(str); finished = Signal(object); failed = Signal(str)
    def __init__(self, source: Path, output: Path, model_size: str, device: str, start_from: float = 0, prompt: str | None = None, existing_words: list[Word] | None = None, existing_rows: list[dict] | None = None):
        super().__init__(); self.source = source; self.output = output; self.model_size = model_size; self.device = device; self.start_from = start_from; self.prompt = prompt
        self.existing_words = existing_words; self.existing_rows = existing_rows; self.paused = Event(); self.cancelled = Event()
    def toggle_pause(self):
        if self.paused.is_set(): self.paused.clear()
        else: self.paused.set()
        return self.paused.is_set()
    def stop(self): self.cancelled.set()
    def run(self):
        try: self.finished.emit(transcribe(self.source, self.output, self.model_size, self.progressed.emit, self.transcript.emit, self.stage.emit, self.paused, self.cancelled, self.device, start_from=self.start_from, prompt=self.prompt, existing_words=self.existing_words, existing_rows=self.existing_rows))
        except Exception as exc: self.failed.emit(str(exc))

class ExportWorker(QThread):
    progressed = Signal(int); finished = Signal(); failed = Signal(str)
    def __init__(self, source: Path, flags: list[Flag], output: Path, segments: list[dict]):
        super().__init__(); self.source = source; self.flags = flags; self.output = output; self.segments = segments
    def run(self):
        try:
            render(self.source, self.flags, self.output, progress_cb=self.progressed.emit)
            export_subtitles(self.segments, self.output.with_suffix(".srt"), self.flags)
            self.finished.emit()
        except Exception as exc: self.failed.emit(str(exc))

class DependencyWorker(QThread):
    progress_msg = Signal(str); finished = Signal(); failed = Signal(str)
    def __init__(self, packages: list[str]):
        super().__init__(); self.packages = packages
    def run(self):
        import subprocess, sys, shutil
        from pathlib import Path

        # 1. Cleanup "ghost" folders (~upb, etc) in site-packages
        self.progress_msg.emit("Cleaning up legacy package folders...")
        try:
            import site
            # site.getsitepackages() may fail in some environments
            paths = []
            try: paths = site.getsitepackages()
            except: pass
            if hasattr(site, 'getusersitepackages'): paths.append(site.getusersitepackages())

            for sp in paths:
                sp_path = Path(sp)
                if not sp_path.exists(): continue
                # Look for folders starting with ~ which pip leaves behind on failed uninstalls
                for ghost in sp_path.glob("**/~*"):
                    try:
                        self.progress_msg.emit(f"  Removing {ghost.name}...")
                        if ghost.is_dir(): shutil.rmtree(ghost)
                        else: ghost.unlink()
                    except: pass
        except Exception as exc:
            self.progress_msg.emit(f"  Cleanup warning: {exc}")

        # 2. Prime the environment with the latest build tools
        # We explicitly force protobuf, ml_dtypes, six, and a compatible numpy to satisfy requirements on Win/3.12+
        priming = [
            ("Upgrading core tools...", [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]),
            ("Fixing compatibility layers...", [sys.executable, "-m", "pip", "install", "--upgrade", "six>=1.17.0", "protobuf>=6.31.1", "ml_dtypes>=0.5.0", "numpy<2.0.0"]),
            ("Installing build helpers...", [sys.executable, "-m", "pip", "install", "cmake", "ninja", "Cython", "packaging", "pybind11", "pyproject-hooks"])
        ]

        for msg, cmd in priming:
            self.progress_msg.emit(msg)
            try:
                # Use a simpler check_call for priming to avoid flooding with too much noise initially
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            except Exception as exc:
                self.failed.emit(f"Priming failed: {exc}"); return

        # 2. Install requested packages with VERBOSE output to see build progress
        for pkg in self.packages:
            self.progress_msg.emit(f"Installing {pkg} (this may take 10-20 minutes if compiling)...")
            try:
                # Use -v (verbose) and --prefer-binary
                # We capture stdout line by line to show compiler progress
                cmd = [sys.executable, "-m", "pip", "install", "-v", "--prefer-binary", pkg]
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    if line.strip():
                        self.progress_msg.emit(f"  {line.strip()}")
                process.wait()
                if process.returncode != 0: raise RuntimeError(f"Pip failed with code {process.returncode}")
            except Exception as exc:
                self.failed.emit(str(exc)); return
        self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("NoNoFilter - Local Audiobook Censor"); self.resize(1180, 800)
        self.source: Path | None = None; self.words: list[Word] = []; self.segments = []; self.flags: list[Flag] = []; self.source_duration = 0
        self.data_dir = Path(os.environ.get("APPDATA", str(Path.home()))) / "NoNoFilter"; self.data_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir = self.data_dir / "projects"; self.projects_dir.mkdir(parents=True, exist_ok=True); self.current_project: Path | None = None
        self.preview_dir = self.data_dir / "previews"; self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.list_path = self.data_dir / "profanity_list.json"; self.terms = load_terms(self.list_path) if self.list_path.exists() else []
        self._build(); self._setup_logging(); self._refresh_terms()
        self._cleanup_orphans()

    def _cleanup_orphans(self):
        """Clean up temporary FFmpeg and project files from previous crashes."""
        try:
            for tmp in Path(tempfile.gettempdir()).glob("tmp*optimized.wav"):
                try: tmp.unlink()
                except: pass
            for tmp in self.projects_dir.glob("*.tmp"):
                try: tmp.unlink()
                except: pass
        except: pass

    def _setup_logging(self):
        self.log_emitter = LogEmitter()
        self.log_emitter.logged.connect(self.append_log)
        handler = ConsoleHandler(self.log_emitter); handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        audio_logger.addHandler(handler); audio_logger.setLevel(logging.DEBUG)
        asr_logger.addHandler(handler); asr_logger.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(handler); logging.getLogger().setLevel(logging.INFO)

    @Slot(str)
    def append_log(self, message: str):
        self.console.append(message)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def _build(self):
        root = QWidget(); self.setCentralWidget(root); layout = QVBoxLayout(root)
        tools = QHBoxLayout()
        self.import_button = QPushButton("Import"); self.import_button.setToolTip("Import a DRM-free audio or video file to begin.")
        self.open_project_button = QPushButton("Open"); self.open_project_button.setToolTip("Open an existing NoNoFilter project file.")
        self.save_project_button = QPushButton("Save"); self.save_project_button.setToolTip("Save the current project, including matches and review status.")
        self.remove_project_button = QPushButton("Remove"); self.remove_project_button.setToolTip("Remove a project from the library (does not delete audio).")
        self.model_box = QComboBox()
        models = [
            ("tiny", "Fastest, lowest accuracy. Good for quick testing."),
            ("base", "Fast, low accuracy. Low memory usage."),
            ("small", "Balanced speed and accuracy."),
            ("medium", "High accuracy, slower. Requires more VRAM/RAM."),
            ("large-v3", "Highest accuracy, very slow."),
            ("turbo", "Optimized Large-v3. Excellent accuracy and speed."),
            ("parakeet-tdt-0.6b-v3", "NVIDIA NeMo Parakeet. Extremely fast and accurate.")
        ]
        for name, tip in models:
            self.model_box.addItem(name)
            self.model_box.setItemData(self.model_box.count() - 1, tip, Qt.ToolTipRole)

        self.model_box.setCurrentText("turbo"); self.model_box.setToolTip("Select the speech-to-text model. Hover over items for details.")
        self.device_box = QComboBox(); self.device_box.addItems(["cpu", "cuda"]); self.device_box.setCurrentText("cpu"); self.device_box.setToolTip("CPU works everywhere. Choose CUDA only if NVIDIA CUDA 12 is installed.")
        self.transcribe_button = QPushButton("Transcribe"); self.transcribe_button.setEnabled(False); self.transcribe_button.setToolTip("Start or resume local speech-to-text analysis.")
        self.load_button = QPushButton("Load"); self.load_button.setEnabled(False); self.load_button.setToolTip("Load an existing .nono.transcript.json file for this audio.")
        self.pause_button = QPushButton("Pause"); self.pause_button.setEnabled(False); self.pause_button.setToolTip("Temporarily pause the transcription process.")
        self.stop_button = QPushButton("Stop"); self.stop_button.setEnabled(False); self.stop_button.setToolTip("Stop transcription and keep segments completed so far.")
        self.match_button = QPushButton("Matches"); self.match_button.setEnabled(False); self.match_button.setToolTip("Find words from your profanity list within the transcript.")
        self.export_button = QPushButton("Export"); self.export_button.setEnabled(False); self.export_button.setToolTip("Render and save the censored media file.")
        tools.addWidget(self.import_button); tools.addWidget(self.open_project_button); tools.addWidget(self.save_project_button); tools.addWidget(self.remove_project_button); tools.addSpacing(10); tools.addWidget(QLabel("Model:")); tools.addWidget(self.model_box); tools.addWidget(QLabel("Device:")); tools.addWidget(self.device_box); tools.addSpacing(10)
        for button in (self.transcribe_button, self.load_button, self.pause_button, self.stop_button, self.match_button, self.export_button): tools.addWidget(button)
        tools.addStretch(); layout.addLayout(tools)

        self.progress = QProgressBar(); self.progress.setVisible(False); layout.addWidget(self.progress)
        self.live_transcript = QTextEdit(); self.live_transcript.setReadOnly(True); self.live_transcript.setPlaceholderText("Active transcription will appear here while processing."); self.live_transcript.setMaximumHeight(80); layout.addWidget(self.live_transcript)

        split = QSplitter(); layout.addWidget(split, 1)
        project = QWidget(); project_layout = QVBoxLayout(project); project_layout.addWidget(QLabel("Active project"))
        search_row = QHBoxLayout(); self.project_search = QLineEdit(); self.project_search.setPlaceholderText("Search transcript..."); search = QPushButton("Search")
        search_row.addWidget(self.project_search); search_row.addWidget(search); project_layout.addLayout(search_row)
        self.chapters = QTableWidget(0, 1); self.chapters.setHorizontalHeaderLabels(["Audio / transcript matches"]); project_layout.addWidget(self.chapters); split.addWidget(project)

        matches_panel = QWidget(); matches_layout = QVBoxLayout(matches_panel)
        bulk_row = QHBoxLayout(); bulk_row.addWidget(QLabel("Matches Review"))
        self.bulk_action_box = QComboBox(); self.bulk_action_box.addItems(["bleep", "mute"]); self.bulk_button = QPushButton("Apply to checked"); self.demo_button = QPushButton("Preview Selected")
        self.censor_ui_toggle = QCheckBox("Censor UI"); self.censor_ui_toggle.setChecked(True)
        bulk_row.addStretch(); bulk_row.addWidget(QLabel("Bulk:")); bulk_row.addWidget(self.bulk_action_box); bulk_row.addWidget(self.bulk_button); bulk_row.addWidget(self.demo_button); bulk_row.addWidget(self.censor_ui_toggle)
        matches_layout.addLayout(bulk_row)
        self.flags_table = QTableWidget(0, 6); self.flags_table.setHorizontalHeaderLabels(["Censor", "Time", "Heard", "Matched term", "Action", "Confidence"]); matches_layout.addWidget(self.flags_table)
        split.addWidget(matches_panel)

        terms = QWidget(); term_layout = QVBoxLayout(terms); term_layout.addWidget(QLabel("Profanity list")); form = QFormLayout()
        self.term_input = QLineEdit(); self.type_box = QComboBox(); self.type_box.addItems(["exact", "fuzzy", "phonetic"]); self.action_box = QComboBox(); self.action_box.addItems(["bleep", "mute"])
        self.term_model = QStringListModel(self); self.completer = QCompleter(self.term_model, self); self.completer.setCaseSensitivity(Qt.CaseInsensitive); self.term_input.setCompleter(self.completer)
        form.addRow("Term", self.term_input); form.addRow("Match", self.type_box); form.addRow("Action", self.action_box); term_layout.addLayout(form)
        add = QPushButton("Add term"); remove = QPushButton("Remove selected"); self.terms_table = QTableWidget(0, 3); self.terms_table.setHorizontalHeaderLabels(["Term", "Match", "Action"])
        term_layout.addWidget(add); term_layout.addWidget(self.terms_table, 1); term_layout.addWidget(remove); split.addWidget(terms); split.setSizes([230, 610, 320])

        player_tools = QHBoxLayout()
        self.play_pause_button = QPushButton("Play"); self.play_pause_button.setFixedWidth(60)
        self.timeline = QSlider(Qt.Horizontal); self.timeline.setRange(0, 1000)
        self.time_label = QLabel("0:00 / 0:00")
        self.filter_toggle = QCheckBox("Filtering On"); self.filter_toggle.setChecked(True)
        self.show_logs_toggle = QCheckBox("Show Logs"); self.show_logs_toggle.setChecked(True)
        self.show_transcription_toggle = QCheckBox("Show Transcription"); self.show_transcription_toggle.setChecked(True)
        player_tools.addWidget(self.play_pause_button); player_tools.addWidget(self.timeline); player_tools.addWidget(self.time_label); player_tools.addWidget(self.filter_toggle); player_tools.addWidget(self.show_logs_toggle); player_tools.addWidget(self.show_transcription_toggle)
        layout.addLayout(player_tools)

        self.console = QTextEdit(); self.console.setReadOnly(True); self.console.setMaximumHeight(120); self.console.setPlaceholderText("Debug console..."); layout.addWidget(self.console)
        self.status = QLabel("Import a DRM-free audio or video file to begin."); layout.addWidget(self.status)

        self.import_button.clicked.connect(self.import_audio); self.open_project_button.clicked.connect(self.open_project); self.save_project_button.clicked.connect(self.save_project); self.remove_project_button.clicked.connect(self.remove_project)
        self.transcribe_button.clicked.connect(self.start_transcription); self.load_button.clicked.connect(self.load_existing_transcript); self.pause_button.clicked.connect(self.toggle_pause); self.stop_button.clicked.connect(self.stop_transcription)
        self.match_button.clicked.connect(self.run_match); self.demo_button.clicked.connect(self.demo_selected); self.bulk_button.clicked.connect(self.apply_bulk_action); self.export_button.clicked.connect(self.export)
        add.clicked.connect(self.add_term); remove.clicked.connect(self.remove_term); self.term_input.textChanged.connect(self.update_suggestions); search.clicked.connect(self.search_project); self.project_search.returnPressed.connect(self.search_project)
        self.flags_table.itemSelectionChanged.connect(self.demo_selected)
        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.timeline.sliderMoved.connect(self.seek_media)
        self.filter_toggle.toggled.connect(self.update_preview_filter)
        self.censor_ui_toggle.toggled.connect(self.refresh_ui)
        self.show_logs_toggle.toggled.connect(self.console.setVisible)
        self.show_transcription_toggle.toggled.connect(self.live_transcript.setVisible)

        self.player = QMediaPlayer(self); self.audio_output = QAudioOutput(self); self.audio_output.setVolume(1.0); self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self.update_timeline); self.player.durationChanged.connect(self.update_duration)
        self.player.mediaStatusChanged.connect(self.start_demo_when_ready); self.player.errorOccurred.connect(self.playback_error)
        self.demo_timer = QTimer(self); self.demo_timer.setSingleShot(True); self.demo_timer.timeout.connect(self.stop_qt_preview); self.pending_demo = None; self.ffplay_process = QProcess(self); self.ffplay_process.finished.connect(self.cleanup_preview)

    def import_audio(self):
        file, _ = QFileDialog.getOpenFileName(self, "Choose media file", "", "Media files (*.mp3 *.m4a *.m4b *.ogg *.opus *.flac *.wav *.mp4 *.mkv *.avi *.mov)")
        if not file: return
        self.source = Path(file); self.current_project = None; self.chapters.setRowCount(1); self.chapters.setItem(0, 0, QTableWidgetItem(self.source.name)); self.transcribe_button.setEnabled(True); self.load_button.setEnabled(self.source.with_suffix(".nono.transcript.json").exists()); self.status.setText(f"Ready: {self.source.name}")
        self.player.setSource(QUrl.fromLocalFile(str(self.source)))

    def _project_data(self):
        flags = self._approved_flags() if self.flags and self.flags_table.rowCount() == len(self.flags) else self.flags
        return {"version": 1, "source_file": str(self.source) if self.source else "", "words": [asdict(word) for word in self.words], "segments": self.segments, "flags": [asdict(flag) for flag in flags]}

    def save_project(self):
        if not self.source:
            QMessageBox.information(self, "Import audio first", "Choose an audio file before saving a project."); return
        if not self.current_project:
            suggested = str(self.projects_dir / f"{self.source.stem}.nono-project.json")
            filename, _ = QFileDialog.getSaveFileName(self, "Save project", suggested, "NoNoFilter project (*.nono-project.json)")
            if not filename: return
            selected = Path(filename)
            self.current_project = selected if selected.name.endswith(".nono-project.json") else selected.with_name(selected.name + ".nono-project.json")
        try:
            self.current_project.parent.mkdir(parents=True, exist_ok=True)
            # Atomic save to prevent corruption
            temp_save = self.current_project.with_suffix(".tmp")
            temp_save.write_text(json.dumps(self._project_data(), indent=2), encoding="utf-8")
            if self.current_project.exists(): self.current_project.unlink()
            temp_save.rename(self.current_project)
            self.status.setText(f"Project saved: {self.current_project.name}")
        except OSError as exc:
            QMessageBox.critical(self, "Could not save project", str(exc))

    def open_project(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open project", str(self.projects_dir), "NoNoFilter project (*.nono-project.json)")
        if not filename: return
        try:
            project = Path(filename); data = json.loads(project.read_text(encoding="utf-8"))
            if not isinstance(data, dict): raise ValueError("Invalid project file format.")
            self.current_project = project
            self.source = Path(data["source_file"]) if data.get("source_file") else None
            self.words = [Word(**w) for w in data.get("words", []) if isinstance(w, dict)]
            self.segments = [s for s in data.get("segments", []) if isinstance(s, dict)]
            self.flags = [Flag(**f) for f in data.get("flags", []) if isinstance(f, dict)]
            self.chapters.setRowCount(0); self.show_transcript(); self._refresh_flags(); self.transcribe_button.setEnabled(bool(self.source)); self.load_button.setEnabled(bool(self.source and self.source.with_suffix(".nono.transcript.json").exists())); self.match_button.setEnabled(bool(self.words)); self.demo_button.setEnabled(bool(self.flags)); self.export_button.setEnabled(bool(self.flags))
            if self.source and self.source.exists(): self.player.setSource(QUrl.fromLocalFile(str(self.source)))
            source_note = "" if self.source and self.source.exists() else " The original audio must be reselected before preview or export."
            self.status.setText(f"Opened project: {project.name}.{source_note}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.critical(self, "Could not open project", f"Error: {exc}")

    def remove_project(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Remove saved project", str(self.projects_dir), "NoNoFilter project (*.nono-project.json)")
        if not filename: return
        project = Path(filename)
        try:
            if project.resolve().parent != self.projects_dir.resolve(): raise ValueError("Only projects in the NoNoFilter project library can be removed here.")
            if QMessageBox.question(self, "Remove project", f"Remove saved project '{project.name}'? This will not delete the audiobook or transcript.") != QMessageBox.StandardButton.Yes: return
            project.unlink()
            if self.current_project and project.resolve() == self.current_project.resolve(): self.current_project = None
            self.status.setText(f"Removed saved project: {project.name}. Audio and transcript were not changed.")
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not remove project", str(exc))

    def load_existing_transcript(self):
        if not self.source: return
        try:
            self.words, self.segments = load_transcript(self.source.with_suffix(".nono.transcript.json")); self.show_transcript(); self.match_button.setEnabled(True)
            self.status.setText(f"Loaded {len(self.words)} words and {len(self.segments)} transcript segments. Add terms, then choose Find matches.")
        except Exception as exc:
            QMessageBox.critical(self, "Could not load transcript", str(exc))

    def start_transcription(self):
        if not self.source: return
        model_size = self.model_box.currentText()

        # Check for NeMo if Parakeet is selected (non-invasive check to avoid UI crash)
        if "parakeet" in model_size:
            import importlib.util
            if importlib.util.find_spec("nemo") is None:
                ans = QMessageBox.question(self, "Install NVIDIA NeMo?",
                    "NVIDIA NeMo is required for the Parakeet model. Would you like to install it now? "
                    "This is a large package and may take several minutes.",
                    QMessageBox.Yes | QMessageBox.No)
                if ans == QMessageBox.Yes:
                    self._install_nemo_and_start()
                    return
                else:
                    return

        self._run_transcription()

    def _install_nemo_and_start(self):
        self.transcribe_button.setEnabled(False)
        self.progress.setVisible(True); self.progress.setRange(0, 0) # Marquee mode
        self.status.setText("Installing NVIDIA NeMo toolkit... Check console for logs.")
        self.console.append("--- Starting NeMo Installation ---")

        # We'll need these for the model to work
        self.dep_worker = DependencyWorker(["nemo_toolkit[asr]", "omegaconf"])
        self.dep_worker.progress_msg.connect(self.console.append)
        self.dep_worker.finished.connect(self._on_nemo_installed)
        self.dep_worker.failed.connect(self._on_nemo_failed)
        self.dep_worker.start()

    def _on_nemo_installed(self):
        self.console.append("--- NeMo Installed Successfully ---")
        self.status.setText("NeMo installed. Starting transcription...")
        self._run_transcription()

    def _on_nemo_failed(self, err):
        self.progress.setVisible(False)
        self.transcribe_button.setEnabled(True)
        self.status.setText("NeMo installation failed.")

        msg = (
            f"An error occurred during installation:\n\n{err}\n\n"
            "On Windows, NeMo often requires the 'C++ Build Tools'.\n"
            "Please install them from: https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
            "Select 'Desktop development with C++' during setup."
        )
        QMessageBox.critical(self, "Installation Failed", msg)

    def _run_transcription(self):
        start_from = 0; prompt = None; existing_words = None; existing_rows = None

        # Auto-detect existing transcript if memory is empty
        transcript_path = self.source.with_suffix(".nono.transcript.json")
        if not self.segments and transcript_path.exists():
            try:
                self.words, self.segments = load_transcript(transcript_path)
                self.show_transcript()
                self.status.setText(f"Detected existing transcript for {self.source.name}.")
            except Exception as exc:
                logging.getLogger("nono_filter").error(f"Could not auto-load existing transcript: {exc}")

        if self.segments:
            msg = QMessageBox(self)
            msg.setWindowTitle("Transcription data exists")
            msg.setText(f"This project already has {self.segments[-1]['end'] / 60:.1f} minutes of transcribed audio. Would you like to resume from where you left off, or RESTART from the beginning?")
            msg.setInformativeText("Warning: Restarting will discard all current transcription data and redo the entire process.")
            resume = msg.addButton("Resume", QMessageBox.AcceptRole)
            restart = msg.addButton("Restart / Redo Entire File", QMessageBox.DestructiveRole)
            cancel = msg.addButton(QMessageBox.Cancel)
            msg.exec()
            if msg.clickedButton() == cancel: return
            if msg.clickedButton() == restart:
                self.words = []; self.segments = []; self.show_transcript()
                start_from = 0; prompt = None; existing_words = None; existing_rows = None
            else: # Resume
                start_from = self.segments[-1]['end']
                existing_words = list(self.words)
                existing_rows = list(self.segments)
                prompt = " ".join([s['text'] for s in self.segments[-5:]]).strip()
                self.live_transcript.append(f"--- Resuming from {start_from / 60:.1f}m ---")

        out = transcript_path
        self.worker = TranscriptionWorker(self.source, out, self.model_box.currentText(), self.device_box.currentText(), start_from=start_from, prompt=prompt, existing_words=existing_words, existing_rows=existing_rows)
        self.worker.start_from_pct = int(start_from / (self.source_duration / 1000) * 100) if self.source_duration > 0 else 0
        self.worker.progressed.connect(self.update_transcription_progress); self.worker.transcript.connect(self.append_transcript); self.worker.stage.connect(self.update_stage); self.worker.finished.connect(self.transcription_done); self.worker.failed.connect(self.failed)
        self.live_transcript.clear() if start_from == 0 else None
        self.progress.setVisible(True); self.progress.setRange(0, 100); self.progress.setValue(self.worker.start_from_pct)
        self.transcribe_button.setEnabled(False); self.model_box.setEnabled(False); self.device_box.setEnabled(False); self.pause_button.setEnabled(True); self.stop_button.setEnabled(True); self.started_at = time.monotonic(); self.last_eta_update = 0; self.last_stage = f"Starting {self.worker.model_size} transcription"
        if hasattr(self, "last_stage_with_eta"): del self.last_stage_with_eta
        self.activity_timer = QTimer(self); self.activity_timer.timeout.connect(self.refresh_activity); self.activity_timer.start(1000); self.status.setText(self.last_stage); self.worker.start()

    def update_transcription_progress(self, val):
        self.progress.setValue(val)
        now = time.monotonic()
        # Ultra-Throttled ETA: Every 10 seconds and only after 2% progress
        if val >= 2 and hasattr(self, "started_at") and (now - self.last_eta_update > 10 or val == 100):
            elapsed = now - self.started_at
            start_val = getattr(self.worker, "start_from_pct", 0)
            progress_done = val - start_val
            if progress_done > 0:
                eta_seconds = int((elapsed / progress_done) * (100 - val))
                eta_str = time.strftime('%M:%S', time.gmtime(eta_seconds))
                self.last_stage_with_eta = f"{self.last_stage} ({val}% - ETA: {eta_str})"
                self.last_eta_update = now
                self.status.setText(self.last_stage_with_eta)
        elif not hasattr(self, "last_stage_with_eta") or val < 2:
            self.status.setText(f"{self.last_stage} ({val}%)")

    def update_stage(self, stage):
        self.last_stage = stage; self.refresh_activity()

    def refresh_activity(self):
        if hasattr(self, "started_at"):
            text = getattr(self, "last_stage_with_eta", self.last_stage)
            self.status.setText(f"{text}  Elapsed: {int(time.monotonic() - self.started_at)}s")

    def finish_activity(self):
        if hasattr(self, "activity_timer"): self.activity_timer.stop()

    def append_transcript(self, text):
        self.live_transcript.append(text)
        self.live_transcript.verticalScrollBar().setValue(self.live_transcript.verticalScrollBar().maximum())

    def toggle_pause(self):
        if hasattr(self, "worker") and self.worker.isRunning():
            paused = self.worker.toggle_pause(); self.pause_button.setText("Resume" if paused else "Pause"); self.status.setText("Transcription paused." if paused else "Transcribing locally...")

    def stop_transcription(self):
        if hasattr(self, "worker") and self.worker.isRunning(): self.worker.stop(); self.pause_button.setEnabled(False); self.stop_button.setEnabled(False); self.status.setText("Stopping after the current transcription segment...")

    def transcription_done(self, data):
        self.finish_activity()
        # Data is already merged in the worker/asr engine now
        self.words = [Word(**x) for x in data["words"]]
        self.segments = data["segments"]

        self.show_transcript(); self.progress.setVisible(False); self.transcribe_button.setEnabled(True); self.model_box.setEnabled(True); self.device_box.setEnabled(True); self.pause_button.setEnabled(False); self.stop_button.setEnabled(False); self.match_button.setEnabled(True)
        next_step = "Add terms, then choose Find matches." if not self.terms else "Choose Find matches to create review candidates."
        self.status.setText(f"Transcript ready: {len(self.words)} words and {len(self.segments)} segments. {next_step}")

    def refresh_ui(self):
        """Redraw all UI components that display censored text."""
        self.show_transcript()
        self._refresh_flags()
        self._refresh_terms()
        if self.project_search.text().strip():
            self.search_project()

    def show_transcript(self):
        """Expose the completed transcript in the active-project pane without requiring a search."""
        self.chapters.setRowCount(len(self.segments))
        flagged_words = {f.word.lower() for f in self.flags}
        do_censor = self.censor_ui_toggle.isChecked()
        for row, segment in enumerate(self.segments):
            text = segment['text'].strip()
            # Censor words that are in our flagged list
            if do_censor:
                text = " ".join([censor_text(w) if w.lower().strip(".,!?\"'") in flagged_words else w for w in text.split()])
            self.chapters.setItem(row, 0, QTableWidgetItem(f"{segment['start']:0.2f}s  {text}"))

    def failed(self, message):
        self.finish_activity(); self.progress.setVisible(False); self.transcribe_button.setEnabled(True); self.model_box.setEnabled(True); self.device_box.setEnabled(True); self.pause_button.setEnabled(False); self.stop_button.setEnabled(False)
        if "stopped by user" in message: self.status.setText("Transcription stopped. The latest completed segments were saved and can be reopened.")
        else: QMessageBox.critical(self, "Transcription failed", message); self.status.setText("Transcription did not finish.")

    def run_match(self):
        if self.flags:
            ans = QMessageBox.question(self, "Redo matching?",
                "You already have flagged words for review. Redoing the matching process will reset all your individual approvals and bleep/mute choices. Are you sure you want to redo it?",
                QMessageBox.Yes | QMessageBox.No)
            if ans == QMessageBox.No:
                return
        self.flags = find_matches(self.words, self.terms); self._refresh_flags(); self.export_button.setEnabled(bool(self.flags)); self.demo_button.setEnabled(bool(self.flags)); self.save_review(); self.status.setText(f"Found {len(self.flags)} review candidates. Uncheck false positives before export.")

    def _refresh_flags(self):
        self.flags_table.setRowCount(len(self.flags))
        do_censor = self.censor_ui_toggle.isChecked()
        for row, flag in enumerate(self.flags):
            keep = QTableWidgetItem(); keep.setCheckState(Qt.Checked if flag.approved else Qt.Unchecked); self.flags_table.setItem(row, 0, keep)
            word = censor_text(flag.word) if do_censor else flag.word
            term = censor_text(flag.term) if do_censor else flag.term
            for col, value in enumerate((f"{flag.start:0.2f}s", word, term), 1): self.flags_table.setItem(row, col, QTableWidgetItem(value))
            action = QComboBox(); action.addItems(["bleep", "mute"]); action.setCurrentText(flag.action); self.flags_table.setCellWidget(row, 4, action); self.flags_table.setItem(row, 5, QTableWidgetItem(f"{flag.confidence:.0%}"))
            action.currentTextChanged.connect(self.update_preview_filter)

    def _approved_flags(self):
        for row, flag in enumerate(self.flags):
            flag.approved = self.flags_table.item(row, 0).checkState() == Qt.Checked; flag.action = self.flags_table.cellWidget(row, 4).currentText()
        return self.flags

    def demo_selected(self):
        row = self.flags_table.currentRow()
        if row < 0 or not self.source: return
        if not self.source.exists(): QMessageBox.warning(self, "Media unavailable", "The file was moved or deleted. Import it again."); return

        flag = self.flags[row]; clip_start = max(0, flag.start - 5); clip_end = flag.end + 5
        # Fully clear the player to release any file locks on Windows
        self.player.stop()
        self.player.setSource(QUrl())
        QApplication.processEvents()

        if not self.filter_toggle.isChecked():
            # Preview original: Seek the global file
            source_url = QUrl.fromLocalFile(str(self.source))
            self.pending_demo = (int(clip_start * 1000), int((clip_end - clip_start) * 1000), flag.word, False, 0)
            if self.player.source() != source_url:
                self.player.setSource(source_url)
            else:
                # If source is same, mediaStatus won't change to trigger start_demo_when_ready
                self.start_demo_when_ready(self.player.mediaStatus())
        else:
            # Preview censored: Use the persistent preview folder
            if self.preview_dir is None:
                self.preview_dir = self.data_dir / "previews"; self.preview_dir.mkdir(parents=True, exist_ok=True)

            self.status.setText("Generating censored preview...")
            # Use a fixed filename to avoid Windows file handle issues with dynamic paths
            preview = self.preview_dir / "active-preview.wav"
            try:
                # We pass the currently visible flags state to render_preview
                render_preview(self.source, self._approved_flags(), clip_start, clip_end, preview)

                # Verify file exists and has content
                if not preview.exists() or preview.stat().st_size < 1000:
                    raise RuntimeError("Preview file is missing or corrupted.")
            except Exception as exc:
                QMessageBox.warning(self, "Preview failed", f"Could not create preview audio:\n\n{exc}")
                return

            # Small delay and process events to ensure Windows file system sync
            time.sleep(0.15)
            QApplication.processEvents()

            self.player.setSource(QUrl.fromLocalFile(str(preview)))
            self.pending_demo = (0, int((clip_end - clip_start) * 1000), flag.word, True, int(clip_start * 1000))

    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState: self.player.pause()
        else: self.player.play()

    def seek_media(self, position):
        # Position is 0-1000 based on scrubber
        if self.player.duration() > 0:
            # If we're playing a censored preview, we should probably stop and switch back to global for scrubbing?
            # Or at least understand the context.
            if hasattr(self, "is_preview") and self.is_preview:
                self.cleanup_preview()
                self.player.setSource(QUrl.fromLocalFile(str(self.source)))
                self.is_preview = False
            self.player.setPosition(int(position / 1000 * self.player.duration()))

    def update_timeline(self, position):
        if not self.timeline.isSliderDown():
            effective_pos = position
            if hasattr(self, "preview_offset") and self.preview_offset is not None:
                effective_pos += self.preview_offset

            total_dur = getattr(self, "source_duration", 0)
            if total_dur > 0:
                self.timeline.setValue(int(effective_pos / total_dur * 1000))

            cur = time.strftime('%M:%S', time.gmtime(effective_pos / 1000))
            dur = time.strftime('%M:%S', time.gmtime(total_dur / 1000))
            self.time_label.setText(f"{cur} / {dur}")

    def update_duration(self, duration):
        if not hasattr(self, "is_preview") or not self.is_preview:
            self.source_duration = duration
        self.timeline.setRange(0, 1000)

    def update_preview_filter(self):
        # If something was playing or paused as a preview, restart it with the new filter setting
        if self.flags_table.currentRow() >= 0:
            self.player.stop()
            self.demo_selected()

    def start_demo_when_ready(self, media_status):
        if not self.pending_demo or media_status not in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
            if media_status == QMediaPlayer.MediaStatus.LoadedMedia:
                self.play_pause_button.setText("Play")
            return

        # Unpack pending demo info
        info = self.pending_demo; self.pending_demo = None
        start = info[0]; duration = info[1]; word = info[2]; is_preview = info[3]
        self.preview_offset = info[4] if len(info) > 4 else None
        self.is_preview = is_preview

        self.player.setPosition(start)
        self.play_pause_button.setText("Play")
        self.demo_timer.start(duration)
        text = censor_text(word) if self.censor_ui_toggle.isChecked() else word
        self.status.setText(f"Cued: Context around {text}. Press Play to listen.")

    def playback_error(self, _error, message):
        if message: self.status.setText(f"Audio preview unavailable: {message}")

    def stop_qt_preview(self):
        self.player.stop(); self.cleanup_preview()

    def cleanup_preview(self, *_args):
        # The preview folder is now persistent to avoid Windows race conditions
        pass

    def apply_bulk_action(self):
        action = self.bulk_action_box.currentText(); changed = 0
        for row, flag in enumerate(self.flags):
            if self.flags_table.item(row, 0).checkState() == Qt.Checked:
                self.flags_table.cellWidget(row, 4).setCurrentText(action); flag.action = action; changed += 1
        self.status.setText(f"Set {changed} approved match(es) to {action}.")

    def save_review(self):
        if self.source: self.source.with_suffix(".nono.review.json").write_text(json.dumps({"source_file": str(self.source), "flags": [asdict(f) for f in self._approved_flags()]}, indent=2), encoding="utf-8")

    def _refresh_terms(self):
        self.terms_table.setRowCount(len(self.terms)); self.term_model.setStringList([term.term for term in self.terms])
        do_censor = self.censor_ui_toggle.isChecked()
        for row, term in enumerate(self.terms):
            text = censor_text(term.term) if do_censor else term.term
            for col, value in enumerate((text, term.match_type, term.action)): self.terms_table.setItem(row, col, QTableWidgetItem(value))

    def update_suggestions(self, text): self.term_model.setStringList(suggestions(text, self.terms) if text else [term.term for term in self.terms])

    def search_project(self):
        query = self.project_search.text().strip().lower()
        if not self.words: return
        if not query:
            self.show_transcript(); self.status.setText("Showing the complete transcript."); return
        matches = [word for word in self.words if query in word.word.lower()]
        self.chapters.setRowCount(len(matches))
        do_censor = self.censor_ui_toggle.isChecked()
        for row, word in enumerate(matches):
            text = censor_text(word.word) if do_censor else word.word
            self.chapters.setItem(row, 0, QTableWidgetItem(f"{word.start:0.2f}s  {text}"))
        self.status.setText(f"Project search found {len(matches)} occurrence(s) of '{query}'.")
    def add_term(self):
        term = self.term_input.text().strip()
        if not term: return
        new_term = Term(term, self.type_box.currentText(), self.action_box.currentText())
        try:
            save_terms(self.list_path, [*self.terms, new_term])
        except OSError as exc:
            QMessageBox.critical(self, "Could not save term", str(exc)); return
        self.terms.append(new_term); self.term_input.clear(); self._refresh_terms(); self.status.setText(f"Saved profanity list to {self.list_path}.")
    def remove_term(self):
        row = self.terms_table.currentRow()
        if row < 0: return
        updated = [term for index, term in enumerate(self.terms) if index != row]
        try:
            save_terms(self.list_path, updated)
        except OSError as exc:
            QMessageBox.critical(self, "Could not save list", str(exc)); return
        self.terms = updated; self._refresh_terms()
    def export(self):
        if not self.source: return
        ext = self.source.suffix.lower() if self.source.suffix.lower() in {".mp3", ".m4a", ".m4b", ".ogg", ".opus", ".flac", ".mp4", ".mkv", ".mov"} else ".mp3"

        # Explicit format options for the user
        filters = (
            "MP3 Audio (*.mp3);;"
            "M4B Audiobook (*.m4b);;"
            "M4A Audio (*.m4a);;"
            "OGG Vorbis (*.ogg);;"
            "Opus Audio (*.opus);;"
            "FLAC Lossless (*.flac);;"
            "WAV Audio (*.wav);;"
            "MP4 Video (*.mp4);;"
            "MKV Video (*.mkv);;"
            "MOV Video (*.mov)"
        )

        output, selected_filter = QFileDialog.getSaveFileName(self, "Save censored media", str(self.source.with_stem(self.source.stem + "_clean").with_suffix(ext)), filters)
        if not output: return

        out_path = Path(output)
        # Force the extension based on the selected filter if the user didn't provide one
        if not out_path.suffix:
            # Extract extension from the filter string (e.g. "*.mp3" -> ".mp3")
            # We map based on the known format names in the filter
            for fmt in ["mp3", "m4b", "m4a", "ogg", "opus", "flac", "wav", "mp4", "mkv", "mov"]:
                if fmt.upper() in selected_filter:
                    out_path = out_path.with_suffix(f".{fmt}")
                    break

        self.save_review()
        # Disable potentially conflicting actions
        for btn in [self.export_button, self.import_button, self.transcribe_button, self.match_button, self.demo_button]:
            btn.setEnabled(False)

        self.progress.setVisible(True); self.progress.setValue(0)
        self.status.setText(f"Exporting to {out_path.name}...")

        self.export_worker = ExportWorker(self.source, self._approved_flags(), out_path, self.segments)
        self.export_worker.progressed.connect(self.progress.setValue)
        self.export_worker.finished.connect(self.export_done)
        self.export_worker.failed.connect(self.export_failed)
        self.export_worker.start()

    def export_done(self):
        self.progress.setVisible(False)
        for btn in [self.export_button, self.import_button, self.transcribe_button, self.match_button, self.demo_button]:
            btn.setEnabled(True)
        self.status.setText("Export complete.")
        QMessageBox.information(self, "Export success", "The censored file has been saved successfully.")

    def export_failed(self, message):
        self.progress.setVisible(False)
        for btn in [self.export_button, self.import_button, self.transcribe_button, self.match_button, self.demo_button]:
            btn.setEnabled(True)
        self.status.setText("Export failed.")
        QMessageBox.critical(self, "Export failed", f"An error occurred during export:\n\n{message}")

def launch():
    app = QApplication.instance() or QApplication([]); window = MainWindow(); window.show(); return app.exec()
