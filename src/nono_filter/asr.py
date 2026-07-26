from __future__ import annotations
import sys
import os
import re
import math
import json
import time
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from dataclasses import asdict

# Workaround for Python 3.12+ compatibility with legacy 'six' importers
for importer in sys.meta_path:
    if "SixMetaPathImporter" in str(type(importer)):
        if not hasattr(importer, "_path"):
            importer._path = []

# Emergency import for compatibility with NumPy 1.x and 2.x legacy aliases
try:
    import warnings
    import numpy as np
    # Silence technical FutureWarnings from the bridge logic
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        # Restore legacy aliases removed in NumPy 1.24+
        if not hasattr(np, 'long'): np.long = int
        if not hasattr(np, 'float'): np.float = float
        if not hasattr(np, 'int'): np.int = int
        if not hasattr(np, 'bool'): np.bool = bool
        if not hasattr(np, 'object'): np.object = object
        # Add new aliases introduced in NumPy 2.0 (for compatibility with newer libs on older NumPy)
        if not hasattr(np, 'ulong'): np.ulong = getattr(np, 'uint', getattr(np, 'uint64', None))

    # Fix numpy.exceptions mismatch
    try:
        import numpy.exceptions as np_exc
        if not hasattr(np_exc, 'RankWarning'): np_exc.RankWarning = np.RankWarning
    except ImportError:
        class FakeExc: pass
        fe = FakeExc()
        fe.RankWarning = np.RankWarning
        sys.modules['numpy.exceptions'] = fe
except ImportError:
    pass

from .models import Word
from .subtitles import export_subtitles

logger = logging.getLogger("nono_filter.asr")

_MODEL_CACHE = {}

def _save_transcript(output: str | Path, source: str | Path, words: list[Word], rows: list[dict], complete: bool) -> dict:
    """Atomically checkpoint transcript data so a close/crash cannot corrupt it."""
    result = {"source_file": str(source), "complete": complete, "words": [asdict(word) for word in words], "segments": rows}
    target = Path(output); temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(target)
    return result

class TranscriptionCancelled(Exception):
    pass

def _transcribe_whisper(model_size: str, device: str, source_path: str, prompt: str | None, status):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install faster-whisper to transcribe locally: pip install faster-whisper") from exc

    cache_key = ("whisper", model_size, device)
    if cache_key in _MODEL_CACHE:
        logger.info(f"Using cached {model_size} Whisper model on {device}.")
        model = _MODEL_CACHE[cache_key]
    else:
        compute_type = "float16" if device == "cuda" else "int8"
        if status: status(f"Loading {model_size} speech model on {device.upper()}...")
        logger.info(f"Loading {model_size} Whisper model on {device}...")
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            _MODEL_CACHE[cache_key] = model
        except RuntimeError as exc:
            if device == "cuda" and "cublas" in str(exc).lower():
                raise RuntimeError("CUDA was selected, but its runtime library (cublas64_12.dll) is unavailable. Select CPU, or install a CUDA 12-compatible NVIDIA runtime.") from exc
            raise

    segments, info = model.transcribe(source_path, word_timestamps=True, initial_prompt=prompt)
    return segments, info.duration

def _transcribe_parakeet_chunked(model, source_path: str, output: str | Path, source: str | Path, start_from: float, progress, partial, status, paused, cancelled):
    import torch
    import soundfile as sf
    from omegaconf import open_dict

    # 1. Ensure word timestamps are enabled in the model configuration
    try:
        if hasattr(model, 'cfg') and hasattr(model.cfg, 'decoding'):
            with open_dict(model.cfg.decoding):
                model.cfg.decoding.preserve_alignments = True
                model.cfg.decoding.compute_timestamps = True
            model.change_decoding_strategy(model.cfg.decoding)
    except Exception as e:
        logger.warning(f"Could not enable word timestamps for Parakeet: {e}")

    with sf.SoundFile(source_path) as f:
        duration = len(f) / f.samplerate

    # 5-minute chunks
    chunk_sec = 300
    num_chunks = math.ceil(duration / chunk_sec)

    all_words = []; all_rows = []

    for i in range(num_chunks):
        if cancelled and cancelled.is_set(): break
        while paused and paused.is_set(): time.sleep(.1)

        c_start = i * chunk_sec
        c_dur = min(chunk_sec, duration - c_start)
        if status: status(f"Parakeet: Analyzing chunk {i+1}/{num_chunks} ({c_start/60:.1f}m)...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # IMPORTANT: Use OUTPUT seeking (-ss after -i) for sample-accurate slices.
            # This prevents beeps from "drifting" or moving to random locations in long files.
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", source_path, "-ss", str(c_start), "-t", str(c_dur), tmp_path], check=True)

            with torch.no_grad():
                results = model.transcribe([tmp_path], return_hypotheses=True)
            if isinstance(results, tuple): results = results[0]
            hyp = results[0]
            if isinstance(hyp, list) and len(hyp) > 0: hyp = hyp[0]

            # Offset timestamps back to global file time
            chunk_offset = c_start + start_from

            # Unit Detection Heuristic: Detect if model uses 10ms, 40ms, or 80ms units.
            word_data = []
            max_raw_end = 0.0

            ts = getattr(hyp, 'timestep', {})
            w_list = ts.get('word', []) if isinstance(ts, dict) else getattr(ts, 'word', [])
            if not w_list and hasattr(hyp, 'words'): w_list = hyp.words

            for w in w_list:
                w_text = w.get('word') if isinstance(w, dict) else getattr(w, 'word', '')
                w_start = w.get('start_offset') if isinstance(w, dict) else getattr(w, 'start_offset', None)
                if w_start is None: w_start = w.get('start_time') if isinstance(w, dict) else getattr(w, 'start_time', 0.0)
                w_end = w.get('end_offset') if isinstance(w, dict) else getattr(w, 'end_offset', None)
                if w_end is None: w_end = w.get('end_time') if isinstance(w, dict) else getattr(w, 'end_time', w_start + 0.5)
                word_data.append({'text': w_text, 'start': w_start, 'end': w_end})
                max_raw_end = max(max_raw_end, w_end)

            # CALIBRATION: If max_raw_end is much larger than chunk duration, it's frames.
            SCALE = 1.0
            if max_raw_end > c_dur * 1.1:
                potential = c_dur / max_raw_end
                if potential < 0.02: SCALE = 0.01      # 10ms frames
                elif potential < 0.05: SCALE = 0.04    # 40ms frames
                else: SCALE = 0.08                     # 80ms frames (FastConformer)
                logger.info(f"Parakeet Calibrated: {SCALE}s units (potential={potential:0.3f})")

            for w in word_data:
                text = w['text'].strip()
                if not text: continue

                raw_start, raw_end = w['start'] * SCALE, w['end'] * SCALE
                all_words.append(Word(text, raw_start + chunk_offset, raw_end + chunk_offset, 1.0))

            if hasattr(hyp, 'text') and hyp.text:
                all_rows.append({"start": chunk_offset, "end": chunk_offset + c_dur, "text": hyp.text})
                if partial: partial(hyp.text)
                _save_transcript(output, source, all_words, all_rows, complete=False)

            if progress:
                p_val = int(((c_start + c_dur) / duration) * 100)
                progress(p_val)
        finally:
            try: os.unlink(tmp_path)
            except: pass

    return all_words, all_rows, duration

def transcribe(source: str | Path, output: str | Path, model_size: str = "small", progress=None, partial=None, status=None, paused=None, cancelled=None, device: str = "cpu", start_from: float = 0, prompt: str | None = None, existing_words: list[Word] | None = None, existing_rows: list[dict] | None = None) -> dict:
    source_path = str(source)
    holder = None

    try:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg is required for transcription.")

        holder = tempfile.TemporaryDirectory()
        optimized_audio = Path(holder.name) / "optimized.wav"

        if status: status("Optimizing audio for AI (extracting 16kHz mono track)...")
        logger.info(f"Extracting 16kHz mono track from {source.name}...")

        cmd = ["ffmpeg", "-v", "error", "-y"]
        if start_from > 0: cmd += ["-ss", str(start_from)]
        cmd += ["-i", str(source), "-ar", "16000", "-ac", "1", str(optimized_audio)]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {result.returncode}. Error: {result.stderr}")
        source_path = str(optimized_audio)
    except Exception as exc:
        if holder: holder.cleanup()
        raise RuntimeError(f"Audio pre-processing failed: {exc}")

    words = list(existing_words) if existing_words else []
    rows = list(existing_rows) if existing_rows else []

    try:
        if "parakeet" in model_size:
            nemo_name = f"nvidia/{model_size}"
            cache_key = ("nemo", nemo_name, device)

            if cache_key in _MODEL_CACHE:
                model = _MODEL_CACHE[cache_key]
            else:
                import nemo.collections.asr as nemo_asr
                if status: status(f"Loading {nemo_name} on {device.upper()}...")
                model = nemo_asr.models.ASRModel.from_pretrained(model_name=nemo_name)
                model = model.to(device); model.eval()
                _MODEL_CACHE[cache_key] = model

            p_words, p_rows, _ = _transcribe_parakeet_chunked(model, source_path, output, source, start_from, progress, partial, status, paused, cancelled)
            words.extend(p_words)
            rows.extend(p_rows)
        else:
            segments, info_duration = _transcribe_whisper(model_size, device, source_path, prompt, status)

            if status: status("Analyzing audio...")
            logger.info("Starting transcription...")

            last_save = time.monotonic()
            for segment in segments:
                while paused and paused.is_set() and not (cancelled and cancelled.is_set()):
                    time.sleep(.1)
                if cancelled and cancelled.is_set():
                    raise TranscriptionCancelled("Transcription stopped by user.")

                start, end = segment.start + start_from, segment.end + start_from
                rows.append({"start": start, "end": end, "text": segment.text})
                for word in segment.words or []:
                    words.append(Word(word.word.strip(), word.start + start_from, word.end + start_from, word.probability))

                now = time.monotonic()
                if now - last_save > 60:
                    _save_transcript(output, source, words, rows, complete=False)
                    last_save = now

                if progress:
                    total_dur = info_duration + start_from if start_from > 0 else info_duration
                    progress(int(min(99, end / max(total_dur, .1) * 100)))
                if partial: partial(segment.text.strip())
                if status: status(f"Transcribed through {end / 60:.1f} minutes of audio.")
                logger.debug(f"Progress: {end / 60:.1f}m")
    finally:
        if holder:
            try: holder.cleanup()
            except: pass

    result = _save_transcript(output, source, words, rows, complete=True)
    export_subtitles(rows, Path(output).with_suffix(".srt"))
    if progress: progress(100)
    return result
