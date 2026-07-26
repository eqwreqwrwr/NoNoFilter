from __future__ import annotations

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

from pathlib import Path
import math, shutil, subprocess, tempfile, json, logging
import numpy as np
import soundfile as sf
from .models import Flag

logger = logging.getLogger("nono_filter")

def _wav_source(source: str | Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    source = Path(source)
    if source.suffix.lower() in {".wav", ".flac"}:
        return source, None
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required for MP3/M4A/M4B audio. Install it and add it to PATH.")
    holder = tempfile.TemporaryDirectory(); wav = Path(holder.name) / "source.wav"
    logger.debug(f"Converting {source.name} to WAV for analysis...")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), str(wav)], check=True, capture_output=True)
    return wav, holder

def _apply_edits(audio, rate: int, flags: list[Flag], offset: float = 0, padding: float = .015, fade: float = .015, offset_samples: int = 0) -> None:
    for flag in flags:
        if not flag.approved: continue
        start = max(0, int((flag.start - offset - padding) * rate)); end = min(len(audio), int((flag.end - offset + padding) * rate))
        if end <= start: continue
        length = end - start; edge = min(int(fade * rate), length // 2)
        action = flag.action.lower().strip()
        logger.debug(f"Applying {action} at {flag.start:0.2f}s (term: {flag.term})")
        if action == "bleep":
            # High-precision localized phase calculation to prevent pitch jitter in long files.
            # We calculate cycles (sample_index * frequency / rate) and use modulo 1.0
            # to keep the input to the sine function small and accurate.
            samples = np.arange(start, end, dtype=np.float64) + offset_samples
            cycles = (samples * 1000.0 / rate) % 1.0
            tone = (.20 * np.sin(2 * math.pi * cycles))[:, None].astype(np.float32)
            replacement = np.repeat(tone, audio.shape[1], axis=1)
        else: # mute
            replacement = np.zeros((length, audio.shape[1]), dtype=audio.dtype)
        envelope = np.ones(length, dtype=np.float32)
        if edge: envelope[:edge] = np.linspace(0, 1, edge, dtype=np.float32); envelope[-edge:] = np.linspace(1, 0, edge, dtype=np.float32)
        audio[start:end] = audio[start:end] * (1 - envelope[:, None]) + replacement * envelope[:, None]

def render(source: str | Path, flags: list[Flag], output: str | Path, padding: float = .015, fade: float = .015, progress_cb=None) -> None:
    source = Path(source); output = Path(output)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        logger.warning("FFmpeg/FFprobe not found. Attempting in-memory processing (limited to small files).")
        # Fallback to in-memory processing if FFmpeg is missing (limited to small files)
        wav, holder = _wav_source(source)
        try:
            audio, rate = sf.read(wav, always_2d=True)
            _apply_edits(audio, rate, flags, padding=padding, fade=fade)
            sf.write(output, audio, rate)
            if progress_cb: progress_cb(100)
            return
        finally:
            if holder: holder.cleanup()

    # Stream through FFmpeg to handle large files and compressed formats efficiently
    logger.info(f"Analyzing {source.name}...")
    # Use robust ffprobe command to get duration and audio stream info
    probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-show_entries", "stream=codec_type,channels,sample_rate", "-of", "json", str(source)]
    probe = subprocess.check_output(probe_cmd)
    data = json.loads(probe)

    audio_info = next((s for s in data.get("streams", []) if s["codec_type"] == "audio"), None)
    if not audio_info: raise ValueError("No audio stream found in source.")

    has_video = any(s["codec_type"] == "video" for s in data.get("streams", []))
    channels = int(audio_info["channels"]); rate = int(audio_info["sample_rate"])
    try:
        total_duration = float(data.get("format", {}).get("duration", 0))
    except (ValueError, TypeError):
        total_duration = 0

    logger.info(f"Rendering censored {'video' if has_video else 'audio'} to {output.name}...")

    # input: extract raw float32 audio. Force probed rate and channels for perfect alignment.
    in_cmd = ["ffmpeg", "-v", "error", "-i", str(source), "-f", "f32le", "-ar", str(rate), "-ac", str(channels), "-acodec", "pcm_f32le", "-"]
    in_proc = subprocess.Popen(in_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    out_cmd = ["ffmpeg", "-y", "-f", "f32le", "-ar", str(rate), "-ac", str(channels), "-i", "-", "-i", str(source)]
    out_cmd += ["-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"]

    if has_video:
        logger.debug("Copying video stream...")
        out_cmd += ["-map", "1:v", "-c:v", "copy"]

    ext = output.suffix.lower()
    if ext == ".mp3": out_cmd += ["-c:a", "libmp3lame", "-q:a", "2"]
    elif ext in {".m4a", ".m4b", ".mp4", ".mov"}: out_cmd += ["-c:a", "aac", "-b:a", "192k"]
    elif ext in {".ogg", ".mkv"}: out_cmd += ["-c:a", "libvorbis", "-q:a", "4"]
    elif ext == ".opus": out_cmd += ["-c:a", "libopus", "-b:a", "128k"]
    elif ext == ".flac": out_cmd += ["-c:a", "flac"]
    elif ext == ".wav": out_cmd += ["-c:a", "pcm_s16le"]

    out_cmd.append(str(output))

    # Use a temp file for stderr to avoid pipe buffer blocking on long runs
    err_file = tempfile.NamedTemporaryFile(delete=False)
    out_proc = subprocess.Popen(out_cmd, stdin=subprocess.PIPE, stderr=err_file)

    try:
        chunk_size = 1024 * 256; frame_bytes = channels * 4; chunk_bytes = chunk_size * frame_bytes
        offset_samples = 0
        while True:
            raw = in_proc.stdout.read(chunk_bytes)
            if not raw: break

            # Ensure we have a complete set of frames
            valid_len = (len(raw) // frame_bytes) * frame_bytes
            if valid_len == 0: break

            block = np.frombuffer(raw[:valid_len], dtype="float32").reshape(-1, channels).copy()
            _apply_edits(block, rate, flags, offset=offset_samples/rate, padding=padding, fade=fade, offset_samples=offset_samples)

            try:
                out_proc.stdin.write(block.tobytes())
            except (BrokenPipeError, OSError):
                break

            offset_samples += len(block)
            if progress_cb and total_duration > 0:
                progress_cb(int(min(99, (offset_samples / rate) / total_duration * 100)))

        out_proc.stdin.close()
        in_proc.terminate()
        out_proc.wait()

        if out_proc.returncode != 0:
            err_file.close()
            err_msg = Path(err_file.name).read_text(errors="replace")
            raise RuntimeError(f"FFmpeg failed with exit code {out_proc.returncode}. Error: {err_msg}")

        if progress_cb: progress_cb(100)
        logger.info(f"Export complete: {output.name}")
    finally:
        err_file.close()
        try: Path(err_file.name).unlink()
        except: pass

def render_preview(source: str | Path, flags: list[Flag], clip_start: float, clip_end: float, output: str | Path) -> None:
    """Create a short, censored WAV clip without altering the source audiobook."""
    clip_start = max(0, clip_start); duration = max(.1, clip_end - clip_start)
    source = Path(source); holder = None
    try:
        if source.suffix.lower() in {".wav", ".flac"}:
            with sf.SoundFile(source) as stream:
                stream.seek(int(clip_start * stream.samplerate)); audio = stream.read(int(duration * stream.samplerate), always_2d=True); rate = stream.samplerate
        else:
            if not shutil.which("ffmpeg"): raise RuntimeError("FFmpeg is required to preview MP3/M4A/M4B audio.")
            holder = tempfile.TemporaryDirectory(); clip = Path(holder.name) / "preview-source.wav"
            # Re-encode short clip for maximum compatibility with Qt Multimedia
            subprocess.run(["ffmpeg", "-y", "-ss", str(clip_start), "-t", str(duration), "-i", str(source), "-ac", "2", "-ar", "44100", str(clip)], check=True, capture_output=True)
            audio, rate = sf.read(clip, always_2d=True)
        _apply_edits(audio, rate, flags, offset=clip_start)
        # Force 16-bit PCM for the preview WAV to ensure Qt can play it.
        # We explicitly use a with-block or ensure the file is closed before returning.
        with sf.SoundFile(output, mode='w', samplerate=rate, channels=audio.shape[1], subtype='PCM_16') as f:
            f.write(audio)
    finally:
        if holder: holder.cleanup()
