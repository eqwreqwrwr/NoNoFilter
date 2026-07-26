from __future__ import annotations

from pathlib import Path
from .models import Flag

def _time(seconds: float, vtt: bool = False) -> str:
    millis = round(seconds * 1000)
    h, millis = divmod(millis, 3_600_000); m, millis = divmod(millis, 60_000); s, millis = divmod(millis, 1000)
    return f"{h:02}:{m:02}:{s:02}{'.' if vtt else ','}{millis:03}"

def export_subtitles(segments: list[dict], output: str | Path, flags: list[Flag] = [], vtt: bool = False) -> None:
    flagged = [(f.start, f.end) for f in flags if f.approved]
    rows = ["WEBVTT", ""] if vtt else []
    for index, segment in enumerate(segments, 1):
        text = segment.get("text", "").strip()
        if any(segment["start"] <= end and segment["end"] >= start for start, end in flagged):
            text = "[censored]"
        rows.extend(([str(index)] if not vtt else []) + [f"{_time(segment['start'], vtt)} --> {_time(segment['end'], vtt)}", text, ""])
    Path(output).write_text("\n".join(rows), encoding="utf-8")
