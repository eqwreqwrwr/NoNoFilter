from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

@dataclass(slots=True)
class Word:
    word: str
    start: float
    end: float
    score: float = 1.0

@dataclass(slots=True)
class Term:
    term: str
    match_type: str = "exact"  # exact | fuzzy | phonetic
    action: str = "bleep"      # bleep | mute
    threshold: int = 85

@dataclass(slots=True)
class Flag:
    word: str
    start: float
    end: float
    term: str
    action: str
    confidence: float
    approved: bool = True

def load_terms(path: str | Path) -> list[Term]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    # Accept both the compact user-authored ["term", ...] format and the
    # richer app format that stores matching and action preferences.
    return [Term(term=item) if isinstance(item, str) else Term(**item) for item in data.get("terms", [])]

def save_terms(path: str | Path, terms: list[Term]) -> None:
    Path(path).write_text(json.dumps({"terms": [asdict(t) for t in terms]}, indent=2), encoding="utf-8")

def load_transcript(path: str | Path) -> tuple[list[Word], list[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return [Word(**word) for word in data.get("words", [])], data.get("segments", [])

def save_flags(path: str | Path, flags: list[Flag]) -> None:
    Path(path).write_text(json.dumps({"flags": [asdict(f) for f in flags]}, indent=2), encoding="utf-8")

def load_flags(path: str | Path) -> list[Flag]:
    return [Flag(**item) for item in json.loads(Path(path).read_text(encoding="utf-8-sig")).get("flags", [])]
