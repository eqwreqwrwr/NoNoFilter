from __future__ import annotations

import re
from rapidfuzz.fuzz import ratio
from .models import Flag, Term, Word

def normalize(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())

def soundex(value: str) -> str:
    """Small dependency-free phonetic key for English ASR near-homophones."""
    value = normalize(value)
    if not value:
        return ""
    groups = {"bfpv": "1", "cgjkqsxz": "2", "dt": "3", "l": "4", "mn": "5", "r": "6"}
    code = {letter: digit for letters, digit in groups.items() for letter in letters}
    result, previous = value[0].upper(), code.get(value[0], "")
    for char in value[1:]:
        digit = code.get(char, "")
        if digit and digit != previous:
            result += digit
        previous = digit
    return (result + "000")[:4]

def find_matches(words: list[Word], terms: list[Term]) -> list[Flag]:
    flags: list[Flag] = []
    for word in words:
        token = normalize(word.word)
        if not token:
            continue
        for term in terms:
            target = normalize(term.term)
            if not target:
                continue
            similarity = ratio(token, target)
            exact = token == target
            phonetic = len(token) >= 3 and soundex(token) == soundex(target)
            matched = exact if term.match_type == "exact" else similarity >= term.threshold if term.match_type == "fuzzy" else (phonetic or similarity >= term.threshold)
            if matched:
                confidence = 1.0 if exact else max(similarity / 100, .80 if phonetic else 0)
                flags.append(Flag(word.word, word.start, word.end, term.term, term.action, confidence))
                break
    return flags

def suggestions(query: str, terms: list[Term], limit: int = 8) -> list[str]:
    q = normalize(query)
    ranked = sorted(terms, key=lambda t: ratio(q, normalize(t.term)), reverse=True)
    return [t.term for t in ranked if q in normalize(t.term) or ratio(q, normalize(t.term)) >= 45][:limit]
