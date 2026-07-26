from nono_filter.matching import find_matches, soundex
from nono_filter.models import Term, Word

def test_soundex_handles_similar_spellings():
    assert soundex("damn") == soundex("dam")

def test_exact_and_phonetic_matches():
    words = [Word("ship", 0, 1), Word("dam", 1, 2)]
    flags = find_matches(words, [Term("damn", "phonetic"), Term("ship", "exact", "mute")])
    assert [item.term for item in flags] == ["ship", "damn"]
    assert flags[0].action == "mute"
