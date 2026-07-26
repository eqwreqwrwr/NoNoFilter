from nono_filter.models import Word

def test_word_creation():
    # Simulate scaled Parakeet timestamps
    SCALE = 0.08
    raw_start = 25
    raw_end = 30
    chunk_offset = 120.0

    text = "test"
    w_start = raw_start * SCALE
    w_end = raw_end * SCALE

    word = Word(text, w_start + chunk_offset, w_end + chunk_offset, 1.0)

    print(f"Word: {word.word}")
    print(f"Start: {word.start}")
    print(f"End: {word.end}")

    assert word.start < word.end
    print("Verification successful: Start < End")

if __name__ == "__main__":
    test_word_creation()
