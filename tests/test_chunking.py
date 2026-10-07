from app.chunking import chunk_text, split_sections


def words(n: int, word: str = "lorem") -> str:
    return " ".join([word] * n)


SAMPLE = "\n".join([
    "Apple Inc. Annual Report",
    "Item 1. Business",
    words(500, "business"),
    "Item 1A. Risk Factors",
    words(120, "risk"),
    "Item 1B.",              # table-of-contents stub with no body
    "Item 7. Management's Discussion and Analysis",
    words(80, "revenue"),
])


def test_sections_detected():
    names = [name for name, _ in split_sections(SAMPLE)]
    assert names == [
        "Cover page",
        "Item 1. Business",
        "Item 1A. Risk Factors",
        "Item 7. Management's Discussion and Analysis",
    ]


def test_chunks_stay_within_sections_and_size():
    chunks = chunk_text(SAMPLE, max_words=200, overlap=50, min_words=40)
    assert all(len(c.text.split()) <= 200 for c in chunks)
    for c in chunks:
        unique = set(c.text.split())
        assert len(unique) == 1  # no chunk mixes two sections
    sections = [c.section for c in chunks]
    assert sections.count("Item 1. Business") == 3  # 500 words, step 150 -> windows at 0, 150, 300
    assert "Item 1A. Risk Factors" in sections
    assert "Cover page" not in sections  # too short, skipped


def test_ids_are_sequential():
    chunks = chunk_text(SAMPLE, max_words=200, overlap=50)
    assert [c.id for c in chunks] == list(range(len(chunks)))


def test_long_line_is_not_a_heading():
    text = "Item 7 " + words(200, "discussion")
    names = [name for name, _ in split_sections(text)]
    assert names == ["Cover page"]
