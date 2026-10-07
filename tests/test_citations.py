from app.chunking import Chunk
from app.index import SearchResult
from app.llm import NO_ANSWER, citation_check, is_refusal, judge


def test_full_coverage():
    answer = "Revenue grew 8 percent year over year [1]. Services drove most of the growth [1][2]."
    result = citation_check(answer, num_sources=3)
    assert result["cited_sources"] == [1, 2]
    assert result["invalid_citations"] == []
    assert result["citation_coverage"] == 1.0


def test_citation_after_period_counts():
    answer = "Revenue grew 8 percent year over year. [1] The company has many employees worldwide."
    assert citation_check(answer, num_sources=2)["citation_coverage"] == 0.5


def test_invalid_citation_flagged():
    result = citation_check("Margins expanded meaningfully this year [7].", num_sources=6)
    assert result["invalid_citations"] == [7]


def test_refusal_detection():
    assert is_refusal(NO_ANSWER)
    assert not is_refusal("Revenue grew 8 percent [1].")


class FakeClient:
    def __init__(self, reply: str):
        self.reply = reply

    def complete(self, system, user, max_tokens=1024):
        return self.reply


RESULTS = [SearchResult(Chunk(0, "Item 7", "Revenue grew 8 percent."), 0.03, 1, 1)]


def test_judge_parses_json_and_clamps():
    client = FakeClient('Here you go: {"groundedness": 1.4, "verdict": "grounded", "unsupported_claims": []}')
    result = judge(client, "q", "a [1]", RESULTS)
    assert result == {"groundedness": 1.0, "verdict": "grounded", "unsupported_claims": []}


def test_judge_handles_bad_output():
    result = judge(FakeClient("not json"), "q", "a", RESULTS)
    assert result["verdict"] == "judge_error"
    assert result["groundedness"] is None
