"""DATA-04: tests for exact and transitive deduplication."""

from pydantic import HttpUrl

from asshole_ai_detector.data.deduplication import (
    content_hash,
    deduplicate_cases,
    normalize_for_dedup,
)
from asshole_ai_detector.schemas.annotation import Verdict
from asshole_ai_detector.schemas.case import AITACase, AITAComment, DerivedFields


def make_case(post_id: str, topic: str = "rent", *, text: str | None = None) -> AITACase:
    story = text or (
        f"My roommate and I disagreed about {topic}. "
        + "We talked about the conflict and considered each other's points of view. " * 6
    )
    verdicts: list[Verdict] = ["NTA", "NTA", "NTA", "NTA", "YTA"]
    comments = [
        AITAComment(
            comment_id=f"{post_id}#{rank}",
            rank=rank,
            verdict=verdict,
            text=f"A test comment with rank {rank}",
        )
        for rank, verdict in enumerate(verdicts, start=1)
    ]
    return AITACase(
        post_id=post_id,
        title="A synthetic AITA case",
        text=story,
        url=HttpUrl(f"https://example.com/{post_id}"),
        valid_comments=comments,
        derived_fields=DerivedFields.from_comments(comments),
    )


def test_duplicate_id_even_if_story_differs() -> None:
    result = deduplicate_cases([make_case("a", "rent"), make_case("a", "chores")])
    assert len(result.cases) == 1
    assert result.removed_count == 1
    assert result.duplicates[0].reason == "duplicate_id"
    assert result.duplicates[0].duplicate_of == "a"


def test_duplicate_content_with_different_ids() -> None:
    first = make_case("a", "rent")
    second = make_case("b", text=first.text)
    result = deduplicate_cases([second, first])
    assert [case.post_id for case in result.cases] == ["a"]
    assert result.duplicates[0].reason == "duplicate_content"
    assert result.duplicates[0].duplicate_of == "a"


def test_whitespace_case_and_paragraphs_do_not_change_hash() -> None:
    first = make_case("a", "car")
    second = make_case("b", text=first.text.upper().replace(" ", "\n\n   "))
    assert normalize_for_dedup(first.text) == normalize_for_dedup(second.text)
    assert content_hash(first) == content_hash(second)
    assert deduplicate_cases([first, second]).removed_count == 1


def test_same_content_different_titles_is_duplicate() -> None:
    first = make_case("a")
    second = make_case("b", text=first.text)
    second.title = "A different title"
    assert deduplicate_cases([first, second]).removed_count == 1


def test_distinct_stories_are_kept() -> None:
    result = deduplicate_cases([make_case("a", "rent"), make_case("b", "chores")])
    assert [case.post_id for case in result.cases] == ["a", "b"]
    assert not result.duplicates


def test_transitive_duplicate_chain_is_one_group() -> None:
    """Same ID connects two stories; story B also matches another post ID."""
    a_rent = make_case("a", "rent")
    a_chores = make_case("a", "chores")
    b_chores = make_case("b", text=a_chores.text)

    result = deduplicate_cases([a_rent, b_chores, a_chores])

    assert len(result.cases) == 1
    assert result.removed_count == 2
    assert {record.post_id for record in result.duplicates} == {"a", "b"}


def test_deterministic_with_reordered_inputs() -> None:
    a = make_case("a", "rent")
    b = make_case("b", text=a.text)
    c = make_case("c", "chores")

    first = deduplicate_cases([a, b, c])
    second = deduplicate_cases([c, b, a])

    assert [case.post_id for case in first.cases] == ["a", "c"]
    assert [case.model_dump_json() for case in first.cases] == [
        case.model_dump_json() for case in second.cases
    ]
    assert first.duplicates == second.duplicates


def test_preferred_ids_can_preserve_test_representation() -> None:
    a = make_case("a")
    z = make_case("z", text=a.text)
    result = deduplicate_cases([a, z], preferred_ids={"z"})
    assert [case.post_id for case in result.cases] == ["z"]
    assert result.duplicates[0].duplicate_of == "z"


def test_report_and_idempotency() -> None:
    story = make_case("a")
    result = deduplicate_cases([story, story, make_case("b", text=story.text)])
    assert result.input_count == 3
    assert result.removed_count == 2
    assert sorted(item.reason for item in result.duplicates) == [
        "duplicate_content",
        "duplicate_id",
    ]
    assert deduplicate_cases(result.cases).removed_count == 0


def test_empty_input() -> None:
    result = deduplicate_cases([])
    assert result.input_count == 0
    assert result.removed_count == 0