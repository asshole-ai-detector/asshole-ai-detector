"""RAG-02 sampling is independent of the model and Chroma runtime."""

from dataclasses import replace

import pytest

from asshole_ai_detector.rag.index import SamplingConfig, select_cases
from asshole_ai_detector.schemas.case import (
    VERDICTS,
    AITACase,
    AITAComment,
    DerivedFields,
)


def case(verdict, disputed, number):
    post_id = f"{verdict}-{disputed}-{number}"
    other = next(v for v in VERDICTS if v != verdict)
    comments = [
        AITAComment(
            comment_id=f"{post_id}#{i}",
            rank=i,
            verdict=other if disputed and i > 3 else verdict,
            text="A synthetic opinion.",
        )
        for i in range(1, 6)
    ]
    return AITACase(
        post_id=post_id,
        title="An invented conflict",
        text="An invented household situation for a test. " * 8,
        url=f"https://example.org/{post_id}",
        valid_comments=comments,
        derived_fields=DerivedFields.from_comments(comments),
    )


def test_sampling_cap_disputed_priority_and_order_independence():
    records = [
        case(v, is_disputed, i)
        for v, nd, nc in (("NTA", 5, 6), ("YTA", 1, 8), ("ESH", 0, 1), ("NAH", 3, 4))
        for is_disputed, n in ((True, nd), (False, nc))
        for i in range(n)
    ]
    config = SamplingConfig(seed=19, max_per_top_verdict=3)
    selected, counts = select_cases(records, config)
    assert select_cases(records[::-1], config) == (selected, counts)
    assert select_cases(records, replace(config, seed=3))[0] != selected
    assert counts["NTA"]["DISPUTED"] == {
        "eligible": 5,
        "selected": 3,
        "excluded_by_cap": 2,
    }
    assert counts["NTA"]["NTA"]["selected"] == 0
    assert counts["NAH"]["NAH"]["selected"] == 0
    assert counts["YTA"]["DISPUTED"]["selected"] == 1
    assert counts["YTA"]["YTA"]["selected"] == 2
    assert counts["ESH"]["ESH"]["selected"] == 1
    assert len({c.post_id for c in selected}) == len(selected) == 10
    assert [c.post_id for c in selected] == sorted(c.post_id for c in selected)


@pytest.mark.parametrize(
    "options",
    [
        {"seed": False},
        {"max_per_top_verdict": 0},
        {"max_per_top_verdict": 5001},
        {"max_per_top_verdict": True},
        {"max_posts": 19999},
        {"max_posts": 20001},
    ],
)
def test_invalid_sampling_config(options):
    with pytest.raises(ValueError):
        SamplingConfig(**options)


def test_duplicate_id_rejected_even_when_it_would_be_sampled_out():
    sample = case("NTA", False, 1)
    with pytest.raises(ValueError, match="Duplicate candidate"):
        select_cases(
            [case("NTA", True, 2), sample, sample],
            SamplingConfig(max_per_top_verdict=1),
        )


def test_empty_selection_records_all_four_classes():
    selected, counts = select_cases([], SamplingConfig())
    assert selected == []
    assert set(counts) == set(VERDICTS)
    assert all(
        entry["eligible"] == 0 for group in counts.values() for entry in group.values()
    )
