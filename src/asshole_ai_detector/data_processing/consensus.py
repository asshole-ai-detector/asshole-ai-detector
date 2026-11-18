
"""Consensus labeling for cleaned AITA cases."""

from collections.abc import Iterable

from asshole_ai_detector.schemas.case import (
    AITACase,
    AITAComment,
    DerivedFields,
    NormalizedAITACase,
)


def label_cleaned_case(
    case: NormalizedAITACase,
    valid_comments: Iterable[AITAComment],
) -> AITACase:
    """Build a labeled case from an already cleaned submission.

    DATA-02 is responsible for filtering invalid comments and posts.
    This function derives consensus from the valid moral verdicts
    and validates the resulting AITACase.
    """

    comments = [
        AITAComment.model_validate(comment)
        for comment in valid_comments
    ]

    derived_fields = DerivedFields.from_comments(comments)

    return AITACase.model_validate(
        {
            "post_id": case.post_id,
            "title": case.title,
            "text": case.text,
            "url": case.url,
            "date": case.date,
            "provider_ambiguity": case.provider_ambiguity,
            "valid_comments": comments,
            "derived_fields": derived_fields,
        }
    )
