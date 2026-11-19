
"""Consensus labeling for cleaned AITA cases."""

from asshole_ai_detector.data.cleaning import CleanedAITACase
from asshole_ai_detector.schemas.case import (
    AITACase,
    AITAComment,
    DerivedFields,
)


def label_cleaned_case(case: CleanedAITACase) -> AITACase:
    """Build a validated AITACase from a cleaned submission.

    DATA-02 has already filtered invalid comments and posts.
    DATA-03 converts the surviving comments into final AITAComment
    objects, assigns their IDs, and calculates consensus.
    """

    # Construct final comments from the single cleaned comment list.
    # Original ranks are preserved, including gaps from filtering.
    comments = [
        AITAComment.model_validate(
            {
                "comment_id": f"{case.post_id}#{comment.rank}",
                "rank": comment.rank,
                "verdict": comment.verdict,
                "text": comment.text,
            }
        )
        for comment in case.comments
    ]

    # Calculate the verdict distribution, plurality verdict,
    # consensus label, and number of valid comments.
    derived_fields = DerivedFields.from_comments(comments)

    # Validate the completed case against the shared schema.
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
