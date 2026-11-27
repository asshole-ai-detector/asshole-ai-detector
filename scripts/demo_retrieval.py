
"""RAG-03: reproducible ChromaDB retrieval demonstration.

Uses synthetic cases and deterministic toy vectors.
Does not use the real Qwen3 embedding model.
"""

import chromadb

from asshole_ai_detector.rag.retrieval import retrieve_dense
from asshole_ai_detector.schemas.case import (
    AITACase,
    AITAComment,
    DerivedFields,
)

QUERY_PREFIX = (
    "Instruct: Given a description of an interpersonal conflict, "
    "retrieve similar conflicts\nQuery: "
)


def toy_embedding(text: str) -> list[float]:
    """Create a deterministic, normalized 1024D test vector."""
    text = text.lower()

    if "roommate" in text:
        position = 0
    elif "work" in text or "coworker" in text:
        position = 1
    else:
        position = 2

    vector = [0.0] * 1024
    vector[position] = 1.0
    return vector


def embed_query(query: str) -> list[float]:
    """Apply the documented query prefix."""
    return toy_embedding(QUERY_PREFIX + query)


def make_case(post_id: str, title: str) -> AITACase:
    comments = [
        AITAComment(
            comment_id=f"{post_id}#{rank}",
            rank=rank,
            verdict="NTA",
            text=f"Synthetic comment {rank}",
        )
        for rank in range(1, 6)
    ]

    return AITACase(
        post_id=post_id,
        title=title,
        text=" ".join(["synthetic example"] * 30),
        url=f"https://example.com/{post_id}",
        valid_comments=comments,
        derived_fields=DerivedFields.from_comments(comments),
    )


def main() -> None:
    cases = {
        "post-roommate": make_case(
            "post-roommate", "Conflict with my roommate"
        ),
        "post-work": make_case(
            "post-work", "Argument with a coworker at work"
        ),
        "post-family": make_case(
            "post-family", "Disagreement with my family"
        ),
    }

    client = chromadb.EphemeralClient()

    collection = client.create_collection(
        name="rag03_demo",
        configuration={"hnsw": {"space": "cosine"}},
    )

    for post_id, case in cases.items():
        collection.add(
            ids=[post_id],
            embeddings=[
                toy_embedding(f"{case.title}\n{case.text}")
            ],
            metadatas=[{
                "post_id": post_id,
                "top_verdict": case.derived_fields.top_verdict,
                "consensus_label": case.derived_fields.consensus_label,
            }],
        )

    queries = [
        ("My roommate and I argued about rent", "post-roommate"),
        ("I had a problem with my coworker", "post-work"),
        ("I argued with my family", "post-family"),
    ]

    print("Indexed posts:", collection.count())

    for query, expected_id in queries:
        hits = retrieve_dense(
            query,
            embed_query=embed_query,
            index=collection,
            get_case=cases.get,
            top_k=3,
        )

        print(f"\nQuery: {query}")

        for hit in hits:
            print(
                f"  {hit.post_id}: "
                f"cosine distance={hit.cosine_distance:.4f}"
            )

        assert hits
        assert hits[0].post_id == expected_id
        assert abs(hits[0].cosine_distance) < 0.0001

    print("\nAll 3 ChromaDB retrieval demonstrations passed!")


if __name__ == "__main__":
    main()
