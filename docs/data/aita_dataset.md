# AITA dataset: structure and observations (DATA-01)

Source: [MattBoraske/reddit-AITA-submissions-and-comments-multiclass](https://huggingface.co/datasets/MattBoraske/reddit-AITA-submissions-and-comments-multiclass),
pinned at revision `84c229ffd926e8efaa1828a5d3803f29231f2d0e` (see `configs/default.yaml`).

Loading and normalization: `asshole_ai_detector.data.aita_adapter` (install with `uv sync --group data`).

## How the numbers were obtained

- Column statistics (types, null counts, label frequencies, lengths): Hugging Face dataset viewer, **all 50,000 rows**.
- Row-level checks (empty posts, deleted comments, consensus labels): **full `test` split (10,000 rows)** and the
  first 6,700 rows of `train`. The `train` rows are sorted by score, so the `test` numbers are the reference.

Re-run `notebooks/data_exploration.ipynb` to reproduce the counts on the full dataset.

## Structure

One row = one submission. Comments are stored as flat columns.

| Column | Type | Notes |
| --- | --- | --- |
| `submission_title` | string | Original casing. Never null. |
| `submission_text` | string | Already **lowercased**. Empty string for ~1.1% of rows (removed posts, title only). |
| `submission_score` | int | 145 to ~81k. Not part of our schema. |
| `submission_url` | string | Reddit permalink. The post ID is extracted from it (`/comments/<id>/`). |
| `submission_date` | string | `YYYY-MM-DD HH:MM:SS`, no timezone (treated as UTC). Range 2019-01-01 to 2022-12-31. |
| `top_comment_1..10` | string | Ranked comments, already lowercased. Later ranks are often null. |
| `top_comment_1..10_classification` | string | `NTA`, `YTA`, `ESH`, `NAH`, `INFO` or null. |
| `ambiguity_score` | float | 0 to 1, > 0 for ~42% of rows. Kept only as `provider_ambiguity`. |
| `flanT5_instruction`, `llama2_instruction` | string | Ready-made fine-tuning prompts. Ignored. |

Provider splits: `train` 40,000 and `test` 10,000, with no shared post IDs. We merge them; our own Golden,
development and index splits are created later.

## Observations

- **No ID column.** The Reddit post ID is taken from the URL; this worked for every row checked.
- **Null labels.** A null label means no verdict could be extracted from the comment (jokes, questions,
  off-topic replies). Labels look regex-extracted: `ywnbta` and `n t a` map to `NTA`.
- **Deleted comments.** `[deleted]` / `[removed]` comments exist (~1,700 in `test`) and always have a null label.
  A few dozen comments are empty strings. No label appears without comment text.
- **Length.** Posts are at most ~3,400 characters (~650 words) and end naturally, so the dataset looks
  length-filtered rather than truncated. The 1,024-token RAG cap is never reached, but ~1/3 of posts have
  more than 400 words and will be truncated by RoBERTa (512 tokens).
- **Class imbalance.** With our rule (>= 5 valid verdicts, >= 70% agreement), the `test` split gives:

  | NTA | YTA | ESH | NAH | DISPUTED | < 5 valid |
  | --- | --- | --- | --- | --- | --- |
  | 7,396 | 1,016 | 52 | 81 | 1,160 | 295 |

  Extrapolated to 50,000 rows, only ~250-300 ESH and ~400 NAH consensus posts exist. The Golden Dataset
  needs 40 of each, so few will remain for the development set and the index.

## Normalized representation

`row_to_normalized` returns a `NormalizedAITACase` (see `schemas/case.py`). Nothing is filtered:

- `post_id` comes from the URL (`None` if it cannot be extracted).
- Empty or whitespace-only strings become `None` (title, text, URL, comment text and labels).
- Dates are parsed and marked as UTC; malformed dates become `None`.
- Comment slots where both text and label are missing are skipped. All other comments keep their
  original `rank`, raw label (including `INFO` and null) and text (including `[deleted]`).
- `ambiguity_score` becomes `provider_ambiguity` (NaN becomes `None`).
