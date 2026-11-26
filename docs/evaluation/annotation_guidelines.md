
# EVAL-01 — Golden Dataset Annotation Guidelines

**Status:** Draft for team review

**Scope:** Initial candidate review and preparation for independent annotation

## 1. Purpose

The Golden Dataset is a held-out collection of 200 AITA
interpersonal-conflict scenarios for final evaluation.

Community verdicts are reference signals, not objective
moral truth.

The Golden Dataset must never be used for training,
model selection, threshold tuning, or RAG indexing.

## 2. Target Categories

| Category | Target |
|---|---:|
| NTA | 40 |
| YTA | 40 |
| ESH | 40 |
| NAH | 40 |
| Community disagreement | 20 |
| Insufficient information | 20 |

Each case belongs to exactly one final category.

## 3. Eligibility

For consensus cases:
- Require at least five valid moral-verdict comments.
- Require at least 70% agreement.
- Exclude INFO from the moral agreement denominator.
- Assign the reference verdict from the winning moral class.

For community disagreement:
- Require at least five valid moral-verdict comments.
- No moral verdict may reach 70% agreement.
- The reference verdict is null.

For insufficient information:
- Human reviewers must identify essential missing facts.
- Community consensus alone does not establish completeness.
- The reference verdict is null.
- The expected uncertainty tier is `withhold`.

Coordinators nominate by ID from the isolated, DATA-02-accepted pool and
privately record the specific missing fact. Reserve the case exclusively for
this review before sampling by consensus. A high INFO count is a signal to
inspect, never a substitute for human judgment. Record how the absent fact
could change the assessment. EVAL-01 cannot nominate a post rejected for too
few moral votes; the team must approve any expansion to that population.

## 4. Source Comment Convention

The shared `GoldenAnnotation` schema allows at most
10 source comments, including INFO.

Use the provider's original ranks 1–10, preserving gaps. Apply DATA-02 filters:
discard deleted/removed, unlabeled, unsupported, and empty comments. Count
valid INFO separately. Divide the largest moral count by the total moral count.
The coordinator reconciles all five counts against the pinned source row.
Independent reviewers never see these votes or community comments.

Do not silently change the schema or manufacture votes
to satisfy validation.

## 5. Independent Annotation

Each final case must be reviewed by two annotators.

The coordinator approves one version of the narrative for the batch and screens
it for identifying details, embedded verdicts, quotes of community comments,
and spoilers. Hold unsuitable narratives for team disposition. Automated text
checks only catch some obvious identifiers and verdict codes. Issue only the
blinded assignment ID and narrative. Keep IDs, mappings, proposals, vote
counts, comments, and agreement in restricted coordinator storage. Do not
search for the source post.

Assign each case to two identified reviewers in a stable reviewer order.
Each works privately without seeing the other's response. Lock both original
submissions before comparison, preserving assignment ID and reviewer identity.

For each case, reviewers should record:
- Their verdict: NTA, YTA, ESH, NAH, or UNDETERMINED.
- Relevant facts supported by the narrative.
- Essential missing information.
- Plausible alternative perspectives.
- Possible framing bias.
- Expected uncertainty tier and initial action. Use `withhold` when essential
  facts prevent a defensible judgment, `split` for competing supported verdicts,
  `leaning` for a tentative judgment, and `confident` for clear support.
  Choose `request_clarification` when a targeted question could supply an
  essential fact, otherwise `proceed`. Use `UNDETERMINED` for a personal
  verdict the evidence does not justify.
- Notes explaining the decision.

Do not infer unsupported motivations, intentions, or events.

Reviewers record these answers in the file from `export_review_template`
(fields `verdict`, `key_facts`, `missing_information`,
`alternative_perspectives`, `framing_bias`, `expected_uncertainty_tier`,
`expected_initial_action`, `notes`). Edit only your own file, and return it to
the coordinator once; it is locked before any comparison.

## 6. Completeness Review

Consider whether the narrative provides enough information
about:
- The narrator's actions.
- The other parties' actions and reactions.
- The relationship and relevant context.
- The triggering event.
- Relevant intent, where necessary.
- The outcome or consequences.

Distinguish essential missing facts from minor details.

A case should be flagged for missing-context review when
the absent information could materially change the
assessment.

## 7. Bias and Alternative Perspectives

Reviewers should distinguish:
- Directly stated facts.
- The narrator's interpretations.
- Unsupported assumptions.
- Plausible interpretations from other parties.

Framing bias should be recorded using the shared schema:
`none`, `possible`, or `clear`.

A one-sided narrative does not automatically imply
deliberate dishonesty.

## 8. Disagreement and Adjudication

Preserve both independent annotator verdicts. Before discussion, compute
percentage agreement and Cohen's kappa over paired original verdicts, including
UNDETERMINED. Record the batch, number of pairs, and any undefined kappa
(e.g. both reviewers always used one class). Keep original responses for audit.

When reviewers disagree:
1. Record both original judgments.
2. Identify the specific reason for disagreement.
3. Discuss evidence and missing information.
4. Record the adjudicator, case, original proposal, final category, and rationale.
   Missing context takes precedence even when community consensus is strong.
   Log reviewed nominations that are declined as well.
5. Reconcile the final record with the proposal and pinned source using
   `validate_completed_annotation`. Pass the explicit decision for any change.
   Count final cases once, check quotas, and document any shortfall or replacement
   drawn from the same verified pool.

Do not overwrite the independent judgments.

The team must agree on the adjudication owner and
procedure before final annotation begins.


## 9. Shared Annotation Schema

Use the existing `GoldenAnnotation` Pydantic model.

Required information includes:
- Case ID and narrative.
- Community votes and agreement.
- Reference verdict and case type.
- Key facts and missing information.
- Alternative perspectives and framing bias.
- Exactly two independent annotator verdicts.
- Annotation notes.
- Expected uncertainty tier.
- Expected initial action.

Do not create fake annotator verdicts or placeholder
human judgments to make records pass validation.

Review manifests generated by `golden_annotation_prep.py` are
metadata-only preparation artifacts. They must not be treated
as completed `GoldenAnnotation` records.

The final source-comment counts, including INFO, must be
verified before final schema validation.



## 10. Data Separation and Safety

Only use candidates from the DATA-04 held-out pool.

Verify that selected cases and their near-duplicates
do not appear in development or retrieval partitions.

Keep story content and completed annotations out of
Git unless an explicit, approved data-handling policy
allows otherwise.

Candidate IDs linked to provisional verdicts are sensitive evaluation metadata.
Keep mappings, source manifests, reviewer submissions and adjudication records
out of Git and away from training, retrieval, prompts, and development users.

## 11. Open Decisions for Team Review

- Who owns the private nomination register and replacement log?
- Should the eligible missing-context population ever extend beyond DATA-02?
- Which narrative representation and content screening protocol are approved?
- Who are the two independent annotators?
- Who performs adjudication?
- Where will completed annotations be stored securely?
- What review evidence is required before dataset freeze?

## 12. Completion Criteria

Before EVAL-01 is considered complete:
- Initial candidates exist for the required categories.
- Eligibility and available counts are documented.
- Split isolation is verified using DATA-04 outputs.
- Candidate categories are mutually exclusive.
- Annotation records can be prepared without fabricated data.
- These guidelines have been reviewed by the team.

EVAL-02 continues independent annotation in batches. Completing and freezing
the full 200-case set is a later milestone. EVAL-01 prepares the candidates,
workflow, and guidelines; a preliminary manifest does not close the issue.
