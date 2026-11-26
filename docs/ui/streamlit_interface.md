# UI-01: Streamlit interface with mock assessments

Prototype of the user interface (Milestone 2, Sections 1.1, 4.1-4.4 and 6). All results are synthetic
`AssessmentResult` examples; no classifier, retrieval pipeline, LLM or MCP server is called.

## Run

```bash
uv sync --group ui
uv run streamlit run apps/streamlit_app.py
```

Choose an example in the sidebar (**Mock assessment**), write any scenario and press **Assess**.
`uv run --group ui pytest` also runs the app smoke tests; without the `ui` group they are skipped, so CI
stays lightweight.

## Files

| File | Role |
| --- | --- |
| `apps/streamlit_app.py` | Streamlit layout and interaction flow. |
| `src/asshole_ai_detector/ui/mock_assessments.py` | Synthetic results, the `AssessmentBackend` protocol and `MockAssessmentBackend`. |
| `src/asshole_ai_detector/ui/presentation.py` | Verdict and tier labels, plus display rules (no Streamlit import, so it is unit-tested in CI). |

## Layout

1. **Scenario form**: multiline text area and an **Assess** button. Empty input shows a warning.
2. **Verdict**: one verdict, two side-by-side possibilities for `split` (`verdict` first, then
   `secondary_verdict`), or **Verdict withheld** when `verdict` is `null`. Each verdict has a text label;
   color is only a visual aid.
3. **Uncertainty tier**: `confident`, `leaning`, `split` or `withhold`, with a short description.
4. **Explanation**, **Alternative perspectives**, **Missing information** (highlighted) and **Limitations**
   (collapsed). Empty lists are not shown.
5. A note that tiers describe the strength of the evidence, not objective certainty (Section 4.2).

Every displayed field comes from `AssessmentResult`. Classifier probabilities, retrieved cases and bias-check
details are not part of that contract and are not shown.

## Clarification flow (Section 4.2)

- A result with non-empty `clarification_questions` is treated as a clarification request: the questions
  (at most three) are shown as inputs instead of a verdict.
- **Submit answers** sends the scenario and the answers back to the backend. **Continue without answering**
  sends empty answers.
- Only one round is allowed. After it, the next result is displayed as final.
- The mock returns the `leaning` example after answers and a withheld verdict when every answer is empty.

## Mock examples

| Example | Tier | Verdict(s) | Shows |
| --- | --- | --- | --- |
| `confident` | confident | YTA | explanation, perspectives, limitation |
| `leaning` | leaning | NTA | missing information, limitations |
| `split` | split | ESH / NAH | two possibilities |
| `withhold` | withhold | null | missing information, no verdict |
| `clarification` | withhold | null | three clarification questions |

## Assumptions for Week 9 integration

- The workflow controller implements `AssessmentBackend.assess(scenario, clarification_answers=None)` and
  returns a validated `AssessmentResult`. The UI only changes which backend `render_sidebar` creates.
- `clarification_answers` maps each question to the user's answer (empty string if skipped). How answers are
  merged into the scenario is the controller's decision.
- The controller enforces the single clarification round; the UI also stops asking after one round.
- Errors from the controller (timeouts, unavailable models) need a separate error state; the controller must not
  return a made-up verdict on failure (ARCH-02, Section 8). This is not designed yet.
- Submitted scenarios live only in the Streamlit session. They are not logged or saved, in line with the rule
  of never storing user stories.
