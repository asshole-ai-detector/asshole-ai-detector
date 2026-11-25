"""Streamlit prototype with mock assessments (UI-01, Milestone 2 Sections 1.1, 4.1-4.4, 6).

Run with: uv sync --group ui && uv run streamlit run apps/streamlit_app.py

Every result comes from ``MockAssessmentBackend``; no classifier, retrieval,
LLM or MCP server is called. The Week 9 controller replaces the backend
without changing how results are displayed.
"""

import streamlit as st

from asshole_ai_detector.schemas.result import AssessmentResult
from asshole_ai_detector.ui.mock_assessments import (
    MOCK_ASSESSMENTS,
    AssessmentBackend,
    MockAssessmentBackend,
)
from asshole_ai_detector.ui.presentation import (
    EVIDENCE_NOTE,
    TIER_STYLES,
    VerdictStyle,
    displayed_verdicts,
    needs_clarification,
)

MOCK_LABELS = {
    "confident": "Confident (YTA)",
    "leaning": "Leaning (NTA)",
    "split": "Split (ESH / NAH)",
    "withhold": "Withheld (missing information)",
    "clarification": "Clarification questions",
}

# Session keys. State lives only in this browser session; nothing is saved.
SCENARIO = "scenario"
RESULT = "result"
ROUND_USED = "clarification_round_used"


def reset_session() -> None:
    for key in (SCENARIO, RESULT, ROUND_USED):
        st.session_state.pop(key, None)


def render_sidebar() -> AssessmentBackend:
    with st.sidebar:
        st.header("Mock mode")
        st.caption(
            "Results are synthetic examples. No AI model is called, so the "
            "assessment does not depend on the text you submit."
        )
        key = st.selectbox(
            "Mock assessment",
            options=list(MOCK_ASSESSMENTS),
            format_func=MOCK_LABELS.__getitem__,
            key="mock_key",
        )
        st.button("Start over", on_click=reset_session)
    return MockAssessmentBackend(key)


def render_scenario_form(backend: AssessmentBackend) -> None:
    with st.form("scenario_form"):
        scenario = st.text_area(
            "Describe the conflict",
            height=220,
            placeholder=(
                "What happened, who was involved, what each person did, "
                "and where things stand now."
            ),
        )
        submitted = st.form_submit_button("Assess", type="primary")

    if submitted:
        if not scenario.strip():
            st.warning("Please describe the conflict before submitting.")
            return
        st.session_state[SCENARIO] = scenario
        st.session_state[RESULT] = backend.assess(scenario)
        st.session_state[ROUND_USED] = False


def render_verdict(style: VerdictStyle, *, caption: str | None = None) -> None:
    if caption:
        st.caption(caption)
    st.markdown(f"### :{style.color}[{style.label}]")
    st.write(style.description)


def render_verdicts(result: AssessmentResult) -> None:
    verdicts = displayed_verdicts(result)
    if len(verdicts) == 2:
        # Split tier: show both possibilities side by side, primary first.
        left, right = st.columns(2)
        with left:
            render_verdict(verdicts[0], caption="Leading possibility")
        with right:
            render_verdict(verdicts[1], caption="Second possibility")
    else:
        render_verdict(verdicts[0])

    tier = TIER_STYLES[result.uncertainty_tier]
    st.markdown(f"**Uncertainty tier:** {tier.title}")
    st.caption(tier.description)


def render_details(result: AssessmentResult) -> None:
    st.subheader("Explanation")
    st.write(result.explanation)

    if result.alternative_perspectives:
        st.subheader("Alternative perspectives")
        for perspective in result.alternative_perspectives:
            st.markdown(f"- {perspective}")

    if result.missing_information:
        st.subheader("Missing information")
        st.warning("\n".join(f"- {item}" for item in result.missing_information))

    if result.limitations:
        with st.expander("Limitations"):
            for limitation in result.limitations:
                st.markdown(f"- {limitation}")


def render_clarification(backend: AssessmentBackend, result: AssessmentResult) -> None:
    """One clarification round with up to three questions (Section 4.2)."""
    st.subheader("A few questions first")
    with st.form("clarification_form"):
        answers = {
            question: st.text_input(question, key=f"answer_{index}")
            for index, question in enumerate(result.clarification_questions)
        }
        submitted = st.form_submit_button("Submit answers", type="primary")
        skipped = st.form_submit_button("Continue without answering")

    if submitted or skipped:
        provided = answers if submitted else {question: "" for question in answers}
        st.session_state[RESULT] = backend.assess(st.session_state[SCENARIO], provided)
        st.session_state[ROUND_USED] = True
        # Rerun so the new result replaces the one already drawn above.
        st.rerun()


def render_result(backend: AssessmentBackend) -> None:
    result: AssessmentResult | None = st.session_state.get(RESULT)
    if result is None:
        return

    st.divider()
    round_used = st.session_state.get(ROUND_USED, False)
    if needs_clarification(result, round_used=round_used):
        st.info(result.explanation)
        render_clarification(backend, result)
        return

    render_verdicts(result)
    render_details(result)
    st.caption(EVIDENCE_NOTE)


def main() -> None:
    st.set_page_config(page_title="Asshole AI Detector", layout="centered")
    st.title("Asshole AI Detector")
    st.write(
        "Describe an interpersonal conflict to get a balanced assessment with "
        "alternative perspectives and an honest uncertainty level."
    )

    backend = render_sidebar()
    render_scenario_form(backend)
    render_result(backend)


main()
