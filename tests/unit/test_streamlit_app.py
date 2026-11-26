"""Smoke tests for the mock Streamlit app (skipped without the ``ui`` group)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[2] / "apps" / "streamlit_app.py"
SCENARIO = "I borrowed my friend's car without asking and returned it late."


def click(app: AppTest, label: str) -> AppTest:
    """Click a button (including form submit buttons) by its label."""
    button = next(b for b in app.button if b.label == label)
    return button.click().run()


def submit(key: str) -> AppTest:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.selectbox(key="mock_key").set_value(key).run()
    app.text_area[0].input(SCENARIO)
    click(app, "Assess")
    assert not app.exception
    return app


def page_text(app: AppTest) -> str:
    elements = [*app.markdown, *app.caption, *app.subheader]
    return "\n".join(element.value for element in elements)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("confident", ["You're the asshole", "Confident"]),
        ("leaning", ["Not the asshole", "Leaning", "Missing information"]),
        ("split", ["Everyone sucks here", "No assholes here", "Split"]),
        ("withhold", ["Verdict withheld", "Withheld"]),
    ],
)
def test_each_tier_renders(key, expected):
    text = page_text(submit(key))
    for fragment in expected:
        assert fragment in text


def test_empty_scenario_shows_warning():
    app = click(AppTest.from_file(str(APP_PATH)).run(), "Assess")
    assert app.warning and "describe the conflict" in app.warning[0].value


def test_clarification_round_then_result():
    app = submit("clarification")
    assert len(app.text_input) == 3

    app.text_input[0].input("They were upset but said nothing.")
    click(app, "Submit answers")

    assert not app.exception
    assert not app.text_input  # One round only: no further questions.
    assert "Not the asshole" in page_text(app)


def test_skipping_clarification_withholds_verdict():
    app = submit("clarification")
    click(app, "Continue without answering")

    assert not app.exception
    assert "Verdict withheld" in page_text(app)
