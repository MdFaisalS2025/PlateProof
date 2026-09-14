"""RED-first tests for the real Owner Copilot Streamlit page (Task 8B),
replacing the old placeholder. Uses streamlit.testing.v1.AppTest -- no
browser, no network, no Ollama required."""

from __future__ import annotations

from datetime import date
from typing import Any


def _write_fixtures(
    write_restaurants: Any,
    write_inspections: Any,
    write_violations: Any,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
    *,
    restaurant_id: str = "nyc:1",
    jurisdiction: str = "nyc",
) -> None:
    write_restaurants(
        [
            restaurant_row(
                restaurant_id=restaurant_id, jurisdiction=jurisdiction, name="Anna's Kitchen"
            )
        ]
    )
    write_inspections(
        [
            inspection_row(
                inspection_id=f"{restaurant_id}:1",
                restaurant_id=restaurant_id,
                jurisdiction=jurisdiction,
                inspection_date=date(2024, 1, 1),
            ),
            inspection_row(
                inspection_id=f"{restaurant_id}:2",
                restaurant_id=restaurant_id,
                jurisdiction=jurisdiction,
                inspection_date=date(2025, 6, 1),
            ),
        ]
    )
    write_violations(
        [
            violation_row(
                violation_event_id=f"{restaurant_id}:v:1",
                inspection_id=f"{restaurant_id}:1",
                restaurant_id=restaurant_id,
                jurisdiction=jurisdiction,
                inspection_date=date(2024, 1, 1),
                violation_code="04L",
                violation_code_norm="04L",
            ),
            violation_row(
                violation_event_id=f"{restaurant_id}:v:2",
                inspection_id=f"{restaurant_id}:2",
                restaurant_id=restaurant_id,
                jurisdiction=jurisdiction,
                inspection_date=date(2025, 6, 1),
                violation_code="04L",
                violation_code_norm="04L",
            ),
        ]
    )


def test_page_loads_without_ollama_configured(app_env: Any, app_path: Any) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(app_path("pages", "3_Owner_Copilot.py"))
    at.run(timeout=30)
    assert not at.exception
    assert any("PlateProof Copilot" in t.value for t in at.title)


def test_ownership_disclaimer_is_always_present(app_env: Any, app_path: Any) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(app_path("pages", "3_Owner_Copilot.py"))
    at.run(timeout=30)
    body_text = "\n".join(m.value for m in list(at.info) + list(at.markdown) + list(at.caption))
    assert "does not verify restaurant ownership" in body_text


def test_restaurant_lookup_by_id_shows_identity(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    write_inspections: Any,
    write_violations: Any,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_fixtures(
        write_restaurants,
        write_inspections,
        write_violations,
        restaurant_row,
        inspection_row,
        violation_row,
    )
    at = AppTest.from_file(app_path("pages", "3_Owner_Copilot.py"))
    at.run(timeout=30)
    at.text_input(key="copilot_restaurant_id").set_value("nyc:1").run(timeout=30)
    assert not at.exception
    body_text = "\n".join(m.value for m in list(at.markdown) + list(at.header) + list(at.subheader))
    assert "Anna's Kitchen" in body_text


def test_example_question_button_produces_a_grounded_answer(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    write_inspections: Any,
    write_violations: Any,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_fixtures(
        write_restaurants,
        write_inspections,
        write_violations,
        restaurant_row,
        inspection_row,
        violation_row,
    )
    at = AppTest.from_file(app_path("pages", "3_Owner_Copilot.py"))
    at.run(timeout=30)
    at.text_input(key="copilot_restaurant_id").set_value("nyc:1").run(timeout=30)
    recurring_buttons = [b for b in at.button if "recurring" in (b.label or "").lower()]
    assert recurring_buttons
    recurring_buttons[0].click().run(timeout=30)
    assert not at.exception
    body_text = "\n".join(m.value for m in at.markdown)
    assert "04L" in body_text


def test_free_text_question_produces_a_grounded_answer(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    write_inspections: Any,
    write_violations: Any,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_fixtures(
        write_restaurants,
        write_inspections,
        write_violations,
        restaurant_row,
        inspection_row,
        violation_row,
    )
    at = AppTest.from_file(app_path("pages", "3_Owner_Copilot.py"))
    at.run(timeout=30)
    at.text_input(key="copilot_restaurant_id").set_value("nyc:1").run(timeout=30)
    at.text_area(key="copilot_question").set_value("What are the recurring violations?").run(
        timeout=30
    )
    submit_buttons = [
        b
        for b in at.button
        if "ask" in (b.label or "").lower() or "submit" in (b.label or "").lower()
    ]
    assert submit_buttons
    submit_buttons[0].click().run(timeout=30)
    assert not at.exception
    body_text = "\n".join(m.value for m in at.markdown)
    assert "04L" in body_text


def test_refusal_display_for_prohibited_question(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    write_inspections: Any,
    write_violations: Any,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_fixtures(
        write_restaurants,
        write_inspections,
        write_violations,
        restaurant_row,
        inspection_row,
        violation_row,
    )
    at = AppTest.from_file(app_path("pages", "3_Owner_Copilot.py"))
    at.run(timeout=30)
    at.text_input(key="copilot_restaurant_id").set_value("nyc:1").run(timeout=30)
    at.text_area(key="copilot_question").set_value("Will this make someone get sick?").run(
        timeout=30
    )
    submit_buttons = [
        b
        for b in at.button
        if "ask" in (b.label or "").lower() or "submit" in (b.label or "").lower()
    ]
    submit_buttons[0].click().run(timeout=30)
    assert not at.exception
    assert list(at.warning)


def test_why_plateproof_says_this_expander_present(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    write_inspections: Any,
    write_violations: Any,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_fixtures(
        write_restaurants,
        write_inspections,
        write_violations,
        restaurant_row,
        inspection_row,
        violation_row,
    )
    at = AppTest.from_file(app_path("pages", "3_Owner_Copilot.py"))
    at.run(timeout=30)
    at.text_input(key="copilot_restaurant_id").set_value("nyc:1").run(timeout=30)
    recurring_buttons = [b for b in at.button if "recurring" in (b.label or "").lower()]
    recurring_buttons[0].click().run(timeout=30)
    expander_labels = [e.label for e in at.expander]
    assert any("Why PlateProof says this" in (label or "") for label in expander_labels)


def test_no_unsafe_html_in_page_source(app_path: Any) -> None:
    source = open(app_path("pages", "3_Owner_Copilot.py"), encoding="utf-8").read()
    assert "unsafe_allow_html" not in source


def test_forecast_disclaimer_language_present_in_source(app_path: Any) -> None:
    source = open(app_path("pages", "3_Owner_Copilot.py"), encoding="utf-8").read()
    assert "not a guarantee" in source or "statistical estimate" in source


def test_michelin_safety_disclaimer_language_present_in_source(app_path: Any) -> None:
    source = open(app_path("pages", "3_Owner_Copilot.py"), encoding="utf-8").read()
    assert "does not indicate food safety" in source or "not food safety" in source.lower()


def test_local_helper_unavailable_still_renders_a_deterministic_answer(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    write_inspections: Any,
    write_violations: Any,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
    monkeypatch: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_fixtures(
        write_restaurants,
        write_inspections,
        write_violations,
        restaurant_row,
        inspection_row,
        violation_row,
    )
    at = AppTest.from_file(app_path("pages", "3_Owner_Copilot.py"))
    at.run(timeout=30)
    at.text_input(key="copilot_restaurant_id").set_value("nyc:1").run(timeout=30)
    recurring_buttons = [b for b in at.button if "recurring" in (b.label or "").lower()]
    recurring_buttons[0].click().run(timeout=30)
    assert not at.exception


def test_local_ai_disabled_state_is_shown(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    write_inspections: Any,
    write_violations: Any,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_fixtures(
        write_restaurants,
        write_inspections,
        write_violations,
        restaurant_row,
        inspection_row,
        violation_row,
    )
    at = AppTest.from_file(app_path("pages", "3_Owner_Copilot.py"))
    at.run(timeout=30)
    body_text = "\n".join(m.value for m in list(at.caption) + list(at.markdown))
    assert "Local AI" in body_text or "local AI" in body_text
