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


# --------------------------------------------------------------------------- #
# Correction: cross-restaurant answer-state isolation, selectable search,
# sanitized UI errors, and graph/corpus/local-AI degradation.
# --------------------------------------------------------------------------- #


def _write_two_restaurants(
    write_restaurants: Any,
    write_inspections: Any,
    write_violations: Any,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
) -> None:
    from datetime import date

    write_restaurants(
        [
            restaurant_row(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                name="Anna's Kitchen",
                normalized_name="annas kitchen",
            ),
            restaurant_row(
                restaurant_id="nyc:2",
                jurisdiction="nyc",
                name="Bob's Diner",
                normalized_name="bobs diner",
            ),
        ]
    )
    write_inspections(
        [
            inspection_row(
                inspection_id="nyc:1:1",
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_date=date(2024, 1, 1),
            ),
            inspection_row(
                inspection_id="nyc:1:2",
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_date=date(2025, 6, 1),
            ),
            inspection_row(
                inspection_id="nyc:2:1",
                restaurant_id="nyc:2",
                jurisdiction="nyc",
                inspection_date=date(2024, 1, 1),
            ),
            inspection_row(
                inspection_id="nyc:2:2",
                restaurant_id="nyc:2",
                jurisdiction="nyc",
                inspection_date=date(2025, 6, 1),
            ),
        ]
    )
    write_violations(
        [
            violation_row(
                violation_event_id="nyc:1:v:1",
                inspection_id="nyc:1:1",
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_date=date(2024, 1, 1),
                violation_code="04L",
                violation_code_norm="04L",
            ),
            violation_row(
                violation_event_id="nyc:1:v:2",
                inspection_id="nyc:1:2",
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_date=date(2025, 6, 1),
                violation_code="04L",
                violation_code_norm="04L",
            ),
            violation_row(
                violation_event_id="nyc:2:v:1",
                inspection_id="nyc:2:1",
                restaurant_id="nyc:2",
                jurisdiction="nyc",
                inspection_date=date(2024, 1, 1),
                violation_code="08A",
                violation_code_norm="08A",
            ),
            violation_row(
                violation_event_id="nyc:2:v:2",
                inspection_id="nyc:2:2",
                restaurant_id="nyc:2",
                jurisdiction="nyc",
                inspection_date=date(2025, 6, 1),
                violation_code="08A",
                violation_code_norm="08A",
            ),
        ]
    )


def test_old_answer_is_cleared_before_a_new_question_when_restaurant_changes(
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

    _write_two_restaurants(
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
    body_text = "\n".join(m.value for m in at.markdown)
    assert "04L" in body_text

    # Switch to restaurant B -- the previous answer (about 04L) must
    # disappear before any new question is asked.
    at.text_input(key="copilot_restaurant_id").set_value("nyc:2").run(timeout=30)
    assert not at.exception
    body_text_after_switch = "\n".join(m.value for m in at.markdown)
    assert "04L" not in body_text_after_switch

    # Now ask restaurant B a question -- only B's answer (08A) must show.
    recurring_buttons_b = [b for b in at.button if "recurring" in (b.label or "").lower()]
    recurring_buttons_b[0].click().run(timeout=30)
    body_text_b = "\n".join(m.value for m in at.markdown)
    assert "08A" in body_text_b
    assert "04L" not in body_text_b


def test_old_answer_is_cleared_when_restaurant_id_becomes_empty(
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
    body_text = "\n".join(m.value for m in at.markdown)
    assert "04L" in body_text

    at.text_input(key="copilot_restaurant_id").set_value("").run(timeout=30)
    assert not at.exception
    body_text_after = "\n".join(m.value for m in at.markdown)
    assert "04L" not in body_text_after


def test_old_answer_is_cleared_when_restaurant_id_becomes_invalid(
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
    body_text = "\n".join(m.value for m in at.markdown)
    assert "04L" in body_text

    at.text_input(key="copilot_restaurant_id").set_value("nyc:does-not-exist").run(timeout=30)
    assert not at.exception
    body_text_after = "\n".join(m.value for m in at.markdown)
    assert "04L" not in body_text_after


def test_search_results_are_selectable_and_selection_loads_the_restaurant(
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
    at.expander[0].expanded = True
    at.run(timeout=30)
    at.text_input(key="copilot_search_query").set_value("Anna").run(timeout=30)
    search_buttons = [b for b in at.button if b.label == "Search"]
    assert search_buttons
    search_buttons[0].click().run(timeout=30)

    select_buttons = [b for b in at.button if "select" in (b.label or "").lower()]
    assert select_buttons
    assert any("Anna's Kitchen" in (b.label or "") for b in select_buttons)
    assert any("nyc:1" in (b.label or "") for b in select_buttons)

    matching = [b for b in select_buttons if "Anna's Kitchen" in (b.label or "")][0]
    matching.click().run(timeout=30)
    assert not at.exception
    body_text = "\n".join(m.value for m in list(at.markdown) + list(at.header))
    assert "Anna's Kitchen" in body_text


def test_duplicate_name_search_results_are_distinguishable(
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

    write_restaurants(
        [
            restaurant_row(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                name="Same Name Cafe",
                normalized_name="same name cafe",
                address="1 First Ave",
            ),
            restaurant_row(
                restaurant_id="florida:1",
                jurisdiction="florida",
                name="Same Name Cafe",
                normalized_name="same name cafe",
                address="2 Second St",
            ),
        ]
    )
    at = AppTest.from_file(app_path("pages", "3_Owner_Copilot.py"))
    at.run(timeout=30)
    at.expander[0].expanded = True
    at.run(timeout=30)
    at.text_input(key="copilot_search_query").set_value("Same Name Cafe").run(timeout=30)
    search_buttons = [b for b in at.button if b.label == "Search"]
    search_buttons[0].click().run(timeout=30)

    select_buttons = [b for b in at.button if "select" in (b.label or "").lower()]
    assert len(select_buttons) == 2
    labels = {b.label for b in select_buttons}
    assert any("nyc:1" in label for label in labels)
    assert any("florida:1" in label for label in labels)
    assert any("NYC" in label.upper() for label in labels)
    assert any("FLORIDA" in label.upper() for label in labels)


def test_search_selection_clears_previous_restaurants_answer(
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

    _write_two_restaurants(
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
    body_text = "\n".join(m.value for m in at.markdown)
    assert "04L" in body_text

    at.expander[0].expanded = True
    at.run(timeout=30)
    at.text_input(key="copilot_search_query").set_value("Bob").run(timeout=30)
    search_buttons = [b for b in at.button if b.label == "Search"]
    search_buttons[0].click().run(timeout=30)
    select_buttons = [b for b in at.button if "select" in (b.label or "").lower()]
    matching = [b for b in select_buttons if "Bob's Diner" in (b.label or "")][0]
    matching.click().run(timeout=30)

    body_text_after = "\n".join(m.value for m in at.markdown)
    assert "04L" not in body_text_after


def test_repository_search_failure_shows_no_exception_class_name(
    app_env: Any, app_path: Any, monkeypatch: Any
) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(app_path("pages", "3_Owner_Copilot.py"))
    at.run(timeout=30)
    at.expander[0].expanded = True
    at.run(timeout=30)
    at.text_input(key="copilot_search_query").set_value("anything").run(timeout=30)

    import theme

    def _boom() -> Any:
        raise RuntimeError("simulated repository failure")

    monkeypatch.setattr(theme, "repository", _boom)
    search_buttons = [b for b in at.button if b.label == "Search"]
    search_buttons[0].click().run(timeout=30)
    assert not at.exception
    error_text = "\n".join(e.value for e in at.error)
    assert "RuntimeError" not in error_text
    assert "Exception" not in error_text


def test_graph_build_failure_shows_a_fixed_unavailable_message(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    restaurant_row: Any,
    monkeypatch: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    write_restaurants([restaurant_row(restaurant_id="nyc:1", jurisdiction="nyc")])
    at = AppTest.from_file(app_path("pages", "3_Owner_Copilot.py"))
    at.run(timeout=30)
    at.text_input(key="copilot_restaurant_id").set_value("nyc:1").run(timeout=30)

    import theme

    from plateproof.graph.models import GraphScaleExceededError

    class _AlwaysFailsGraphService:
        def get(self) -> Any:
            raise GraphScaleExceededError("simulated failure")

    monkeypatch.setattr(theme, "graph_service", lambda: _AlwaysFailsGraphService())
    recurring_buttons = [b for b in at.button if "recurring" in (b.label or "").lower()]
    recurring_buttons[0].click().run(timeout=30)
    assert not at.exception
    error_text = "\n".join(e.value for e in at.error)
    assert "temporarily unavailable" in error_text.lower()
    assert "GraphScaleExceededError" not in error_text
    assert "Traceback" not in error_text


def test_fact_only_intent_works_without_a_corpus(
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
    assert not at.exception
    body_text = "\n".join(m.value for m in at.markdown)
    assert "04L" in body_text


def test_guidance_intent_reports_unavailable_without_a_corpus(
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
    guidance_buttons = [b for b in at.button if "official guidance" in (b.label or "").lower()]
    assert guidance_buttons
    guidance_buttons[0].click().run(timeout=30)
    assert not at.exception
    body_text = "\n".join(m.value for m in at.markdown)
    assert "does not currently have reviewed official guidance" in body_text
