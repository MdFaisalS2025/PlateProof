"""Streamlit page smoke tests via AppTest. No browser, no network, no model
training or data download -- every page reads a tiny temporary datastore."""

from __future__ import annotations

from typing import Any


def test_home_page_smoke(app_env: Any, app_path: Any) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(app_path("Home.py"))
    at.run(timeout=30)
    assert not at.exception
    assert any("PlateProof" in t.value for t in at.title)


def test_search_page_smoke_with_no_restaurants(app_env: Any, app_path: Any) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(app_path("pages", "1_Restaurant_Search.py"))
    at.run(timeout=30)
    assert not at.exception


def test_search_page_shows_results(
    app_env: Any, app_path: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    from streamlit.testing.v1 import AppTest

    write_restaurants([restaurant_row(restaurant_id="nyc:1", name="Anna's Kitchen")])
    at = AppTest.from_file(app_path("pages", "1_Restaurant_Search.py"))
    at.run(timeout=30)
    assert not at.exception
    body_text = "\n".join(m.value for m in at.markdown)
    assert "Anna's Kitchen" in body_text


def test_search_page_omits_google_link_when_integration_disabled(
    app_env: Any, app_path: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    """Disabled is the default -- app_env() with no overrides."""
    from streamlit.testing.v1 import AppTest

    write_restaurants([restaurant_row(restaurant_id="nyc:1", name="Anna's Kitchen")])
    at = AppTest.from_file(app_path("pages", "1_Restaurant_Search.py"))
    at.run(timeout=30)
    assert not at.exception
    body_text = "\n".join(m.value for m in at.markdown)
    assert "View on Google Maps" not in body_text


def test_search_page_shows_google_link_when_integration_enabled(
    app_env: Any, app_path: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    from streamlit.testing.v1 import AppTest

    app_env(google_integration_enabled="true")
    write_restaurants([restaurant_row(restaurant_id="nyc:1", name="Anna's Kitchen")])
    at = AppTest.from_file(app_path("pages", "1_Restaurant_Search.py"))
    at.run(timeout=30)
    assert not at.exception
    body_text = "\n".join(m.value for m in at.markdown)
    assert "View on Google Maps" in body_text
    assert "google.com/maps/search" in body_text
    caption_text = "\n".join(c.value for c in at.caption)
    lowered = caption_text.lower()
    assert "confirm" in lowered
    assert "not affiliated" in lowered or "independent" in lowered


def test_inspection_history_page_smoke_without_restaurant_id(app_env: Any, app_path: Any) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(app_path("pages", "2_Inspection_History.py"))
    at.run(timeout=30)
    assert not at.exception
    assert any("Enter a restaurant ID" in i.value for i in at.info)


def test_inspection_history_page_shows_restaurant_not_found(app_env: Any, app_path: Any) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(app_path("pages", "2_Inspection_History.py"))
    at.run(timeout=30)
    at.text_input[0].set_value("nyc:does-not-exist").run(timeout=30)
    assert not at.exception
    assert any("No restaurant found" in e.value for e in at.error)


def test_inspection_history_page_shows_nyc_native_fields(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    write_inspections: Any,
    restaurant_row: Any,
    inspection_row: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    write_restaurants([restaurant_row(restaurant_id="nyc:1")])
    write_inspections([inspection_row()])
    at = AppTest.from_file(app_path("pages", "2_Inspection_History.py"))
    at.run(timeout=30)
    at.text_input[0].set_value("nyc:1").run(timeout=30)
    assert not at.exception
    body_text = "\n".join(m.value for m in at.markdown)
    assert "Score" in body_text
    assert "Grade" in body_text


def test_inspection_history_page_shows_a_correction_contact_path(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    write_inspections: Any,
    restaurant_row: Any,
    inspection_row: Any,
) -> None:
    """Spec's Privacy-and-safety section requires "a correction/contact
    path" alongside the official-source link -- PlateProof doesn't operate
    its own record-correction intake, so this points to the same official
    agency link already shown, rather than inventing unsupported
    infrastructure."""
    from streamlit.testing.v1 import AppTest

    write_restaurants([restaurant_row(restaurant_id="nyc:1")])
    write_inspections([inspection_row()])
    at = AppTest.from_file(app_path("pages", "2_Inspection_History.py"))
    at.run(timeout=30)
    at.text_input[0].set_value("nyc:1").run(timeout=30)
    assert not at.exception
    body_text = "\n".join(m.value for m in at.markdown)
    caption_text = "\n".join(c.value for c in at.caption)
    lowered = caption_text.lower()
    assert "correct" in lowered or "dispute" in lowered
    assert "data.cityofnewyork.us" in body_text


def test_model_card_page_smoke_no_model_configured(app_env: Any, app_path: Any) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(app_path("pages", "4_Model_Card.py"))
    at.run(timeout=30)
    assert not at.exception
    assert any("No ready PlateProof model" in w.value for w in at.warning)


# The Owner Copilot page's real Task 8B behavior (restaurant lookup,
# example/free-text questions, grounded/refused answers, disclaimers,
# citations, local-AI status) is covered by
# tests/app/test_owner_copilot_page.py.
