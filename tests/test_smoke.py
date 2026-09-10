def test_import_plateproof() -> None:
    import plateproof

    assert plateproof.__version__ == "0.1.0"


def test_settings_default_to_google_disabled() -> None:
    from plateproof.core.config import Settings, get_settings

    settings = Settings(_env_file=None)

    assert settings.google_integration_enabled is False
    assert settings.local_llm_enabled is False
    assert get_settings().google_integration_enabled is False
