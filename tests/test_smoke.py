from pathlib import Path


def test_import_plateproof() -> None:
    import plateproof

    assert plateproof.__version__ == "0.1.0"


def test_settings_default_to_google_disabled(isolated_settings_environment: None) -> None:
    from plateproof.core.config import Settings, get_settings

    settings = Settings(_env_file=None)

    assert settings.google_integration_enabled is False
    assert settings.local_llm_enabled is False
    assert get_settings().google_integration_enabled is False


def test_settings_task7_defaults_are_safe_and_local(isolated_settings_environment: None) -> None:
    from plateproof.core.config import Settings

    settings = Settings(_env_file=None)

    assert settings.nyc_model_artifact_path is None
    assert settings.florida_model_artifact_path is None
    assert settings.prediction_table_path is None
    assert settings.expose_non_ready_model_cards is False
    assert settings.max_page_size == 50
    assert settings.prediction_staleness_days == 90
    assert settings.google_integration_enabled is False


def test_settings_resolve_path_uses_app_base_dir_not_cwd(
    isolated_settings_environment: None,
) -> None:
    from plateproof.core.config import Settings

    settings = Settings(_env_file=None)
    resolved = settings.resolve_path(Path("data/processed"))

    assert resolved.is_absolute()
    # resolved against the application base dir, not the (chdir'd) test cwd
    assert resolved != (Path.cwd() / "data/processed").resolve()


def test_settings_resolve_path_leaves_absolute_paths_untouched(
    isolated_settings_environment: None,
) -> None:
    from plateproof.core.config import Settings

    settings = Settings(_env_file=None)
    absolute = (Path.cwd() / "somewhere").resolve()

    assert settings.resolve_path(absolute) == absolute
