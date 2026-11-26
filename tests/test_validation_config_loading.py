"""Where the validation config comes from, and what a broken one does."""

import logging
from pathlib import Path

import pytest

from core.data.ingestion.validation import config as config_module
from core.data.ingestion.validation.config import (
    SHIPPED_CONFIG_PATH,
    ValidationConfigError,
    load_config,
)

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config" / "ingestion.yaml"


@pytest.fixture
def no_user_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def test_shipped_config_path_is_the_repo_file():
    assert SHIPPED_CONFIG_PATH == REPO_CONFIG
    assert SHIPPED_CONFIG_PATH.exists()


def test_run_outside_repo_root_loads_shipped_file(tmp_path, monkeypatch, caplog, no_user_config):
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO, logger=config_module.__name__)

    config = load_config(use_environment=False)

    assert f"Validation config loaded from {REPO_CONFIG}" in caplog.text
    assert config.optical.std_dev_min_reflectance == 0.0001


def test_shipped_value_reaches_config_from_another_dir(tmp_path, monkeypatch, no_user_config):
    edited = tmp_path / "ingestion.yaml"
    edited.write_text(REPO_CONFIG.read_text().replace("std_dev_min: 1.0", "std_dev_min: 7.5"))
    monkeypatch.setattr(config_module, "SHIPPED_CONFIG_PATH", edited)
    monkeypatch.chdir(tmp_path)

    assert load_config(use_environment=False).optical.std_dev_min == 7.5


def test_config_with_syntax_error_stops_and_names_file(tmp_path, monkeypatch, no_user_config):
    broken = tmp_path / "ingestion.yaml"
    broken.write_text(REPO_CONFIG.read_text().replace("enabled: true", "enabled: [true"))
    monkeypatch.setattr(config_module, "SHIPPED_CONFIG_PATH", broken)

    with pytest.raises(ValidationConfigError, match=str(broken)):
        load_config(use_environment=False)


def test_explicit_missing_config_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_config(str(tmp_path / "nope.yaml"), use_environment=False)


def test_ingest_validation_does_not_swallow_config_error(tmp_path, monkeypatch, no_user_config):
    from cli.commands import ingest

    broken = tmp_path / "ingestion.yaml"
    broken.write_text("validation: [\n")
    monkeypatch.setattr(config_module, "SHIPPED_CONFIG_PATH", broken)

    with pytest.raises(ValidationConfigError, match=str(broken)):
        ingest._validate_downloaded_image(tmp_path / "scene.tif", {"id": "x"})


def test_broken_config_stops_run_ingest_before_any_download(tmp_path, monkeypatch, no_user_config):
    import json

    from cli.commands import ingest, run

    broken = tmp_path / "ingestion.yaml"
    broken.write_text("validation: [\n")
    monkeypatch.setattr(config_module, "SHIPPED_CONFIG_PATH", broken)
    downloads = []
    monkeypatch.setattr(ingest, "process_item", lambda **kw: downloads.append(kw) or True)
    discovery = tmp_path / "discovery.json"
    discovery.write_text(json.dumps({"results": [{"id": "a"}, {"id": "b"}]}))

    with pytest.raises(ValidationConfigError, match=str(broken)):
        run.run_ingest(discovery, tmp_path / "data", {}, data_mode=run.DATA_MODE_REAL)
    assert downloads == []


def test_broken_config_stops_ingest_command_before_any_download(tmp_path, monkeypatch, no_user_config):
    import json

    from click.testing import CliRunner

    from cli.commands import ingest

    broken = tmp_path / "ingestion.yaml"
    broken.write_text("validation: [\n")
    monkeypatch.setattr(config_module, "SHIPPED_CONFIG_PATH", broken)
    downloads = []
    monkeypatch.setattr(ingest, "process_item", lambda **kw: downloads.append(kw) or True)
    discovery = tmp_path / "discovery.json"
    discovery.write_text(json.dumps({"results": [{"id": "a", "source": "sentinel2"}]}))

    result = CliRunner().invoke(
        ingest.ingest, ["--input", str(discovery), "--output", str(tmp_path / "data")]
    )

    assert result.exit_code == 1
    assert str(broken) in result.output
    assert downloads == []


def _config_with_std_dev_min(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(REPO_CONFIG.read_text().replace("std_dev_min: 1.0", f"std_dev_min: {value}"))
    return path


def test_override_order_user_then_system_then_shipped(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    user = _config_with_std_dev_min(home / ".firstlight/config/ingestion.yaml", 3.0)
    system = _config_with_std_dev_min(tmp_path / "etc/ingestion.yaml", 2.0)
    shipped = _config_with_std_dev_min(tmp_path / "shipped/ingestion.yaml", 1.5)
    monkeypatch.setattr(config_module, "SYSTEM_CONFIG_PATH", system)
    monkeypatch.setattr(config_module, "SHIPPED_CONFIG_PATH", shipped)

    assert load_config(use_environment=False).optical.std_dev_min == 3.0
    user.unlink()
    assert load_config(use_environment=False).optical.std_dev_min == 2.0
    system.unlink()
    assert load_config(use_environment=False).optical.std_dev_min == 1.5


def test_explicit_path_beats_every_file(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    _config_with_std_dev_min(home / ".firstlight/config/ingestion.yaml", 3.0)
    explicit = _config_with_std_dev_min(tmp_path / "mine.yaml", 9.0)

    assert load_config(str(explicit), use_environment=False).optical.std_dev_min == 9.0


def test_user_file_with_one_key_keeps_every_other_shipped_value(tmp_path, monkeypatch, caplog):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    user = home / ".firstlight/config/ingestion.yaml"
    user.parent.mkdir(parents=True)
    user.write_text("validation:\n  optical:\n    thresholds:\n      std_dev_min: 3.0\n")
    shipped = tmp_path / "shipped/ingestion.yaml"
    shipped.parent.mkdir()
    shipped.write_text(
        REPO_CONFIG.read_text()
        .replace("non_zero_ratio_min: 0.05", "non_zero_ratio_min: 0.2")
        .replace("std_dev_min_db: 2.0", "std_dev_min_db: 4.5")
    )
    monkeypatch.setattr(config_module, "SYSTEM_CONFIG_PATH", tmp_path / "etc/none.yaml")
    monkeypatch.setattr(config_module, "SHIPPED_CONFIG_PATH", shipped)
    caplog.set_level(logging.INFO, logger=config_module.__name__)

    config = load_config(use_environment=False)

    assert config.optical.std_dev_min == 3.0
    assert config.optical.non_zero_ratio_min == 0.2
    assert config.sar.std_dev_min_db == 4.5
    assert config.optional_optical_bands == ["swir1", "swir2", "coastal", "red_edge"]
    assert f"Validation config loaded from {user} over {shipped}" in caplog.text


def test_user_file_falls_through_to_system_before_shipped(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    user = home / ".firstlight/config/ingestion.yaml"
    user.parent.mkdir(parents=True)
    user.write_text("validation:\n  optical:\n    thresholds:\n      std_dev_min: 3.0\n")
    system = tmp_path / "etc/ingestion.yaml"
    system.parent.mkdir()
    system.write_text("validation:\n  optical:\n    thresholds:\n      nodata_ratio_max: 0.5\n")
    monkeypatch.setattr(config_module, "SYSTEM_CONFIG_PATH", system)
    monkeypatch.setattr(config_module, "SHIPPED_CONFIG_PATH", REPO_CONFIG)

    config = load_config(use_environment=False)

    assert config.optical.std_dev_min == 3.0
    assert config.optical.nodata_ratio_max == 0.5
    assert config.optical.std_dev_min_reflectance == 0.0001


def test_broken_user_file_stops_even_with_good_shipped(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    user = home / ".firstlight/config/ingestion.yaml"
    user.parent.mkdir(parents=True)
    user.write_text("validation:\n  optical: [\n")

    with pytest.raises(ValidationConfigError, match=str(user)):
        load_config(use_environment=False)
