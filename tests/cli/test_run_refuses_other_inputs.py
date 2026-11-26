"""A run into a directory that holds another run's stages refuses to reuse them."""

import json

import pytest
from click.testing import CliRunner

from cli.commands import run as run_module

ARGS = ["-b", "-82.2,26.3,-81.7,26.8", "-t", "storm", "-s", "2022-09-15", "-e", "2022-10-12"]


def _earlier_run(workdir, start, end):
    """State left by a run that discovered and ingested, then stopped at analyze."""
    workdir.mkdir()
    (workdir / ".workflow_state.json").write_text(json.dumps({
        "completed_stages": ["discover", "ingest"],
        "stage_results": {},
        "config": {
            "area_path": None,
            "bbox": "-82.2,26.3,-81.7,26.8",
            "analyze_bbox": "-82.2,26.3,-81.7,26.8",
            "max_cloud": 30.0,
            "start_date": f"{start}T00:00:00",
            "end_date": f"{end}T00:00:00",
            "event_type": "storm",
            "algorithm": None,
            "data_mode": "real",
        },
    }))


@pytest.fixture
def analyze_calls(monkeypatch):
    calls = []

    def fake_analyze(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("analyze reached")

    monkeypatch.setattr(run_module, "run_analyze", fake_analyze)
    return calls


def test_other_dates_refused_before_any_stage(tmp_path, analyze_calls):
    workdir = tmp_path / "out"
    _earlier_run(workdir, "2022-09-29", "2022-10-02")

    result = CliRunner().invoke(run_module.run, ARGS + ["-o", str(workdir)])

    assert result.exit_code == 2
    assert "different inputs (end_date, start_date)" in result.output
    assert analyze_calls == []
    state = json.loads((workdir / ".workflow_state.json").read_text())
    assert state["config"]["start_date"] == "2022-09-29T00:00:00"


def test_same_inputs_resume_at_the_next_stage(tmp_path, analyze_calls):
    workdir = tmp_path / "out"
    _earlier_run(workdir, "2022-09-15", "2022-10-12")

    result = CliRunner().invoke(run_module.run, ARGS + ["-o", str(workdir)])

    assert "Discover: skipped" in result.output
    assert "Ingest: skipped" in result.output
    assert len(analyze_calls) == 1


DEFAULT_DATE_ARGS = ["-b", "-82.2,26.3,-81.7,26.8", "-t", "storm"]


def _yesterdays_run_with_default_dates(workdir):
    """State left by yesterday's run that gave no dates, stopped at analyze."""
    from datetime import datetime, timedelta

    end = datetime.now() - timedelta(days=1, minutes=3)
    _earlier_run(workdir, "x", "x")
    state = json.loads((workdir / ".workflow_state.json").read_text())
    state["config"]["end_date"] = end.isoformat()
    state["config"]["start_date"] = (end - timedelta(days=7)).isoformat()
    (workdir / ".workflow_state.json").write_text(json.dumps(state))
    return state["config"]


def test_default_end_date_resumes_the_next_day(tmp_path, analyze_calls):
    workdir = tmp_path / "out"
    stored = _yesterdays_run_with_default_dates(workdir)

    result = CliRunner().invoke(run_module.run, DEFAULT_DATE_ARGS + ["-o", str(workdir)])

    assert "different inputs" not in result.output
    assert "Ingest: skipped" in result.output
    assert len(analyze_calls) == 1
    state = json.loads((workdir / ".workflow_state.json").read_text())
    assert state["config"]["end_date"] == stored["end_date"]
    assert state["config"]["start_date"] == stored["start_date"]


def test_changed_explicit_end_date_still_refused_on_resume(tmp_path, analyze_calls):
    workdir = tmp_path / "out"
    _yesterdays_run_with_default_dates(workdir)

    result = CliRunner().invoke(
        run_module.run, DEFAULT_DATE_ARGS + ["-e", "2022-10-12", "-o", str(workdir)]
    )

    assert result.exit_code == 2
    assert "different inputs (end_date" in result.output
    assert analyze_calls == []


def test_fresh_directory_still_defaults_end_date_to_now(tmp_path, monkeypatch):
    from datetime import datetime

    seen = {}

    def fake_discover(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("discover reached")

    monkeypatch.setattr(run_module, "run_discover", fake_discover)

    CliRunner().invoke(run_module.run, DEFAULT_DATE_ARGS + ["-o", str(tmp_path / "out")])

    assert abs((datetime.now() - seen["end_date"]).total_seconds()) < 120
    assert (seen["end_date"] - seen["start_date"]).days == 7
