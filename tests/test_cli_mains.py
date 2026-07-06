"""main() argv coverage for validate_run_records + run_eval (R20).

Both modules expose a CLI main() that reads sys.argv. Drive the offline paths
(self-test / list / convert / validate, and the argument-guard early returns)
via monkeypatched argv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import validate_run_records as V  # noqa: E402
import run_eval as R  # noqa: E402


def _argv(monkeypatch, mod_name, argv):
    monkeypatch.setattr(sys, "argv", [mod_name, *argv])


# ---------------------------------------------------------------------------
# validate_run_records.main()
# ---------------------------------------------------------------------------
def test_validate_main_self_test(monkeypatch, capsys):
    _argv(monkeypatch, "validate_run_records.py", ["--self-test"])
    rc = V.main()
    assert rc == 0
    assert "validated" in capsys.readouterr().out


def test_validate_main_requires_input(monkeypatch):
    _argv(monkeypatch, "validate_run_records.py", [])
    with pytest.raises(SystemExit):
        V.main()


def test_validate_main_jsonl(monkeypatch, capsys, tmp_path):
    # build a valid record via the module's own fixtures, write it, validate it
    rec = V.self_test_records()[0]
    p = tmp_path / "rec.jsonl"
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    _argv(monkeypatch, "validate_run_records.py", ["--jsonl", str(p)])
    rc = V.main()
    assert rc == 0
    assert "ok:" in capsys.readouterr().out


def test_validate_main_results_convert_and_write(monkeypatch, capsys, tmp_path):
    results = [{
        "run_id": "r1",
        "task": {"id": "t1", "category": "C", "prompt": "p", "scoring_type": "json_exact"},
        "provider": {"id": "tested", "model": "m", "base_url": "u", "auth_env": "K"},
        "metrics": {"ok": True, "server_model": "m", "content_chars": 5},
    }]
    rp = tmp_path / "results.json"
    rp.write_text(json.dumps(results), encoding="utf-8")
    out = tmp_path / "converted.jsonl"
    _argv(monkeypatch, "validate_run_records.py",
          ["--results", str(rp), "--write-jsonl", str(out)])
    rc = V.main()
    assert rc == 0
    assert out.exists()


# ---------------------------------------------------------------------------
# run_eval.main()
# ---------------------------------------------------------------------------
def _tasks_file(tmp_path: Path) -> Path:
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps({"tasks": [
        {"id": "t1", "category": "Reasoning", "prompt": "hi", "scoring_type": "manual",
         "difficulty": "easy"},
        {"id": "t2", "category": "QA", "prompt": "yo", "scoring_type": "json_exact",
         "difficulty": "medium"},
    ]}), encoding="utf-8")
    return p


def test_run_eval_main_list_tasks(monkeypatch, capsys, tmp_path):
    _argv(monkeypatch, "run_eval.py", ["--tasks", str(_tasks_file(tmp_path)), "--list-tasks"])
    rc = R.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "t1" in out and "t2" in out


def test_run_eval_main_missing_providers(monkeypatch, tmp_path):
    # without --list-tasks and without --providers -> exit code 2
    _argv(monkeypatch, "run_eval.py", ["--tasks", str(_tasks_file(tmp_path))])
    rc = R.main()
    assert rc == 2


def test_run_eval_main_unknown_task_id(monkeypatch, tmp_path):
    _argv(monkeypatch, "run_eval.py",
          ["--tasks", str(_tasks_file(tmp_path)), "--task-id", "ghost", "--list-tasks"])
    rc = R.main()
    assert rc == 2  # unknown task id -> 2


def test_run_eval_main_self_test(monkeypatch, capsys):
    _argv(monkeypatch, "run_eval.py", ["--self-test"])
    rc = R.main()
    assert rc == 0
    assert "self-test ok" in capsys.readouterr().out
