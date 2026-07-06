"""Integration tests for eval_cli read-only / dry-run subcommands.

eval_cli.py is the 1500-line CLI surface. Its pure helpers are covered in
test_eval_cli_helpers.py; here we drive the actual subcommand handlers
end-to-end with on-disk fixtures and argparse.Namespace args — no live network.
These exercise the config-loading, override, and report-rendering paths that
helper tests can't reach.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval_cli as E  # noqa: E402


def _providers_file(tmp_path: Path) -> Path:
    data = {
        "tested_model": {
            "provider_id": "tested", "base_url": "https://gw.x/v1", "model": "claude-opus-4-6",
            "api_key_env": "TESTED_KEY", "protocol": "anthropic_messages", "auth_type": "x-api-key",
        },
        "judge_model": {
            "provider_id": "judge", "base_url": "https://gw.x/v1", "model": "gpt-5.5",
            "api_key_env": "JUDGE_KEY", "protocol": "openai_chat", "auth_type": "bearer",
        },
    }
    p = tmp_path / "providers.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _ns(**kw) -> argparse.Namespace:
    # defaults common to many handlers; override per test
    base = dict(providers=None, provider=None, live=False, campaign_id=None,
                campaigns_dir=None, runs_dir=None)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _no_secrets(monkeypatch):
    # ensure no real local_secrets.env interferes; keys absent is fine for dry/config paths
    monkeypatch.setattr(E, "load_local_env", lambda *a, **k: {}, raising=False)


# ---------------------------------------------------------------------------
# load_two_model_config + apply_model_overrides via real files
# ---------------------------------------------------------------------------
def test_load_two_model_config_reads_roles(tmp_path):
    models = E.load_two_model_config(_providers_file(tmp_path))
    assert set(models) == {"tested_model", "judge_model"}
    assert models["tested_model"].model == "claude-opus-4-6"
    assert models["judge_model"].protocol == "openai_chat"


def test_load_two_model_config_extra_role(tmp_path):
    data = json.loads(_providers_file(tmp_path).read_text(encoding="utf-8"))
    data["suspect_model"] = {
        "provider_id": "suspect", "base_url": "https://other/v1", "model": "claude-opus-4-6",
        "api_key_env": "SUSPECT_KEY", "protocol": "anthropic_messages", "auth_type": "x-api-key",
    }
    p = tmp_path / "p2.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    models = E.load_two_model_config(p)
    assert "suspect_model" in models


def test_sanitized_models_hides_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("TESTED_KEY", raising=False)
    models = E.load_two_model_config(_providers_file(tmp_path))
    out = E.sanitized_models(models)
    assert out["tested_model"]["api_key_env"] == "TESTED_KEY"
    assert out["tested_model"]["api_key_present"] is False
    # no raw secret value anywhere
    assert "api_key" not in out["tested_model"] or out["tested_model"].get("api_key") != "secret"


# placeholder-r7


# ---------------------------------------------------------------------------
# fingerprint subcommand (config-only, no network)
# ---------------------------------------------------------------------------
def test_fingerprint_command_prints_doc(tmp_path, capsys):
    args = _ns(providers=str(_providers_file(tmp_path)), provider="tested_model", live=False)
    rc = E.fingerprint(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["provider_id"] == "tested"
    assert out["schema_version"]  # protocol_fingerprint schema present
    # config-only fingerprint, not live -> REVIEW decision
    assert out.get("decision") == "REVIEW"


def test_fingerprint_command_rejects_bad_role(tmp_path):
    args = _ns(providers=str(_providers_file(tmp_path)), provider="nonexistent_role")
    with pytest.raises(ValueError, match="must be tested_model or judge_model"):
        E.fingerprint(args)


def test_fingerprint_judge_role(tmp_path, capsys):
    args = _ns(providers=str(_providers_file(tmp_path)), provider="judge_model")
    assert E.fingerprint(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["provider_id"] == "judge"


# ---------------------------------------------------------------------------
# baseline_inspect / baseline_versions (read-only over baseline dirs)
# ---------------------------------------------------------------------------
def _make_baseline(tmp_path: Path, baseline_id="OFFICIAL-X") -> Path:
    bdir = tmp_path / "baselines" / baseline_id
    bdir.mkdir(parents=True)
    (bdir / "baseline.json").write_text(json.dumps({
        "baseline_id": baseline_id,
        "schema_version": "claude_baseline_v1",
        "sample_count": 5,
        "protocol": {"stop_reason_enum_rate": 1.0},
    }), encoding="utf-8")
    return bdir


def test_baseline_inspect_command(tmp_path, capsys):
    _make_baseline(tmp_path)
    args = _ns(baseline_id="OFFICIAL-X", baselines_dir=str(tmp_path / "baselines"))
    rc = E.baseline_inspect(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["baseline_id"] == "OFFICIAL-X"
    assert out["sample_count"] == 5


def test_baseline_inspect_missing_raises(tmp_path):
    args = _ns(baseline_id="GHOST", baselines_dir=str(tmp_path / "baselines"))
    with pytest.raises(ValueError, match="baseline not found"):
        E.baseline_inspect(args)


def test_baseline_versions_legacy_hint(tmp_path, capsys):
    # a single-file baseline with no versions/ dir -> legacy hint, rc 0
    _make_baseline(tmp_path)
    args = _ns(baseline_id="OFFICIAL-X", baselines_dir=str(tmp_path / "baselines"))
    rc = E.baseline_versions(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["baseline_id"] == "OFFICIAL-X"
    assert "legacy" in out.get("note", "").lower()


# placeholder-r7b


# ---------------------------------------------------------------------------
# dry-run probe subcommands (no network: print reference + return 0)
# ---------------------------------------------------------------------------
def test_error_envelope_dry_run(tmp_path, capsys):
    args = _ns(providers=str(_providers_file(tmp_path)), provider="tested_model", live=False)
    rc = E.error_envelope(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["probe"] == "error_envelope"
    assert out["evidence_status"] == "dry_run_reference_only"
    assert "missing_max_tokens" in out["variants"]


def test_error_envelope_rejects_bad_role(tmp_path):
    args = _ns(providers=str(_providers_file(tmp_path)), provider="ghost", live=False)
    with pytest.raises(ValueError, match="must be tested_model or judge_model"):
        E.error_envelope(args)


def test_sse_fingerprint_dry_run(tmp_path, capsys):
    args = _ns(providers=str(_providers_file(tmp_path)), provider="tested_model", live=False)
    rc = E.sse_fingerprint(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["probe"] == "sse_event_order"
    assert out["evidence_status"] == "dry_run_reference_only"


def test_needle_dry_run(tmp_path, capsys):
    args = _ns(providers=str(_providers_file(tmp_path)), provider="tested_model", live=False,
               target_tokens=2000, seed=3, baseline_id=None,
               baselines_dir=str(tmp_path / "baselines"))
    rc = E.needle(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["probe"] == "needle_recall"
    assert out["evidence_status"] == "dry_run_reference_only"


