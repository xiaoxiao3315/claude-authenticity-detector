"""Tests for local_env.py — .env parsing + env-name override resolution.

local_env loads KEY=VALUE secrets from local_secrets.env into the process
without clobbering existing shell env (unless override=True). It's the gate
between on-disk secret names and the live key lookup, and was at 20%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import local_env as LE  # noqa: E402


# ---------------------------------------------------------------------------
# _parse_env_line
# ---------------------------------------------------------------------------
def test_parse_basic():
    assert LE._parse_env_line("FOO=bar") == ("FOO", "bar")


def test_parse_export_prefix():
    assert LE._parse_env_line("export FOO=bar") == ("FOO", "bar")


def test_parse_strips_matching_quotes():
    assert LE._parse_env_line('FOO="bar baz"') == ("FOO", "bar baz")
    assert LE._parse_env_line("FOO='bar'") == ("FOO", "bar")


def test_parse_skips_comments_and_blanks():
    assert LE._parse_env_line("# comment") is None
    assert LE._parse_env_line("   ") is None
    assert LE._parse_env_line("") is None


def test_parse_rejects_no_equals():
    assert LE._parse_env_line("NOTANENVLINE") is None


def test_parse_rejects_bad_key():
    assert LE._parse_env_line("1BAD=x") is None
    assert LE._parse_env_line("has-dash=x") is None


def test_parse_value_with_equals_sign():
    # only the first = splits; the rest is value
    assert LE._parse_env_line("URL=https://x/y?a=b") == ("URL", "https://x/y?a=b")


# ---------------------------------------------------------------------------
# load_local_env
# ---------------------------------------------------------------------------
def test_load_missing_file_returns_empty(tmp_path):
    assert LE.load_local_env(tmp_path / "nope.env") == {}


def test_load_sets_new_keys(tmp_path, monkeypatch):
    env = tmp_path / "s.env"
    env.write_text("FOO_KEY=secret1\nexport BAR_KEY='secret2'\n", encoding="utf-8")
    monkeypatch.delenv("FOO_KEY", raising=False)
    monkeypatch.delenv("BAR_KEY", raising=False)
    loaded = LE.load_local_env(env)
    assert loaded == {"FOO_KEY": "secret1", "BAR_KEY": "secret2"}
    import os
    assert os.environ["FOO_KEY"] == "secret1"
    assert os.environ["BAR_KEY"] == "secret2"


def test_load_does_not_override_existing_by_default(tmp_path, monkeypatch):
    env = tmp_path / "s.env"
    env.write_text("FOO_KEY=fromfile\n", encoding="utf-8")
    monkeypatch.setenv("FOO_KEY", "fromshell")
    loaded = LE.load_local_env(env)
    assert "FOO_KEY" not in loaded            # skipped, already set
    import os
    assert os.environ["FOO_KEY"] == "fromshell"


def test_load_override_true_replaces(tmp_path, monkeypatch):
    env = tmp_path / "s.env"
    env.write_text("FOO_KEY=fromfile\n", encoding="utf-8")
    monkeypatch.setenv("FOO_KEY", "fromshell")
    loaded = LE.load_local_env(env, override=True)
    assert loaded["FOO_KEY"] == "fromfile"
    import os
    assert os.environ["FOO_KEY"] == "fromfile"


def test_load_skips_empty_values(tmp_path, monkeypatch):
    env = tmp_path / "s.env"
    env.write_text("EMPTY_KEY=\nGOOD_KEY=val\n", encoding="utf-8")
    monkeypatch.delenv("EMPTY_KEY", raising=False)
    monkeypatch.delenv("GOOD_KEY", raising=False)
    loaded = LE.load_local_env(env)
    assert "EMPTY_KEY" not in loaded
    assert loaded["GOOD_KEY"] == "val"


# ---------------------------------------------------------------------------
# env_override
# ---------------------------------------------------------------------------
def test_env_override_uses_env_when_present(monkeypatch):
    monkeypatch.setenv("MODEL_ENV_VAR", "from-env")
    raw = {"model": "default-model", "model_env": "MODEL_ENV_VAR"}
    assert LE.env_override(raw, "model") == "from-env"


def test_env_override_falls_back_to_raw(monkeypatch):
    monkeypatch.delenv("MODEL_ENV_VAR", raising=False)
    raw = {"model": "default-model", "model_env": "MODEL_ENV_VAR"}
    assert LE.env_override(raw, "model") == "default-model"


def test_env_override_no_env_name():
    raw = {"model": "default-model"}
    assert LE.env_override(raw, "model") == "default-model"
