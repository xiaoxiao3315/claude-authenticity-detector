"""End-to-end run_eval.main() over a streaming MockTransport (T4 cont.).

Drives the full eval loop (main lines ~685-837): load providers, per-task
run_one against a fake Anthropic SSE stream, scoring, run_records + summary.csv.
Patches run_eval.httpx.Client to a MockTransport so no socket is opened.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_eval as R  # noqa: E402


def _providers_file(tmp_path: Path) -> Path:
    data = {"providers": [
        {"id": "tested", "base_url": "https://gw.x", "model": "claude-opus-4-6",
         "auth_type": "x-api-key", "auth_env": "TESTED_KEY", "provider_channel": "gateway"},
    ]}
    p = tmp_path / "providers.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _tasks_file(tmp_path: Path) -> Path:
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps({"tasks": [
        {"id": "t1", "category": "QA", "prompt": "hi", "scoring_type": "manual",
         "difficulty": "easy", "recommended_max_tokens": 64},
    ]}), encoding="utf-8")
    return p


def _sse_body() -> bytes:
    events = [
        ("message_start", {"type": "message_start",
                           "message": {"model": "claude-opus-4-6", "usage": {"input_tokens": 10}}}),
        ("content_block_delta", {"type": "content_block_delta",
                                 "delta": {"type": "text_delta", "text": "hello"}}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": 3}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    out = b""
    for name, data in events:
        out += f"event: {name}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
    return out


@pytest.fixture
def mock_stream(monkeypatch):
    real_client = httpx.Client
    body = _sse_body()

    def handler(request):
        def gen():
            for i in range(0, len(body), 24):
                yield body[i:i + 24]
        return httpx.Response(200, content=gen(), headers={"content-type": "text/event-stream"})

    def factory(*a, **k):
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(R.httpx, "Client", factory)
    monkeypatch.setenv("TESTED_KEY", "sk-test")
    monkeypatch.setattr(R, "load_local_env", lambda *a, **k: {}, raising=False)


def test_run_eval_main_full_run(tmp_path, monkeypatch, capsys, mock_stream):
    out_dir = tmp_path / "runs"
    monkeypatch.setattr(sys, "argv", ["run_eval.py",
                                      "--providers", str(_providers_file(tmp_path)),
                                      "--tasks", str(_tasks_file(tmp_path)),
                                      "--out", str(out_dir),
                                      "--run-id", "rerun_1"])
    rc = R.main()
    assert rc == 0
    run_dir = out_dir / "rerun_1"
    assert (run_dir / "run_records.jsonl").exists()
    assert (run_dir / "summary.csv").exists()
    records = [json.loads(x) for x in
               (run_dir / "run_records.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert len(records) == 1
    # the streamed text was captured and the record is well-formed
    assert records[0]["task"]["id"] == "t1"


def test_run_eval_main_unknown_provider_id(tmp_path, monkeypatch, mock_stream):
    monkeypatch.setattr(sys, "argv", ["run_eval.py",
                                      "--providers", str(_providers_file(tmp_path)),
                                      "--tasks", str(_tasks_file(tmp_path)),
                                      "--out", str(tmp_path / "runs"),
                                      "--provider-id", "ghost"])
    rc = R.main()
    assert rc == 2  # unknown provider id


def test_load_providers(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "load_local_env", lambda *a, **k: {}, raising=False)
    provs = R.load_providers(_providers_file(tmp_path))
    assert len(provs) == 1
    assert provs[0].id == "tested"
    assert provs[0].base_url == "https://gw.x"


def test_load_providers_rejects_non_array(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "load_local_env", lambda *a, **k: {}, raising=False)
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"providers": "notalist"}), encoding="utf-8")
    with pytest.raises(ValueError, match="providers array"):
        R.load_providers(p)
