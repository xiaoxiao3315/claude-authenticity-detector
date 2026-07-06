"""MockTransport streaming coverage for compatibility.run_compatibility_suite (R13).

run_compatibility_suite opens a streaming httpx request per case. We patch
compatibility.httpx.Client to a MockTransport that replays a genuine Anthropic
SSE event sequence, so the full suite runner — call_streaming_messages,
run_probe, evaluate_case, manifest assembly — runs offline end to end.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import compatibility as K  # noqa: E402
from run_eval import Provider  # noqa: E402


def _sse_body() -> bytes:
    events = [
        ("message_start", {"type": "message_start",
                           "message": {"model": "claude-opus-4-6",
                                       "usage": {"input_tokens": 10}}}),
        ("content_block_start", {"type": "content_block_start",
                                 "content_block": {"type": "text"}}),
        ("content_block_delta", {"type": "content_block_delta",
                                 "delta": {"type": "text_delta", "text": "hello"}}),
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": 3}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    out = b""
    for name, data in events:
        out += f"event: {name}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
    return out


@pytest.fixture
def mock_stream_client(monkeypatch):
    real_client = httpx.Client
    body = _sse_body()

    def handler(request: httpx.Request) -> httpx.Response:
        def gen():
            for i in range(0, len(body), 24):
                yield body[i:i + 24]
        return httpx.Response(200, content=gen(),
                              headers={"content-type": "text/event-stream"})

    class _Factory:
        def __init__(self, *a, **k):
            self._c = real_client(transport=httpx.MockTransport(handler))
        def __enter__(self):
            return self._c
        def __exit__(self, *a):
            self._c.close()

    monkeypatch.setattr(K.httpx, "Client", _Factory)
    monkeypatch.setenv("TESTED_KEY", "sk-test")
    monkeypatch.setattr(K, "load_local_env", lambda *a, **k: {}, raising=False)
    # auth_header lives in run_eval (imported into compatibility); patch its env loader too
    import run_eval
    monkeypatch.setattr(run_eval, "load_local_env", lambda *a, **k: {}, raising=False)


def _provider():
    return Provider(id="tested", base_url="https://gw.x", model="claude-opus-4-6",
                    auth_type="x-api-key", auth_env="TESTED_KEY")


def _suite_file(tmp_path: Path) -> Path:
    suite = {
        "suite_version": "compatibility_suite_v1",
        "default_max_tokens": 64,
        "cases": [
            {"id": "msg_probe", "category": "messages",
             "request": {"prompt": "say hello"}, "expected_substring": "hello"},
            {"id": "sse_probe", "category": "sse", "request": {"prompt": "stream please"}},
        ],
    }
    p = tmp_path / "suite.json"
    p.write_text(json.dumps(suite), encoding="utf-8")
    return p


# placeholder-r13


# ---------------------------------------------------------------------------
# call_streaming_messages — single streamed request
# ---------------------------------------------------------------------------
def test_call_streaming_messages(tmp_path, mock_stream_client):
    real = K.httpx.Client  # the patched factory
    with real() as client:
        metrics, text = K.call_streaming_messages(
            client, _provider(), {"model": "claude-opus-4-6", "messages": [], "stream": True},
            tmp_path / "ev.jsonl")
    assert metrics.http_status == 200
    assert text == "hello"
    assert metrics.server_model == "claude-opus-4-6"
    assert "message_start" in metrics.event_types
    assert metrics.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# run_compatibility_suite — full offline suite run
# ---------------------------------------------------------------------------
def test_run_compatibility_suite_end_to_end(tmp_path, mock_stream_client):
    runs = tmp_path / "runs"
    runs.mkdir()
    result = K.run_compatibility_suite(
        runs_dir=runs, provider=_provider(), suite_path=_suite_file(tmp_path), run_id="compat_1")
    assert result["run_id"] == "compat_1"
    assert result["manifest"]["status"] == "completed"
    # both cases graded; genuine Anthropic stream -> not FAIL overall
    assert result["suite_status"] in {"PASS", "WARN"}
    # records + summary written
    run_dir = runs / "compat_1"
    assert (run_dir / "compatibility_records.jsonl").exists()
    assert (run_dir / "compatibility_summary.csv").exists()
    records = K.read_jsonl(run_dir / "compatibility_records.jsonl")
    assert len(records) == 2
    case_ids = {r["case_id"] for r in records}
    assert case_ids == {"msg_probe", "sse_probe"}


def test_run_compatibility_suite_progress_events(tmp_path, mock_stream_client):
    runs = tmp_path / "runs"
    runs.mkdir()
    events = []
    K.run_compatibility_suite(
        runs_dir=runs, provider=_provider(), suite_path=_suite_file(tmp_path),
        run_id="compat_2", progress_callback=lambda e: events.append(e["event"]))
    assert "run_started" in events
    assert "task_completed" in events


def test_run_compatibility_suite_stop_requested(tmp_path, mock_stream_client):
    runs = tmp_path / "runs"
    runs.mkdir()
    result = K.run_compatibility_suite(
        runs_dir=runs, provider=_provider(), suite_path=_suite_file(tmp_path),
        run_id="compat_3", job_control={"stop_requested": True})
    assert result["manifest"]["status"] == "stopped"


# ---------------------------------------------------------------------------
# list / read compatibility runs
# ---------------------------------------------------------------------------
def test_list_and_read_compatibility_runs(tmp_path, mock_stream_client):
    runs = tmp_path / "runs"
    runs.mkdir()
    K.run_compatibility_suite(runs_dir=runs, provider=_provider(),
                              suite_path=_suite_file(tmp_path), run_id="compat_x")
    listed = K.list_compatibility_runs(runs)
    assert any(r.get("run_id") == "compat_x" for r in listed)
    detail = K.read_compatibility_run(runs, "compat_x")
    assert detail["manifest"]["run_id"] == "compat_x" or detail.get("run_id") == "compat_x"

