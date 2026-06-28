"""Tests for job_runtime pure helpers + progress-event state machine (T3)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import job_runtime as J  # noqa: E402


# ---------------------------------------------------------------------------
# _safe_int / _to_float / progress_event / emit_progress_event
# ---------------------------------------------------------------------------
def test_safe_int():
    assert J._safe_int("5") == 5
    assert J._safe_int(None, 3) == 3
    assert J._safe_int("", 3) == 3
    assert J._safe_int("bad", 7) == 7


def test_to_float():
    assert J._to_float("3.5") == 3.5
    assert J._to_float(None) is None
    assert J._to_float("") is None
    assert J._to_float("bad") is None


def test_progress_event():
    assert J.progress_event("run_started", total_tasks=3) == {"event": "run_started", "total_tasks": 3}


def test_emit_progress_event():
    seen = []
    J.emit_progress_event(lambda e: seen.append(e), "task_started", current_task_id="t1")
    assert seen == [{"event": "task_started", "current_task_id": "t1"}]
    # None callback is a no-op
    J.emit_progress_event(None, "x")


# ---------------------------------------------------------------------------
# normalize_progress_event_patch — every event branch
# ---------------------------------------------------------------------------
def _patch(event, **kw):
    kw.setdefault("total_tasks", 5)
    kw.setdefault("now_iso", lambda: "2026-06-28T00:00:00")
    return J.normalize_progress_event_patch(event, **kw)


def test_patch_run_started():
    p = _patch({"event": "run_started", "total_tasks": 7, "current_provider": "tested"},
               initial_phase="init")
    assert p["total_tasks"] == 7
    assert p["current_provider"] == "tested"
    assert p["current_phase"] == "init"


def test_patch_task_started():
    p = _patch({"event": "task_started", "current_task_id": "t1", "phase": "calling"})
    assert p["current_task_id"] == "t1"
    assert p["current_phase"] == "calling"


def test_patch_task_phase():
    p = _patch({"event": "task_phase", "phase": "scoring", "total_tasks": 9,
                "current_task_id": "t2", "current_provider": "p"})
    assert p["current_phase"] == "scoring"
    assert p["total_tasks"] == 9
    assert p["current_task_id"] == "t2"


def test_patch_task_completed_ok():
    p = _patch({"event": "task_completed", "ok": True, "completed_tasks": 2},
               current_success_count=1, current_failure_count=0)
    assert p["success_count"] == 2
    assert p["failure_count"] == 0
    assert p["completed_tasks"] == 2


def test_patch_task_completed_fail():
    p = _patch({"event": "task_completed", "ok": False, "error": "boom"},
               current_success_count=3, current_failure_count=1)
    assert p["success_count"] == 3
    assert p["failure_count"] == 2
    assert p["last_error"] == "boom"


def test_patch_run_paused():
    p = _patch({"event": "run_paused", "completed_tasks": 4})
    assert p["status"] == "paused"
    assert p["current_phase"] == "paused"
    assert p["paused_at"] == "2026-06-28T00:00:00"


def test_patch_run_resumed():
    p = _patch({"event": "run_resumed", "completed_tasks": 4})
    assert p["status"] == "running"
    assert p["stop_reason"] is None


def test_patch_run_stopped():
    p = _patch({"event": "run_stopped", "stop_reason": "user_stop_requested"})
    assert p["status"] == "stopped"
    assert p["current_phase"] == "stopped"
    assert p["stop_reason"] == "user_stop_requested"


def test_patch_unknown_event_is_none():
    assert _patch({"event": "mystery"}) is None


# ---------------------------------------------------------------------------
# pause / stop control
# ---------------------------------------------------------------------------
def test_job_stop_requested():
    assert J.job_stop_requested(None) is False
    assert J.job_stop_requested({}) is False
    assert J.job_stop_requested({"stop_requested": True}) is True


def test_wait_for_job_resume_no_pause():
    assert J.wait_for_job_resume(None, None, 0, 1) is False
    assert J.wait_for_job_resume({"pause_requested": False}, None, 0, 1) is False


def test_wait_for_job_resume_paused_then_stop():
    events = []
    jc = {"pause_requested": True, "stop_requested": True}  # paused but stop set -> no resume_event -> break
    out = J.wait_for_job_resume(jc, lambda e: events.append(e["event"]), 1, 3)
    assert out is True
    assert "run_paused" in events


def test_check_job_pause_stop_stop_short_circuits():
    assert J.check_job_pause_stop({"stop_requested": True}, None,
                                  completed_tasks=0, total_tasks=1) is True


def test_check_job_pause_stop_delegates_to_waiter():
    calls = {}
    def fake_waiter(jc, cb, completed, total):
        calls["hit"] = (completed, total)
        return False
    out = J.check_job_pause_stop({}, None, completed_tasks=2, total_tasks=5,
                                 resume_waiter=fake_waiter)
    assert out is False
    assert calls["hit"] == (2, 5)
