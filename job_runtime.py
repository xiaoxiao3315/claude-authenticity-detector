from __future__ import annotations

from datetime import datetime
from threading import Event, RLock
from typing import Any, Callable


ProgressFn = Callable[[dict[str, Any]], None]
ResumeWaiterFn = Callable[[dict[str, Any] | None, ProgressFn | None, int, int], bool]

TERMINAL_JOB_STATUSES = {"completed", "failed", "stopped"}
CONTROLLABLE_JOB_STATUSES = {"queued", "running", "pausing", "paused"}
ARTIFACT_ID_FIELDS = (
    "run_id",
    "compatibility_run_id",
    "rescore_id",
    "import_run_id",
    "campaign_id",
    "trace_eval_id",
    "gate_id",
    "audit_export_id",
)


def _safe_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def progress_event(event: str, **fields: Any) -> dict[str, Any]:
    payload = {"event": event}
    payload.update(fields)
    return payload


def emit_progress_event(
    progress_callback: ProgressFn | None,
    event: str,
    **fields: Any,
) -> None:
    if progress_callback:
        progress_callback(progress_event(event, **fields))


def normalize_progress_event_patch(
    event: dict[str, Any],
    *,
    total_tasks: int,
    current_completed: int = 0,
    current_success_count: int = 0,
    current_failure_count: int = 0,
    current_last_error: Any = None,
    current_provider: Any = None,
    current_task_id: Any = None,
    initial_phase: str | None = None,
    default_task_phase: str = "calling_model",
    default_completed_phase: str = "completed",
    now_iso: Callable[[], str] | None = None,
) -> dict[str, Any] | None:
    event_type = event.get("event")
    current_time = now_iso or (lambda: datetime.now().isoformat(timespec="seconds"))
    if event_type == "run_started":
        patch: dict[str, Any] = {
            "total_tasks": event.get("total_tasks", total_tasks),
            "current_provider": event.get("current_provider", current_provider),
        }
        phase = event.get("phase") or initial_phase
        if phase:
            patch["current_phase"] = phase
        return patch
    if event_type == "task_started":
        return {
            "current_task_id": event.get("current_task_id", current_task_id),
            "current_provider": event.get("current_provider", current_provider),
            "current_phase": event.get("phase") or default_task_phase,
            "completed_tasks": event.get("completed_tasks", current_completed),
        }
    if event_type == "task_phase":
        patch = {
            "current_phase": event.get("phase") or default_task_phase,
            "completed_tasks": event.get("completed_tasks", current_completed),
        }
        if "total_tasks" in event:
            patch["total_tasks"] = event.get("total_tasks", total_tasks)
        if "current_task_id" in event:
            patch["current_task_id"] = event.get("current_task_id")
        if "current_provider" in event:
            patch["current_provider"] = event.get("current_provider")
        return patch
    if event_type == "task_completed":
        ok = bool(event.get("ok"))
        return {
            "completed_tasks": event.get("completed_tasks", current_completed),
            "current_phase": event.get("phase") or default_completed_phase,
            "success_count": current_success_count + (1 if ok else 0),
            "failure_count": current_failure_count + (0 if ok else 1),
            "last_error": event.get("error") if not ok else event.get("last_error", current_last_error),
        }
    if event_type == "run_paused":
        return {
            "status": "paused",
            "paused_at": current_time(),
            "current_task_id": None,
            "current_phase": "paused",
            "completed_tasks": event.get("completed_tasks", current_completed),
            "stop_reason": "paused",
        }
    if event_type == "run_resumed":
        return {
            "status": "running",
            "resumed_at": current_time(),
            "completed_tasks": event.get("completed_tasks", current_completed),
            "stop_reason": None,
        }
    if event_type == "run_stopped":
        stopped_at = current_time()
        return {
            "status": "stopped",
            "completed_at": stopped_at,
            "stopped_at": stopped_at,
            "current_task_id": None,
            "current_phase": "stopped",
            "completed_tasks": event.get("completed_tasks", current_completed),
            "stop_reason": event.get("stop_reason") or "user_stop_requested",
        }
    return None


class JobRuntime:
    def __init__(
        self,
        *,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.controls: dict[str, dict[str, Any]] = {}
        self.lock = RLock()
        self._id_factory = id_factory
        self._clock = clock or datetime.now

    def _now(self) -> datetime:
        return self._clock()

    def _now_iso(self) -> str:
        return self._now().isoformat(timespec="seconds")

    def _new_job_id(self) -> str:
        if self._id_factory:
            return self._id_factory()
        return self._now().strftime("job_%Y%m%d_%H%M%S_%f")

    def create_job(
        self,
        *,
        benchmark_mode: str,
        total_tasks: int,
        job_type: str | None = None,
        current_provider: str | None = None,
        initial_phase: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> str:
        job_id = self._new_job_id()
        resume_event = Event()
        resume_event.set()
        now = self._now_iso()
        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "paused_at": None,
            "stopped_at": None,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "benchmark_mode": benchmark_mode,
            "total_tasks": total_tasks,
            "completed_tasks": 0,
            "current_task_id": None,
            "current_provider": current_provider,
            "current_phase": initial_phase,
            "phase": initial_phase,
            "success_count": 0,
            "failure_count": 0,
            "percent": 0.0,
            "error": None,
            "last_error": None,
            "stop_reason": None,
        }
        if job_type:
            job["job_type"] = job_type
        if extra_fields:
            job.update(extra_fields)
        job["artifact_ids"] = self.artifact_ids(job)
        with self.lock:
            self.controls[job_id] = {
                "pause_requested": False,
                "stop_requested": False,
                "resume_event": resume_event,
            }
            self.jobs[job_id] = job
        return job_id

    def artifact_ids(self, job: dict[str, Any]) -> dict[str, Any]:
        if isinstance(job.get("artifact_ids"), dict):
            artifact_ids = {
                key: value
                for key, value in job["artifact_ids"].items()
                if key not in ARTIFACT_ID_FIELDS and value not in (None, "")
            }
        else:
            artifact_ids = {}
        for key in ARTIFACT_ID_FIELDS:
            value = job.get(key)
            if value not in (None, ""):
                artifact_ids[key] = value
        return artifact_ids

    def normalize_job(self, job: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(job)
        phase = enriched.get("current_phase") or enriched.get("status") or enriched.get("phase")
        enriched["phase"] = phase
        enriched["completed_tasks"] = _safe_int(enriched.get("completed_tasks"))
        enriched["total_tasks"] = _safe_int(enriched.get("total_tasks"))
        enriched["success_count"] = _safe_int(enriched.get("success_count"))
        enriched["failure_count"] = _safe_int(enriched.get("failure_count"))
        enriched["last_error"] = enriched.get("last_error") or enriched.get("error")
        enriched["artifact_ids"] = self.artifact_ids(enriched)

        status = str(enriched.get("status") or "")
        enriched["can_pause"] = status == "running"
        enriched["can_resume"] = status == "paused"
        enriched["can_stop"] = status in CONTROLLABLE_JOB_STATUSES
        return enriched

    def get_response(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return {"error": "job not found"}
            return self.normalize_job(job)

    def list_responses(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        with self.lock:
            jobs = sorted(
                (self.normalize_job(job) for job in self.jobs.values()),
                key=lambda job: job.get("created_at") or "",
                reverse=True,
            )
        return jobs if limit is None else jobs[:limit]

    def job_percent(self, job: dict[str, Any]) -> float:
        total = _safe_int(job.get("total_tasks"))
        completed = _safe_int(job.get("completed_tasks"))
        if total <= 0:
            return 0.0
        return round(min(100.0, completed * 100.0 / total), 2)

    def job_eta_seconds(self, job: dict[str, Any]) -> float | None:
        total = _safe_int(job.get("total_tasks"))
        completed = _safe_int(job.get("completed_tasks"))
        elapsed = _to_float(job.get("elapsed_seconds"))
        if total <= 0 or completed <= 0 or elapsed is None:
            return None
        remaining = max(0, total - completed)
        return round((elapsed / completed) * remaining, 1)

    def update_progress(self, job_id: str, patch: dict[str, Any]) -> None:
        with self.lock:
            job = self.jobs[job_id]
            now = self._now()
            normalized_patch = dict(patch)
            if "phase" in normalized_patch and "current_phase" not in normalized_patch:
                normalized_patch["current_phase"] = normalized_patch["phase"]
            if "current_phase" in normalized_patch:
                normalized_patch["phase"] = normalized_patch["current_phase"]
            job.update(normalized_patch)
            job["percent"] = self.job_percent(job)
            job["updated_at"] = now.isoformat(timespec="seconds")
            started_at = job.get("started_at")
            if started_at:
                try:
                    completed_at = job.get("completed_at")
                    end = datetime.fromisoformat(completed_at) if completed_at else now
                    started = datetime.fromisoformat(started_at)
                    job["elapsed_seconds"] = round(max(0.0, (end - started).total_seconds()), 1)
                except (TypeError, ValueError):
                    pass
            job["eta_seconds"] = self.job_eta_seconds(job)
            job["artifact_ids"] = self.artifact_ids(job)

    def control_job(self, job_id: str, action: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise ValueError("job not found")
            status = str(job.get("status") or "")
            if status in TERMINAL_JOB_STATUSES:
                return self.normalize_job(job)
            control = self.controls.get(job_id)
            if not control:
                raise ValueError("job control not found")

            if action == "pause":
                if status in {"queued", "running"}:
                    control["pause_requested"] = True
                    resume_event = control.get("resume_event")
                    if isinstance(resume_event, Event):
                        resume_event.clear()
                    self.update_progress(
                        job_id,
                        {
                            "status": "pausing",
                            "stop_reason": "pause_requested",
                        },
                    )
            elif action == "resume":
                if status in {"paused", "pausing"}:
                    control["pause_requested"] = False
                    resume_event = control.get("resume_event")
                    if isinstance(resume_event, Event):
                        resume_event.set()
                    self.update_progress(
                        job_id,
                        {
                            "status": "running",
                            "resumed_at": self._now_iso(),
                            "stop_reason": None,
                        },
                    )
            elif action == "stop":
                if status in CONTROLLABLE_JOB_STATUSES:
                    control["stop_requested"] = True
                    control["pause_requested"] = False
                    resume_event = control.get("resume_event")
                    if isinstance(resume_event, Event):
                        resume_event.set()
                    self.update_progress(
                        job_id,
                        {
                            "status": "stopping",
                            "stop_reason": "user_stop_requested",
                        },
                    )
            else:
                raise ValueError(f"unsupported job action: {action}")
            return self.normalize_job(self.jobs[job_id])

    def job_value(self, job_id: str, key: str, default: Any = None) -> Any:
        with self.lock:
            return self.jobs.get(job_id, {}).get(key, default)

    def job_total(self, job_id: str) -> int:
        return _safe_int(self.job_value(job_id, "total_tasks"))

    def job_completed(self, job_id: str) -> int:
        return _safe_int(self.job_value(job_id, "completed_tasks"))

    def job_has_field(self, job_id: str, key: str) -> bool:
        with self.lock:
            return key in self.jobs.get(job_id, {})

    def get_control(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            return self.controls.get(job_id)

    def drop_control_if_terminal(self, job_id: str) -> None:
        with self.lock:
            if self.jobs.get(job_id, {}).get("status") in TERMINAL_JOB_STATUSES:
                self.controls.pop(job_id, None)


def wait_for_job_resume(
    job_control: dict[str, Any] | None,
    progress_callback: ProgressFn | None,
    completed_tasks: int,
    total_tasks: int,
) -> bool:
    if not job_control or not job_control.get("pause_requested"):
        return False
    emit_progress_event(
        progress_callback,
        "run_paused",
        completed_tasks=completed_tasks,
        total_tasks=total_tasks,
    )
    resume_event = job_control.get("resume_event")
    while job_control.get("pause_requested") and not job_control.get("stop_requested"):
        if isinstance(resume_event, Event):
            resume_event.wait(0.25)
        else:
            break
    if job_control.get("stop_requested"):
        return True
    emit_progress_event(
        progress_callback,
        "run_resumed",
        completed_tasks=completed_tasks,
        total_tasks=total_tasks,
    )
    return False


def job_stop_requested(job_control: dict[str, Any] | None) -> bool:
    return bool(job_control and job_control.get("stop_requested"))


def check_job_pause_stop(
    job_control: dict[str, Any] | None,
    progress_callback: ProgressFn | None,
    *,
    completed_tasks: int,
    total_tasks: int,
    resume_waiter: ResumeWaiterFn | None = None,
) -> bool:
    if job_stop_requested(job_control):
        return True
    waiter = resume_waiter or wait_for_job_resume
    return bool(waiter(job_control, progress_callback, completed_tasks, total_tasks))


def _self_test() -> None:
    fixed_now = datetime(2026, 1, 1, 12, 0, 0)
    job_ids = iter(("job_test", "job_status"))
    runtime = JobRuntime(id_factory=lambda: next(job_ids), clock=lambda: fixed_now)
    job_id = runtime.create_job(
        benchmark_mode="mode_10",
        total_tasks=2,
        current_provider="provider_a",
        initial_phase="preparing",
        extra_fields={"run_id": "run_1"},
    )
    created = runtime.get_response(job_id)
    assert created["phase"] == "preparing"
    assert created["current_phase"] == "preparing"
    assert created["completed_tasks"] == 0
    assert created["total_tasks"] == 2
    assert created["success_count"] == 0
    assert created["failure_count"] == 0
    assert created["last_error"] is None
    assert created["artifact_ids"] == {"run_id": "run_1"}

    status_job_id = runtime.create_job(benchmark_mode="mode_10", total_tasks=1)
    assert runtime.get_response(status_job_id)["phase"] == "queued"
    runtime.update_progress(status_job_id, {"status": "running"})
    assert runtime.get_response(status_job_id)["phase"] == "running"

    runtime.update_progress(
        job_id,
        {
            "status": "running",
            "started_at": fixed_now.isoformat(timespec="seconds"),
            "current_phase": "calling_model",
            "completed_tasks": 1,
            "success_count": 1,
        },
    )
    running = runtime.get_response(job_id)
    assert running["phase"] == "calling_model"
    assert running["percent"] == 50.0
    assert running["can_pause"] is True

    pausing = runtime.control_job(job_id, "pause")
    assert pausing["status"] == "pausing"
    assert pausing["can_stop"] is True
    resumed = runtime.control_job(job_id, "resume")
    assert resumed["status"] == "running"
    stopping = runtime.control_job(job_id, "stop")
    assert stopping["status"] == "stopping"
    assert stopping["stop_reason"] == "user_stop_requested"

    completed_patch = normalize_progress_event_patch(
        {"event": "task_completed", "ok": True, "completed_tasks": 1, "phase": "completed"},
        total_tasks=2,
        current_success_count=3,
        current_failure_count=1,
    )
    assert completed_patch == {
        "completed_tasks": 1,
        "current_phase": "completed",
        "success_count": 4,
        "failure_count": 1,
        "last_error": None,
    }

    stopped_patch = normalize_progress_event_patch(
        {"event": "run_stopped", "completed_tasks": 1},
        total_tasks=2,
        now_iso=lambda: "2026-01-01T12:00:00",
    )
    assert stopped_patch is not None
    assert stopped_patch["status"] == "stopped"
    assert stopped_patch["completed_at"] == "2026-01-01T12:00:00"
    assert stopped_patch["stop_reason"] == "user_stop_requested"

    runtime.update_progress(
        job_id,
        {
            "status": "stopped",
            "completed_at": fixed_now.isoformat(timespec="seconds"),
            "current_phase": "stopped",
        },
    )
    runtime.drop_control_if_terminal(job_id)
    assert runtime.get_control(job_id) is None
    assert wait_for_job_resume(None, None, 0, 1) is False


if __name__ == "__main__":
    _self_test()
    print("job_runtime self-test ok")
