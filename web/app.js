const state = {
  currentJobId: null,
  leaderboard: { entries: [] },
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmtNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function fmtPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${Math.round(Number(value) * 100)}%`;
}

function fmtMs(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const n = Number(value);
  return n >= 1000 ? `${(n / 1000).toFixed(1)}s` : `${Math.round(n)}ms`;
}

function fmtDate(value) {
  if (!value) return "-";
  return String(value).replace("T", " ").slice(0, 16);
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

function decisionClass(value) {
  if (value === "GO") return "pill-go";
  if (value === "NO-GO") return "pill-nogo";
  return "pill-review";
}

function statusText(value) {
  const map = {
    completed: "已完成",
    failed: "失败",
    running: "运行中",
    pending: "等待中",
    cancelled: "已取消",
  };
  return map[value] || value || "-";
}

function decisionText(value) {
  const map = {
    GO: "GO / 通过",
    REVIEW: "REVIEW / 需复核",
    "NO-GO": "NO-GO / 不通过",
  };
  return map[value] || value || "-";
}

function renderJobs(jobs) {
  const list = $("jobList");
  list.innerHTML = "";
  if (!jobs.length) {
    list.innerHTML = '<div class="muted">暂无任务</div>';
    return;
  }
  for (const job of jobs) {
    const button = document.createElement("button");
    button.className = "job-item";
    button.innerHTML = `<strong>${job.job_id}</strong><span>${escapeHtml(statusText(job.status))} / ${escapeHtml(decisionText(job.final_decision || "pending"))}</span>`;
    button.addEventListener("click", () => loadJob(job.job_id));
    list.appendChild(button);
  }
}

async function refreshJobs() {
  const [jobsData, leaderboardData] = await Promise.all([
    api("/api/jobs"),
    api("/api/leaderboard?limit=20"),
  ]);
  renderJobs(jobsData.jobs || []);
  renderLeaderboard(leaderboardData);
}

async function openLatest() {
  const latest = await api("/api/jobs/latest");
  if (latest && latest.job_id) {
    await loadJob(latest.job_id);
  }
}

async function loadJob(jobId) {
  state.currentJobId = jobId;
  const [jobState, summary, results, events, artifacts] = await Promise.all([
    api(`/api/jobs/${encodeURIComponent(jobId)}/state`),
    api(`/api/jobs/${encodeURIComponent(jobId)}/summary`),
    api(`/api/jobs/${encodeURIComponent(jobId)}/results`),
    api(`/api/jobs/${encodeURIComponent(jobId)}/events`),
    api(`/api/jobs/${encodeURIComponent(jobId)}/artifacts`),
  ]);
  renderState(jobState, artifacts.artifacts || []);
  renderSummary(summary);
  renderResults(results || []);
  renderEvents(events.events || []);
}

function renderState(jobState, artifacts) {
  $("jobTitle").textContent = jobState.job_id || "未加载任务";
  $("jobStatus").textContent = statusText(jobState.status);
  const progress = jobState.progress || {};
  $("jobProgress").textContent = `${progress.completed || 0} / ${progress.total || 0}`;
  $("jobDecision").textContent = decisionText(jobState.final_decision);
  $("jobDecision").className = decisionClass(jobState.final_decision);
  const pack = artifacts.find((item) => item.name === "acceptance_pack.zip");
  const link = $("downloadPack");
  if (pack && jobState.job_id) {
    link.href = `/api/jobs/${encodeURIComponent(jobState.job_id)}/artifacts/acceptance_pack.zip`;
    link.classList.remove("link-disabled");
  } else {
    link.href = "#";
    link.classList.add("link-disabled");
  }
}

function renderSummary(summary) {
  const metrics = summary.metrics || {};
  const gate = summary.quality_gate || {};
  const cards = [
    ["成功率", fmtPercent(metrics.success_rate), `${metrics.ok_count ?? 0} / ${metrics.sample_count ?? 0}`],
    ["平均分", fmtNumber(metrics.average_score_0_10, 1), "验收模型打分"],
    ["Gate 分", fmtNumber(metrics.gate_score, 1), "验收门槛"],
    ["P95 延迟", fmtMs(metrics.p95_first_content_token_ms), "首 token"],
    ["Benchmark 分", fmtNumber(metrics.benchmark_score, 1), "加权总分"],
    ["质量分", fmtNumber(metrics.quality_score, 1), "0-1000"],
  ];
  $("metricBoard").innerHTML = cards.map(([label, value, hint]) => `
    <div class="metric-card">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <em>${escapeHtml(hint)}</em>
    </div>
  `).join("");

  $("gateStatus").textContent = decisionText(gate.decision);
  $("gateStatus").className = `muted ${decisionClass(gate.decision)}`;
  const blockers = (gate.blockers || []).map((item) => ({ ...item, kind: "阻断" }));
  const reviews = (gate.review_items || []).map((item) => ({ ...item, kind: "复核" }));
  const reasons = [...blockers, ...reviews];
  $("gateReasons").innerHTML = reasons.length ? reasons.map((item) => `
    <div class="reason-item ${item.kind === "阻断" ? "reason-blocker" : ""}">
      <strong>${escapeHtml(item.kind)} / ${escapeHtml(item.rule_id || "-")}</strong>
      <span>${escapeHtml(item.details || "")}</span>
      <code>${escapeHtml(item.metric || "")}: ${escapeHtml(item.observed ?? "-")} / ${escapeHtml(item.threshold ?? "-")}</code>
    </div>
  `).join("") : '<div class="empty-note">暂无验收原因</div>';

  const samples = summary.samples || [];
  $("scoreProfileStatus").textContent = `${samples.length} 题`;
  $("scoreBars").innerHTML = samples.map((item) => {
    const score = item.score === null || item.score === undefined ? 0 : Number(item.score);
    const width = Math.max(0, Math.min(100, score * 10));
    const status = item.ok ? "通过" : "失败";
    return `
      <div class="score-row">
        <div class="score-label">
          <code>${escapeHtml(item.task_id || "-")}</code>
          <span>${escapeHtml(status)} · ${escapeHtml(fmtMs(item.latency_ms))}</span>
        </div>
        <div class="score-track" title="${escapeHtml(item.error || "")}">
          <span class="${item.ok ? "" : "score-failed"}" style="width:${width}%"></span>
        </div>
        <strong>${escapeHtml(fmtNumber(item.score, 1))}</strong>
      </div>
    `;
  }).join("");
}

function renderLeaderboard(data) {
  const status = $("leaderboardStatus");
  const body = $("leaderboardBody");
  if (!status || !body) return;
  const rows = (data && data.entries) || [];
  state.leaderboard.entries = rows;
  status.textContent = rows.length ? `${rows.length} 个模型 / ${data.raw_run_count || 0} 次真实任务` : "暂无真实排行数据";
  body.innerHTML = "";
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="9" class="empty-table">暂无 benchmark 分数</td></tr>';
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.className = "leaderboard-row";
    tr.tabIndex = 0;
    tr.dataset.runId = row.latest_run_id || row.run_id || "";
    const decision = row.gate_decision || "-";
    tr.innerHTML = `
      <td><span class="rank-badge">#${escapeHtml(row.rank || "-")}</span></td>
      <td>
        <strong>${escapeHtml(row.model || "-")}</strong>
        <span class="cell-note">${escapeHtml(row.provider_display_name || row.provider_id || "-")} / ${escapeHtml(row.mode || "-")} / ${row.live_provider ? "真实" : "干跑"}</span>
      </td>
      <td><span class="${decisionClass(decision)}">${escapeHtml(decisionText(decision))}</span></td>
      <td>
        <strong>${escapeHtml(fmtNumber(row.score, 1))}</strong>
        <span class="cell-note">最新 ${escapeHtml(fmtNumber(row.latest_score, 1))}</span>
      </td>
      <td>${escapeHtml(fmtNumber(row.quality_score, 1))}</td>
      <td>${escapeHtml(fmtPercent(row.success_rate))}</td>
      <td>${escapeHtml(fmtMs(row.p95_first_content_token_ms))}</td>
      <td>${escapeHtml(row.history_count || 1)}</td>
      <td><code>${escapeHtml(row.latest_run_id || row.run_id || "-")}</code><span class="cell-note">${escapeHtml(fmtDate(row.generated_at || row.completed_at))}</span></td>
    `;
    const openRun = () => {
      if (tr.dataset.runId) loadJob(tr.dataset.runId);
    };
    tr.addEventListener("click", openRun);
    tr.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openRun();
      }
    });
    body.appendChild(tr);
  }
}

function renderResults(results) {
  $("resultCount").textContent = `${results.length} 题`;
  const body = $("resultsBody");
  body.innerHTML = "";
  for (const item of results) {
    const tr = document.createElement("tr");
    const score = item.score || {};
    tr.innerHTML = `
      <td><code>${escapeHtml(item.task?.id || "-")}</code></td>
      <td>${escapeHtml(item.tested_model || "-")}</td>
      <td>${escapeHtml(item.judge_model || "-")}</td>
      <td>${escapeHtml(score.score ?? "-")}</td>
      <td>${item.tested_ok ? "已完成" : "失败"}</td>
    `;
    body.appendChild(tr);
  }
}

function renderEvents(events) {
  $("eventCount").textContent = `${events.length} 条事件`;
  $("eventLog").textContent = events.slice(-120).map((event) => {
    const label = event.task_id ? `${event.type} ${event.task_id}` : event.type;
    return `${event.at || ""}  ${label || ""}`;
  }).join("\n");
}

async function loadConfig() {
  const data = await api("/api/config");
  const providers = data.providers || {};
  const tested = providers.tested_model || {};
  const judge = providers.judge_model || {};
  $("configStatus").textContent = data.exists ? "已加载本地配置" : "没有本地配置";
  const form = $("configForm");
  form.tested_base_url.value = tested.base_url || "";
  form.tested_model.value = tested.model || "";
  form.tested_protocol.value = tested.protocol || "openai_chat";
  form.tested_auth_type.value = tested.auth_type || "bearer";
  form.judge_base_url.value = judge.base_url || "";
  form.judge_model.value = judge.model || "";
  form.judge_protocol.value = judge.protocol || "openai_chat";
  form.judge_auth_type.value = judge.auth_type || "bearer";
}

async function saveConfig(event) {
  event.preventDefault();
  const form = event.currentTarget;
  await api("/api/config", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      providers: {
        tested_model: {
          provider_id: "tested_model",
          base_url: form.tested_base_url.value,
          model: form.tested_model.value,
          protocol: form.tested_protocol.value,
          auth_type: form.tested_auth_type.value,
          api_key: form.tested_api_key.value,
        },
        judge_model: {
          provider_id: "judge_model",
          base_url: form.judge_base_url.value,
          model: form.judge_model.value,
          protocol: form.judge_protocol.value,
          auth_type: form.judge_auth_type.value,
          api_key: form.judge_api_key.value,
        },
      },
    }),
  });
  form.tested_api_key.value = "";
  form.judge_api_key.value = "";
  await loadConfig();
}

$("refreshJobs").addEventListener("click", refreshJobs);
$("openLatest").addEventListener("click", openLatest);
$("configForm").addEventListener("submit", saveConfig);

Promise.all([refreshJobs(), loadConfig()])
  .then(openLatest)
  .catch((error) => {
    $("eventLog").textContent = error.message;
  });
