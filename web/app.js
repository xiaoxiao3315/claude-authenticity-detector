const state = {
  currentJobId: null,
  currentCampaignId: null,
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
    button.innerHTML = `<strong>${escapeHtml(job.job_id)}</strong><span>${escapeHtml(statusText(job.status))} / ${escapeHtml(decisionText(job.final_decision || "pending"))}</span>`;
    button.addEventListener("click", () => loadJob(job.job_id));
    list.appendChild(button);
  }
}

function renderPackLink(linkId, statusId, pack, href) {
  const link = $(linkId);
  const status = $(statusId);
  const verification = pack && pack.verification ? pack.verification : null;
  const verified = verification && verification.verified === true;
  if (link) {
    if (pack && href && verified) {
      link.href = href;
      link.classList.remove("link-disabled");
    } else {
      link.href = "#";
      link.classList.add("link-disabled");
    }
  }
  if (status) {
    if (!pack) {
      status.textContent = "未导出";
    } else if (verified) {
      status.textContent = "已校验";
    } else {
      status.textContent = `需重新导出 / ${verification?.error || "未校验"}`;
    }
  }
}

async function refreshJobs() {
  const [jobsResult, leaderboardResult] = await Promise.allSettled([
    api("/api/jobs"),
    api("/api/leaderboard?limit=20"),
  ]);
  if (jobsResult.status === "fulfilled") {
    renderJobs(jobsResult.value.jobs || []);
  } else {
    $("jobList").innerHTML = `<div class="muted">${escapeHtml(jobsResult.reason.message || "任务列表加载失败")}</div>`;
  }
  if (leaderboardResult.status === "fulfilled") {
    renderLeaderboard(leaderboardResult.value);
  } else {
    $("leaderboardStatus").textContent = leaderboardResult.reason.message || "排行加载失败";
    $("leaderboardBody").innerHTML = '<tr><td colspan="15" class="empty-table">排行加载失败，可刷新重试</td></tr>';
  }
}

async function openLatest() {
  const latest = await api("/api/jobs/latest");
  if (latest && latest.job_id) {
    await loadJob(latest.job_id);
  }
}

async function loadJob(jobId) {
  state.currentJobId = jobId;
  const jobState = await api(`/api/jobs/${encodeURIComponent(jobId)}/state`);
  const [summaryResult, resultsResult, eventsResult, artifactsResult] = await Promise.allSettled([
    api(`/api/jobs/${encodeURIComponent(jobId)}/summary`),
    api(`/api/jobs/${encodeURIComponent(jobId)}/results`),
    api(`/api/jobs/${encodeURIComponent(jobId)}/events`),
    api(`/api/jobs/${encodeURIComponent(jobId)}/artifacts`),
  ]);
  const artifacts = artifactsResult.status === "fulfilled" ? artifactsResult.value : { artifacts: [] };
  renderState(jobState, artifacts.artifacts || []);
  if (summaryResult.status === "fulfilled") renderSummary(summaryResult.value);
  if (resultsResult.status === "fulfilled") renderResults(resultsResult.value || []);
  if (eventsResult.status === "fulfilled") {
    renderEvents(eventsResult.value.events || []);
  } else {
    $("eventLog").textContent = eventsResult.reason.message || "事件日志加载失败";
  }
}

function renderState(jobState, artifacts) {
  $("jobTitle").textContent = jobState.job_id || "未加载任务";
  $("jobStatus").textContent = statusText(jobState.status);
  const progress = jobState.progress || {};
  $("jobProgress").textContent = `${progress.completed || 0} / ${progress.total || 0}`;
  $("jobDecision").textContent = decisionText(jobState.final_decision);
  $("jobDecision").className = decisionClass(jobState.final_decision);
  const pack = artifacts.find((item) => item.name === "acceptance_pack.zip");
  renderPackLink("downloadPack", "downloadPackStatus", pack, jobState.job_id ? `/api/jobs/${encodeURIComponent(jobState.job_id)}/artifacts/acceptance_pack.zip` : "");
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
  status.textContent = rows.length ? `${rows.length} 个 campaign / 兼容组 ${data.selected_comparison_key_id || "-"}` : "暂无真实 campaign 排行";
  body.innerHTML = "";
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="15" class="empty-table">暂无 completed live campaign；dry-run 可通过 API 参数 include_dry_run=true 查看</td></tr>';
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.className = "leaderboard-row";
    tr.tabIndex = 0;
    tr.dataset.campaignId = row.campaign_id || "";
    tr.innerHTML = `
      <td><span class="rank-badge">#${escapeHtml(row.rank || "-")}</span></td>
      <td>
        <strong>${escapeHtml(row.tested_model || "-")}</strong>
        <span class="cell-note">${escapeHtml(row.tested_provider_id || "-")} / ${row.live_provider ? "真实" : "干跑"}</span>
      </td>
      <td>${escapeHtml(row.judge_model || "-")}</td>
      <td>${escapeHtml(row.completed_runs ?? 0)} / ${escapeHtml(row.total_runs ?? 0)}</td>
      <td>${escapeHtml(row.total_cases ?? 0)}</td>
      <td>${escapeHtml(fmtPercent(row.model_response_success_rate))}</td>
      <td>${escapeHtml(fmtPercent(row.transport_success_rate))}</td>
      <td>${escapeHtml(fmtNumber(row.average_quality_score, 2))}</td>
      <td>${escapeHtml(fmtPercent(row.protocol_compatibility_score))}</td>
      <td>${escapeHtml(fmtPercent(row.model_name_consistency_rate))}</td>
      <td>${escapeHtml(fmtMs(row.p50_latency_ms))} / ${escapeHtml(fmtMs(row.p95_latency_ms))}</td>
      <td><span class="${decisionClass(row.model_confidence_decision)}">${escapeHtml(decisionText(row.model_confidence_decision))}</span></td>
      <td><span class="${decisionClass(row.gateway_reliability_decision)}">${escapeHtml(decisionText(row.gateway_reliability_decision))}</span></td>
      <td><span class="${decisionClass(row.overall_decision)}">${escapeHtml(decisionText(row.overall_decision))}</span></td>
      <td><code>${escapeHtml(row.campaign_id || "-")}</code><span class="cell-note">${escapeHtml(fmtDate(row.latest_tested_at))}</span></td>
    `;
    const openCampaign = () => {
      if (tr.dataset.campaignId) loadCampaign(tr.dataset.campaignId);
    };
    tr.addEventListener("click", openCampaign);
    tr.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openCampaign();
      }
    });
    body.appendChild(tr);
  }
}

async function loadCampaign(campaignId) {
  state.currentCampaignId = campaignId;
  const [summary, runs, artifacts] = await Promise.all([
    api(`/api/campaigns/${encodeURIComponent(campaignId)}/summary`),
    api(`/api/campaigns/${encodeURIComponent(campaignId)}/runs`),
    api(`/api/campaigns/${encodeURIComponent(campaignId)}/artifacts`),
  ]);
  renderCampaignDetail(summary, runs.runs || [], artifacts.artifacts || []);
}

function metricTile(label, value, hint = "") {
  return `
    <div class="mini-metric">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      ${hint ? `<em>${escapeHtml(hint)}</em>` : ""}
    </div>
  `;
}

function renderCampaignDetail(summary, runs, artifacts) {
  const target = $("campaignDetail");
  if (!target) return;
  const status = $("campaignDetailStatus");
  const pack = artifacts.find((item) => item.name === "acceptance_pack.zip");
  if (status) status.textContent = summary.campaign_id || "-";
  renderPackLink("campaignDownloadPack", "campaignDownloadPackStatus", pack, summary.campaign_id ? `/api/campaigns/${encodeURIComponent(summary.campaign_id)}/artifacts/acceptance_pack.zip` : "");
  const metrics = summary.metrics || {};
  const decisions = summary.decisions || {};
  const trend = summary.trend || [];
  const samples = summary.samples || [];
  const failures = summary.failure_counts || {};
  const maxBenchmark = Math.max(1, ...trend.map((item) => Number(item.benchmark_score || 0)));
  const maxFailure = Math.max(1, ...Object.values(failures).map((value) => Number(value || 0)));
  target.innerHTML = `
    <div class="campaign-summary-grid">
      ${metricTile("总轮数", `${metrics.completed_runs ?? 0} / ${metrics.total_runs ?? 0}`)}
      ${metricTile("总题数", metrics.total_cases ?? 0)}
      ${metricTile("模型成功率", fmtPercent(metrics.model_response_success_rate))}
      ${metricTile("网关成功率", fmtPercent(metrics.transport_success_rate))}
      ${metricTile("平均质量", fmtNumber(metrics.average_quality_score, 2), `中位 ${fmtNumber(metrics.median_quality_score, 2)}`)}
      ${metricTile("P50 / P95", `${fmtMs(metrics.p50_latency_ms)} / ${fmtMs(metrics.p95_latency_ms)}`)}
      ${metricTile("重试/替换", `${metrics.retried_request_count ?? 0} / ${metrics.total_retry_count ?? 0}`, `替换轮次 ${metrics.replaced_run_count ?? 0}`)}
    </div>
    <div class="decision-strip">
      <span class="${decisionClass(decisions.model_confidence_decision)}">模型身份/质量迹象：${escapeHtml(decisionText(decisions.model_confidence_decision))}</span>
      <span class="${decisionClass(decisions.gateway_reliability_decision)}">网关稳定性：${escapeHtml(decisionText(decisions.gateway_reliability_decision))}</span>
      <span class="${decisionClass(decisions.overall_decision)}">综合结论：${escapeHtml(decisionText(decisions.overall_decision))}</span>
    </div>
    <div class="detail-columns">
      <section>
        <h4>各轮质量趋势</h4>
        <div class="trend-list">
          ${trend.map((item) => {
            const width = Math.max(2, Math.min(100, Number(item.benchmark_score || 0) / maxBenchmark * 100));
            return `<div class="trend-row"><code>R${escapeHtml(item.round)}</code><span><i style="width:${width}%"></i></span><strong>${escapeHtml(fmtNumber(item.benchmark_score, 1))}</strong></div>`;
          }).join("") || '<div class="empty-note">暂无趋势数据</div>'}
        </div>
      </section>
      <section>
        <h4>失败原因分布</h4>
        <div class="failure-list">
          ${Object.entries(failures).map(([name, value]) => {
            const width = Math.max(2, Math.min(100, Number(value || 0) / maxFailure * 100));
            return `<div class="trend-row"><code>${escapeHtml(name)}</code><span><i class="failure-bar" style="width:${width}%"></i></span><strong>${escapeHtml(value)}</strong></div>`;
          }).join("") || '<div class="empty-note">暂无失败原因</div>'}
        </div>
      </section>
    </div>
    <div class="table-wrap detail-table">
      <table>
        <thead><tr><th>轮次</th><th>Run</th><th>状态</th><th>Benchmark</th><th>质量</th><th>网关成功率</th><th>P95</th></tr></thead>
        <tbody>
          ${(summary.child_runs || []).map((run) => `
            <tr>
              <td>R${escapeHtml(run.round)}</td>
              <td><code>${escapeHtml(run.run_id)}</code></td>
              <td>${escapeHtml(statusText(run.status))}</td>
              <td>${escapeHtml(fmtNumber(run.benchmark_score, 1))}</td>
              <td>${escapeHtml(fmtNumber(run.average_quality_score, 2))}</td>
              <td>${escapeHtml(fmtPercent(run.transport_success_rate))}</td>
              <td>${escapeHtml(fmtMs(run.p95_latency_ms))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
    <div class="table-wrap detail-table">
      <table>
        <thead><tr><th>题目</th><th>轮次</th><th>状态</th><th>得分</th><th>延迟</th><th>model_requested</th><th>model_returned</th><th>Judge 理由</th></tr></thead>
        <tbody>
          ${samples.slice(0, 120).map((sample) => `
            <tr>
              <td><code>${escapeHtml(sample.task_id || "-")}</code></td>
              <td>R${escapeHtml(sample.round || "-")}</td>
              <td>${sample.ok ? "通过" : `失败 / ${escapeHtml(sample.error_type || "unknown")}`}</td>
              <td>${escapeHtml(fmtNumber(sample.score, 1))}</td>
              <td>${escapeHtml(fmtMs(sample.latency_ms))}</td>
              <td>${escapeHtml(sample.model_requested || "-")}</td>
              <td>${escapeHtml(sample.model_returned || "-")}</td>
              <td>${escapeHtml(sample.judge_reason || sample.error || "")}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
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
  try {
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
  } catch (error) {
    $("configStatus").textContent = error.message || "保存失败";
  }
}

$("refreshJobs").addEventListener("click", refreshJobs);
$("openLatest").addEventListener("click", openLatest);
$("configForm").addEventListener("submit", saveConfig);

Promise.all([refreshJobs(), loadConfig()])
  .then(openLatest)
  .catch((error) => {
    $("eventLog").textContent = error.message;
  });
