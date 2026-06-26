const state = {
  currentJobId: null,
  currentCampaignId: null,
  leaderboard: { entries: [] },
  config: { providers: {} },
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
  if (value === "GO" || value === "PASS") return "pill-go";
  if (value === "NO-GO" || value === "FAIL") return "pill-nogo";
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
    GO: "PASS / 通过",
    PASS: "PASS / 通过",
    REVIEW: "RETEST / 自动复验",
    RETEST: "RETEST / 自动复验",
    "NO-GO": "FAIL / 不通过",
    FAIL: "FAIL / 不通过",
    PENDING: "PENDING / 等待结果",
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
    let leaderboard = leaderboardResult.value;
    if (!((leaderboard.entries || []).length)) {
      try {
        leaderboard = await api("/api/leaderboard?include_dry_run=true&limit=20");
        leaderboard.dry_run_fallback = true;
      } catch (error) {
        leaderboard = leaderboardResult.value;
      }
    }
    renderLeaderboard(leaderboard);
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
  const reviews = (gate.review_items || []).map((item) => ({ ...item, kind: "复测" }));
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
  const fallback = data && data.dry_run_fallback;
  status.textContent = rows.length
    ? `${rows.length} 个 campaign / 兼容组 ${data.selected_comparison_key_id || "-"}${fallback ? " / dry-run fallback" : ""}`
    : "暂无真实 campaign 排行";
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
      <td><span class="${decisionClass(row.model_outcome || row.model_confidence_decision)}">${escapeHtml(decisionText(row.model_outcome || row.model_confidence_decision))}</span></td>
      <td><span class="${decisionClass(row.gateway_outcome || row.gateway_reliability_decision)}">${escapeHtml(decisionText(row.gateway_outcome || row.gateway_reliability_decision))}</span></td>
      <td><span class="${decisionClass(row.overall_outcome || row.overall_decision)}">${escapeHtml(decisionText(row.overall_outcome || row.overall_decision))}</span></td>
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
  const [summary, runs, artifacts, authenticity] = await Promise.all([
    api(`/api/campaigns/${encodeURIComponent(campaignId)}/summary`),
    api(`/api/campaigns/${encodeURIComponent(campaignId)}/runs`),
    api(`/api/campaigns/${encodeURIComponent(campaignId)}/artifacts`),
    api(`/api/campaigns/${encodeURIComponent(campaignId)}/authenticity`),
  ]);
  renderCampaignDetail(summary, runs.runs || [], artifacts.artifacts || [], authenticity);
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

function reasonChips(items) {
  const values = Array.isArray(items) ? items : [];
  if (!values.length) return '<span class="empty-note">无额外原因</span>';
  return values.slice(0, 8).map((item) => `<code>${escapeHtml(item)}</code>`).join("");
}

function renderCampaignDetail(summary, runs, artifacts, authenticity = {}) {
  const target = $("campaignDetail");
  if (!target) return;
  const status = $("campaignDetailStatus");
  const pack = artifacts.find((item) => item.name === "acceptance_pack.zip");
  if (status) status.textContent = summary.campaign_id || "-";
  renderPackLink("campaignDownloadPack", "campaignDownloadPackStatus", pack, summary.campaign_id ? `/api/campaigns/${encodeURIComponent(summary.campaign_id)}/artifacts/acceptance_pack.zip` : "");
  const metrics = summary.metrics || {};
  const decisions = summary.decisions || {};
  const outcomes = summary.outcomes || {};
  const trend = summary.trend || [];
  const samples = summary.samples || [];
  const failures = summary.failure_counts || {};
  const authMetrics = authenticity.metrics || {};
  const authDecisions = authenticity.decisions || {};
  const authReasons = authenticity.reasons || {};
  const authStats = authMetrics.statistical_confidence || {};
  const authStatReasons = [
    `样本 ${authStats.total_samples ?? 0}/${authStats.min_sample_threshold ?? 30}`,
    authStats.sample_threshold_met ? "sample_threshold_met" : "sample_threshold_not_met",
    Array.isArray(authStats.bootstrap_95_ci) ? `bootstrap_95_ci ${authStats.bootstrap_95_ci.join("..")}` : "bootstrap_ci_missing",
    Array.isArray(authStats.anomalies) && authStats.anomalies.length ? `anomalies ${authStats.anomalies.length}` : "no_anomaly",
  ];
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
      ${metricTile("下一步动作", outcomes.next_action || summary.next_action || "-", outcomes.next_action_reason || summary.next_action_reason || "")}
    </div>
    <section class="authenticity-panel">
      <h4>可信度证据层</h4>
      <div class="campaign-summary-grid">
        ${metricTile("模型质量", decisionText(authDecisions.model_quality_decision), fmtNumber(authMetrics.model_quality_score, 2))}
        ${metricTile("网关稳定性", decisionText(authDecisions.gateway_reliability_decision), fmtNumber(authMetrics.gateway_reliability_score, 2))}
        ${metricTile("协议指纹", decisionText(authDecisions.protocol_fingerprint_decision), fmtNumber(authMetrics.protocol_fingerprint_score, 2))}
        ${metricTile("基线相似度", decisionText(authDecisions.baseline_similarity_decision), fmtNumber(authMetrics.baseline_similarity_score, 2))}
        ${metricTile("可审计性", decisionText(authDecisions.auditability_decision), fmtNumber(authMetrics.auditability_score, 2))}
        ${metricTile("综合可信度", decisionText(authDecisions.overall_trust_decision), fmtNumber(authMetrics.overall_trust_score, 2))}
      </div>
      <div class="decision-strip">
        <span class="${decisionClass(authDecisions.model_quality_decision)}">模型质量：${escapeHtml(decisionText(authDecisions.model_quality_decision))}</span>
        <span class="${decisionClass(authDecisions.gateway_reliability_decision)}">网关稳定性：${escapeHtml(decisionText(authDecisions.gateway_reliability_decision))}</span>
        <span class="${decisionClass(authDecisions.protocol_fingerprint_decision)}">协议指纹：${escapeHtml(decisionText(authDecisions.protocol_fingerprint_decision))}</span>
        <span class="${decisionClass(authDecisions.baseline_similarity_decision)}">基线相似度：${escapeHtml(decisionText(authDecisions.baseline_similarity_decision))}</span>
        <span class="${decisionClass(authDecisions.auditability_decision)}">可审计性：${escapeHtml(decisionText(authDecisions.auditability_decision))}</span>
        <span class="${decisionClass(authDecisions.overall_trust_decision)}">综合可信度：${escapeHtml(decisionText(authDecisions.overall_trust_decision))}</span>
      </div>
      <div class="auth-reasons">
        <div><strong>协议证据</strong>${reasonChips(authReasons.protocol_fingerprint)}</div>
        <div><strong>基线证据</strong>${reasonChips(authReasons.baseline_similarity)}</div>
        <div><strong>审计证据</strong>${reasonChips(authReasons.auditability)}</div>
        <div><strong>统计证据</strong>${reasonChips(authStatReasons)}</div>
      </div>
    </section>
    <div class="decision-strip">
      <span class="${decisionClass(outcomes.model_outcome || summary.model_outcome || decisions.model_confidence_decision)}">模型身份/质量迹象：${escapeHtml(decisionText(outcomes.model_outcome || summary.model_outcome || decisions.model_confidence_decision))}</span>
      <span class="${decisionClass(outcomes.gateway_outcome || summary.gateway_outcome || decisions.gateway_reliability_decision)}">网关稳定性：${escapeHtml(decisionText(outcomes.gateway_outcome || summary.gateway_outcome || decisions.gateway_reliability_decision))}</span>
      <span class="${decisionClass(outcomes.overall_outcome || summary.overall_outcome || decisions.overall_decision)}">综合结论：${escapeHtml(decisionText(outcomes.overall_outcome || summary.overall_outcome || decisions.overall_decision))}</span>
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

function setDatalist(id, values) {
  const list = $(id);
  if (!list) return;
  list.innerHTML = "";
  for (const value of values || []) {
    const option = document.createElement("option");
    option.value = value;
    list.appendChild(option);
  }
}

function providerFor(role) {
  return (state.config.providers || {})[role] || {};
}

function configPayloadFor(role, prefix, form) {
  const current = providerFor(role);
  return {
    provider_id: current.provider_id || role,
    base_url: form[`${prefix}_base_url`].value,
    model: form[`${prefix}_model`].value,
    protocol: form[`${prefix}_protocol`].value,
    auth_type: form[`${prefix}_auth_type`].value,
    reasoning_effort: form[`${prefix}_reasoning_effort`].value,
    api_key: form[`${prefix}_api_key`].value,
  };
}

function renderProbeResult(result) {
  const output = $("probeOutput");
  const roleLabel = result.role === "tested_model" ? "被测模型" : "验收模型";
  const models = result.text_models || result.models || [];
  const modelPreview = models.slice(0, 12);
  const probe = result.reasoning_probe || {};
  const supported = probe.supported_values || ["none", "low", "medium", "high", "xhigh"];
  const rejected = (probe.rejected || []).map((item) => item.value);
  const datalistId = result.role === "tested_model" ? "testedModelOptions" : "judgeModelOptions";
  setDatalist(datalistId, models);
  output.classList.remove("muted");
  output.innerHTML = `
    <strong>${escapeHtml(roleLabel)}检测完成</strong>
    <span>模型数量：${escapeHtml(result.model_count ?? 0)}，文本模型：${escapeHtml(models.length)}</span>
    <span>可选模型：${modelPreview.map(escapeHtml).join("、") || "-"}</span>
    <span>推理强度选项：${supported.map(escapeHtml).join("、")}${rejected.length ? `；不支持：${rejected.map(escapeHtml).join("、")}` : ""}</span>
  `;
}

async function probeConfig(role) {
  const output = $("probeOutput");
  output.classList.add("muted");
  output.textContent = role === "tested_model" ? "正在检测被测 Key..." : "正在检测验收 Key...";
  try {
    const result = await api(`/api/config/probe?role=${encodeURIComponent(role)}&reasoning=false`);
    if (result.error) {
      output.textContent = result.error;
      return;
    }
    renderProbeResult(result);
  } catch (error) {
    output.textContent = error.message || "检测失败";
  }
}

async function loadConfig() {
  const data = await api("/api/config");
  state.config = data;
  const providers = data.providers || {};
  const tested = providers.tested_model || {};
  const judge = providers.judge_model || {};
  $("configStatus").textContent = data.exists ? "已加载本地配置" : "没有本地配置";
  const form = $("configForm");
  form.tested_base_url.value = tested.base_url || "";
  form.tested_model.value = tested.model || "";
  form.tested_protocol.value = tested.protocol || "openai_chat";
  form.tested_auth_type.value = tested.auth_type || "bearer";
  form.tested_reasoning_effort.value = tested.reasoning_effort || "";
  form.judge_base_url.value = judge.base_url || "";
  form.judge_model.value = judge.model || "";
  form.judge_protocol.value = judge.protocol || "openai_chat";
  form.judge_auth_type.value = judge.auth_type || "bearer";
  form.judge_reasoning_effort.value = judge.reasoning_effort || "";
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
          tested_model: configPayloadFor("tested_model", "tested", form),
          judge_model: configPayloadFor("judge_model", "judge", form),
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
$("probeTested").addEventListener("click", () => probeConfig("tested_model"));
$("probeJudge").addEventListener("click", () => probeConfig("judge_model"));

Promise.all([refreshJobs(), loadConfig()])
  .then(async () => {
    await openLatest();
    const firstCampaign = state.leaderboard.entries[0];
    if (!state.currentCampaignId && firstCampaign && firstCampaign.campaign_id) {
      await loadCampaign(firstCampaign.campaign_id);
    }
  })
  .catch((error) => {
    $("eventLog").textContent = error.message;
  });
