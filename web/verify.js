// Claude 真伪检测 — 前端逻辑 + anime.js 开场
const $ = (id) => document.getElementById(id);

const riskAck = $("riskAck");
const liveBtn = $("liveBtn");
const dryRunBtn = $("dryRunBtn");

// 风险勾选才解锁 live 按钮
riskAck.addEventListener("change", () => {
  liveBtn.disabled = !riskAck.checked;
});

function collectPayload(live) {
  return {
    base_url: $("baseUrl").value.trim(),
    model: $("model").value.trim(),
    protocol: $("protocol").value,
    auth_type: $("authType").value,
    baseline_id: $("baselineId").value.trim() || "OFFICIAL-CLAUDE-OPUS46",
    api_key: $("apiKey").value,
    with_capability: $("withCapability").checked,
    live: live,
    risk_ack: riskAck.checked,
  };
}

// 4 类判定 → 图标 / 文案 / 配色 class
const VERDICT_MAP = {
  matches_official:    { icon: "✅", label: "真·官方 Claude", cls: "v-ok" },
  suspected_downgrade: { icon: "⚠️", label: "疑似降级",       cls: "v-warn" },
  suspected_wrapper:   { icon: "❌", label: "疑似套壳",       cls: "v-bad" },
  insufficient_evidence: { icon: "❔", label: "证据不足",     cls: "v-unknown" },
};

function showState(state) {
  $("resultEmpty").hidden = state !== "empty";
  $("resultLoading").hidden = state !== "loading";
  $("resultBody").hidden = state !== "body";
}

// 强证据 check 名（与后端 baseline_registry STRONG 集合对齐）
const STRONG_CHECKS = new Set([
  "stop_reason_enum", "usage_naming_dialect", "model_id",
  "sse_event_order", "error_envelope", "needle_fake_1m", "request_failure_rate",
]);

function renderVerdict(data) {
  const v = data.verdict || {};
  const key = v.verdict || "insufficient_evidence";
  const meta = VERDICT_MAP[key] || VERDICT_MAP.insufficient_evidence;
  const banner = $("verdictBanner");
  banner.className = "verdict-banner " + meta.cls;
  $("verdictIcon").textContent = meta.icon;
  $("verdictTitle").textContent = meta.label + (data.live ? "" : "（dry-run）");
  const conf = typeof v.confidence === "number" ? v.confidence : 0;
  $("verdictConf").textContent = "置信度 " + conf.toFixed(2);
  $("confFill").style.width = Math.round(conf * 100) + "%";
  renderEvidence(v);
  $("reportText").textContent = data.report_text || "(无报告文本)";
  showState("body");
}

function renderEvidence(v) {
  const host = $("evidenceGroups");
  host.innerHTML = "";
  const chain = (v.evidence_chain || []).filter((e) => !e.probe_error);
  const perr = v.probe_errors || [];
  const strong = chain.filter((e) => STRONG_CHECKS.has(e.check) && !e.advisory);
  const corro = chain.filter((e) => !STRONG_CHECKS.has(e.check) && !e.advisory);
  const advisory = chain.filter((e) => e.advisory);

  if (perr.length) host.appendChild(group("⚠ 探针未完成", perr.map((p) => ({ text: p })), "g-err"));
  if (strong.length) host.appendChild(group("强证据（定罪级）", strong.map(fmtEv), "g-strong"));
  if (corro.length) host.appendChild(group("佐证（参考）", corro.map(fmtEv), "g-corro"));
  if (advisory.length) host.appendChild(group("仅参考（不计票）", advisory.map(fmtEv), "g-adv"));
  if (!host.children.length) {
    const p = document.createElement("p");
    p.className = "evidence-none";
    p.textContent = "（无结构化证据，见下方报告文本）";
    host.appendChild(p);
  }
}

function fmtEv(e) {
  const extra = ["order_ok", "silent_truncation", "result"]
    .filter((k) => e[k] != null).map((k) => `${k}=${e[k]}`).join(" ");
  return {
    label: e.check,
    base: e.baseline != null ? JSON.stringify(e.baseline) : "—",
    obs: e.observed != null ? JSON.stringify(e.observed) : "—",
    extra,
  };
}

function group(title, rows, cls) {
  const box = document.createElement("div");
  box.className = "evidence-group " + cls;
  const h = document.createElement("div");
  h.className = "evidence-title";
  h.textContent = title;
  box.appendChild(h);
  for (const r of rows) {
    const row = document.createElement("div");
    row.className = "evidence-row";
    if (r.text) {
      row.textContent = "✗ " + r.text;
    } else {
      row.innerHTML = `<span class="ev-check">${r.label}</span>` +
        `<span class="ev-cmp">基线 <code>${r.base}</code> → 实测 <code>${r.obs}</code>${r.extra ? " · " + r.extra : ""}</span>`;
    }
    box.appendChild(row);
  }
  return box;
}

// PLACEHOLDER_JS
async function runVerify(live) {
  const payload = collectPayload(live);
  if (!payload.base_url || !payload.model) {
    alert("请填写网关地址和模型");
    return;
  }
  showState("loading");
  setProgress(live ? "准备中…" : "dry-run 验证中…", null);
  dryRunBtn.disabled = true;
  liveBtn.disabled = true;
  try {
    const resp = await fetch("/api/authenticity/verify", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    const ctype = resp.headers.get("content-type") || "";
    if (ctype.includes("text/event-stream")) {
      await consumeSSE(resp);
    } else {
      const data = await resp.json();
      if (!resp.ok) { renderError(resp.status, data); return; }
      renderVerdict(data);
    }
  } catch (err) {
    renderError(0, { error: "请求失败：" + err });
  } finally {
    dryRunBtn.disabled = false;
    liveBtn.disabled = !riskAck.checked;
  }
}

function setProgress(label, frac) {
  $("loadingText").textContent = label;
  const bar = $("loadingBar");
  if (bar) bar.style.width = frac == null ? "" : Math.round(frac * 100) + "%";
}

function renderError(status, data) {
  $("verdictBanner").className = "verdict-banner v-bad";
  $("verdictIcon").textContent = "⛔";
  $("verdictTitle").textContent = "无法检测";
  $("verdictConf").textContent = status ? "HTTP " + status : "错误";
  $("confFill").style.width = "0%";
  $("evidenceGroups").innerHTML = "";
  $("reportText").textContent = (data && data.error) || JSON.stringify(data, null, 2);
  showState("body");
}

// 读 POST 返回的 SSE 流：event: progress / result / error
async function consumeSSE(resp) {
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const chunks = buf.split("\n\n");
    buf = chunks.pop();
    for (const chunk of chunks) {
      const ev = parseSSE(chunk);
      if (!ev) continue;
      if (ev.event === "progress") {
        const d = ev.data;
        const frac = d.total ? d.done / d.total : null;
        setProgress(d.label || d.stage || "检测中…", frac);
      } else if (ev.event === "result") {
        renderVerdict(ev.data);
      } else if (ev.event === "error") {
        renderError(0, ev.data);
      }
    }
  }
}

function parseSSE(chunk) {
  let event = "message", data = "";
  for (const line of chunk.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return null;
  try { return { event, data: JSON.parse(data) }; } catch { return null; }
}

dryRunBtn.addEventListener("click", () => runVerify(false));
liveBtn.addEventListener("click", () => runVerify(true));

// ---------- anime.js 开场序列（失败则静态，绝不空白） ----------
(async () => {
  try {
    const { animate, stagger } = await import("animejs");
    animate(".reveal", {
      opacity: [0, 1],
      translateY: [16, 0],
      duration: 620,
      delay: stagger(110),
      ease: "out(3)",
    });
  } catch (e) {
    // anime 未加载 → reveal 默认已可见，无需处理
  }
})();

