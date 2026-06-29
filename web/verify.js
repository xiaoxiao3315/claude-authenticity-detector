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
  $("reportText").textContent = data.report_text || "(无报告文本)";
  showState("body");
}

// PLACEHOLDER_JS
async function runVerify(live) {
  const payload = collectPayload(live);
  if (!payload.base_url || !payload.model) {
    alert("请填写网关地址和模型");
    return;
  }
  $("loadingText").textContent = live ? "live 检测中（已强制 ≥2 秒间隔，请耐心等待）…" : "dry-run 验证中…";
  showState("loading");
  dryRunBtn.disabled = true;
  liveBtn.disabled = true;
  try {
    const resp = await fetch("/api/authenticity/verify", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) {
      $("verdictBanner").className = "verdict-banner v-bad";
      $("verdictIcon").textContent = "⛔";
      $("verdictTitle").textContent = "无法检测";
      $("verdictConf").textContent = "HTTP " + resp.status;
      $("confFill").style.width = "0%";
      $("reportText").textContent = data.error || JSON.stringify(data, null, 2);
      showState("body");
      return;
    }
    renderVerdict(data);
  } catch (err) {
    $("reportText").textContent = "请求失败：" + err;
    showState("body");
  } finally {
    dryRunBtn.disabled = false;
    liveBtn.disabled = !riskAck.checked;
  }
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

