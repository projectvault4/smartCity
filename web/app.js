const cardsRoot = document.getElementById("cards");
const summaryRoot = document.getElementById("summaryList");
const reasonsRoot = document.getElementById("reasons");
const forecastFor = document.getElementById("forecastFor");
const lastUpdated = document.getElementById("lastUpdated");
const errorBox = document.getElementById("errorBox");
const refreshButton = document.getElementById("refreshButton");
const explainableMetricsBody = document.getElementById("explainableMetricsBody");
const pearsonPairs = document.getElementById("pearsonPairs");
const spearmanPairs = document.getElementById("spearmanPairs");
const grangerLinks = document.getElementById("grangerLinks");

const cardTemplate = document.getElementById("cardTemplate");
const reasonTemplate = document.getElementById("reasonTemplate");

function formatDateTime(value) {
  return new Date(value).toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatValue(metric) {
  return Number(metric.prediction).toFixed(0);
}

function changeText(change) {
  if (Math.abs(change) < 1) {
    return "About the same as the latest reading";
  }
  if (change > 0) {
    return `Up by ${Math.round(change)} from the latest reading`;
  }
  return `Down by ${Math.round(Math.abs(change))} from the latest reading`;
}

function statusClass(status) {
  const normalized = status.toLowerCase();
  if (normalized.includes("good") || normalized.includes("normal") || normalized.includes("light") || normalized.includes("low") || normalized.includes("stable") || normalized.includes("cool")) {
    return "status-good";
  }
  if (normalized.includes("moderate") || normalized.includes("busy") || normalized.includes("high") || normalized.includes("warm")) {
    return "status-warn";
  }
  return "status-bad";
}

function renderCards(metrics) {
  cardsRoot.innerHTML = "";
  metrics.forEach((metric) => {
    const fragment = cardTemplate.content.cloneNode(true);
    fragment.querySelector(".metric-label").textContent = metric.label;
    fragment.querySelector(".status-pill").textContent = metric.status;
    fragment.querySelector(".status-pill").classList.add(statusClass(metric.status));
    fragment.querySelector(".metric-value").textContent = formatValue(metric);
    fragment.querySelector(".metric-range").textContent = `Expected range ${Math.round(metric.lower)} to ${Math.round(metric.upper)}`;
    fragment.querySelector(".metric-change").textContent = changeText(metric.change);
    cardsRoot.appendChild(fragment);
  });
}

function renderSummary(lines) {
  summaryRoot.innerHTML = "";
  lines.forEach((line) => {
    const item = document.createElement("li");
    item.textContent = line;
    summaryRoot.appendChild(item);
  });
}

function renderReasons(metrics) {
  reasonsRoot.innerHTML = "";
  metrics.forEach((metric) => {
    const fragment = reasonTemplate.content.cloneNode(true);
    fragment.querySelector("h3").textContent = metric.label;
    const list = fragment.querySelector("ul");
    metric.explanations.forEach((reason) => {
      const item = document.createElement("li");
      item.textContent = reason;
      list.appendChild(item);
    });
    reasonsRoot.appendChild(fragment);
  });
}

function renderExplainableMetrics(rows) {
  explainableMetricsBody.innerHTML = "";
  if (!rows || !rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="4">Explainable model quality will appear here after the forecast loads.</td>`;
    explainableMetricsBody.appendChild(tr);
    return;
  }
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.target}</td>
      <td>${row.MAE.toFixed(2)}</td>
      <td>${row.RMSE.toFixed(2)}</td>
      <td>${row.NRMSE.toFixed(4)}</td>
    `;
    explainableMetricsBody.appendChild(tr);
  });
}

function renderPairs(root, pairs, label) {
  root.innerHTML = "";
  if (!pairs || !pairs.length) {
    const item = document.createElement("div");
    item.className = "chip-item";
    item.textContent = `${label} correlation results will appear here after the analytics load.`;
    root.appendChild(item);
    return;
  }
  pairs.forEach((pair) => {
    const item = document.createElement("div");
    item.className = "chip-item";
    item.innerHTML = `<strong>${pair.left} ↔ ${pair.right}</strong><span>${label}: ${pair.value.toFixed(3)}</span>`;
    root.appendChild(item);
  });
}

function renderGranger(rows) {
  grangerLinks.innerHTML = "";
  if (!rows || !rows.length) {
    const item = document.createElement("div");
    item.className = "insight-item";
    item.textContent = "Granger causality highlights will appear here after the analytics load.";
    grangerLinks.appendChild(item);
    return;
  }
  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "insight-item";
    const lagText = row.lag ? `best lag ${row.lag}` : "lag unavailable";
    const significance = row.significant ? "significant at 5%" : "not significant at 5%";
    item.innerHTML = `<strong>${row.cause} → ${row.effect}</strong><span>${lagText}, p-value ${row.p_value}, ${significance}</span>`;
    grangerLinks.appendChild(item);
  });
}

function setError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

async function loadForecast() {
  refreshButton.disabled = true;
  refreshButton.textContent = "Refreshing...";
  clearError();

  try {
    const response = await fetch("/api/forecast", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Unable to load forecast");
    }

    forecastFor.textContent = formatDateTime(payload.forecast_for);
    lastUpdated.textContent = formatDateTime(payload.last_updated);
    renderCards(payload.metrics);
    renderSummary(payload.summary);
    renderReasons(payload.metrics);
    renderExplainableMetrics(payload.explainable_metrics);
    renderPairs(pearsonPairs, payload.analytics.pearson_top_pairs, "Pearson");
    renderPairs(spearmanPairs, payload.analytics.spearman_top_pairs, "Spearman");
    renderGranger(payload.analytics.granger_top_links);
  } catch (error) {
    setError(error.message);
  } finally {
    refreshButton.disabled = false;
    refreshButton.textContent = "Refresh Forecast";
  }
}

refreshButton.addEventListener("click", loadForecast);
loadForecast();
