const cardsRoot = document.getElementById("cards");
const summaryRoot = document.getElementById("summaryList");
const reasonsRoot = document.getElementById("reasons");
const forecastFor = document.getElementById("forecastFor");
const lastUpdated = document.getElementById("lastUpdated");
const sourceFreshness = document.getElementById("sourceFreshness");
const freshnessNote = document.getElementById("freshnessNote");
const limitingSource = document.getElementById("limitingSource");
const errorBox = document.getElementById("errorBox");
const refreshButton = document.getElementById("refreshButton");
const explainableMetricsBody = document.getElementById("explainableMetricsBody");
const pearsonPairs = document.getElementById("pearsonPairs");
const spearmanPairs = document.getElementById("spearmanPairs");
const grangerLinks = document.getElementById("grangerLinks");
const anomalyUpdated = document.getElementById("anomalyUpdated");
const severitySummary = document.getElementById("severitySummary");
const eventList = document.getElementById("eventList");
const anomalyTimeline = document.getElementById("anomalyTimeline");

const cardTemplate = document.getElementById("cardTemplate");
const reasonTemplate = document.getElementById("reasonTemplate");
const eventTemplate = document.getElementById("eventTemplate");

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

function formatDateOnly(value) {
  if (!value) {
    return "Unavailable";
  }
  return new Date(`${value}T00:00:00`).toLocaleDateString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function renderDataFreshness(freshness) {
  sourceFreshness.innerHTML = "";
  const sources = freshness?.sources || [];
  const limiting = freshness?.limiting_source;
  limitingSource.textContent = limiting ? `Limited by ${limiting}` : "Current source status";

  if (!sources.length) {
    const item = document.createElement("div");
    item.className = "source-item";
    item.innerHTML = `<strong>${formatDateOnly(freshness?.latest_prepared)}</strong><span>Prepared Dataset</span>`;
    sourceFreshness.appendChild(item);
  } else {
    sources.forEach((source) => {
      const item = document.createElement("div");
      item.className = "source-item";
      if (source.name === limiting) {
        item.classList.add("source-limiting");
      }
      item.innerHTML = `<strong>${formatDateOnly(source.latest)}</strong><span>${source.name}</span>`;
      sourceFreshness.appendChild(item);
    });
  }

  freshnessNote.textContent = freshness?.note || "";
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

function severityClass(severity) {
  return `severity-${String(severity || "low").toLowerCase()}`;
}

function renderSeveritySummary(counts) {
  severitySummary.innerHTML = "";
  ["Critical", "High", "Medium", "Low"].forEach((severity) => {
    const item = document.createElement("div");
    item.className = `severity-card ${severityClass(severity)}`;
    item.innerHTML = `<strong>${counts?.[severity] || 0}</strong><span>${severity}</span>`;
    severitySummary.appendChild(item);
  });
}

function renderEvents(events) {
  eventList.innerHTML = "";
  if (!events || !events.length) {
    const empty = document.createElement("p");
    empty.className = "helper-text";
    empty.textContent = "No urban anomalies crossed the current event threshold.";
    eventList.appendChild(empty);
    return;
  }
  events.forEach((event) => {
    const fragment = eventTemplate.content.cloneNode(true);
    const card = fragment.querySelector(".event-card");
    const severity = fragment.querySelector(".event-severity");
    severity.textContent = `${event.severity} · ${event.anomaly_score.toFixed(2)}`;
    severity.classList.add(severityClass(event.severity));
    card.classList.add(severityClass(event.severity));
    fragment.querySelector("h3").textContent = event.event_type;
    fragment.querySelector(".event-time").textContent = `${formatDateTime(event.timestamp)} · ${event.latitude.toFixed(4)}, ${event.longitude.toFixed(4)}`;
    fragment.querySelector(".event-description").textContent = event.description;
    const driverList = fragment.querySelector(".driver-list");
    event.drivers.forEach((driver) => {
      const chip = document.createElement("span");
      chip.textContent = `${driver.feature}: ${driver.contribution}`;
      driverList.appendChild(chip);
    });
    eventList.appendChild(fragment);
  });
}

function renderAnomalyTimeline(events) {
  anomalyTimeline.innerHTML = "";
  const sorted = [...(events || [])].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  if (!sorted.length) {
    anomalyTimeline.textContent = "Timeline appears after events are detected.";
    return;
  }
  sorted.forEach((event) => {
    const bar = document.createElement("span");
    bar.className = `timeline-bar ${severityClass(event.severity)}`;
    bar.style.height = `${Math.max(18, event.anomaly_score * 72)}px`;
    bar.title = `${event.event_type} · ${formatDateTime(event.timestamp)}`;
    anomalyTimeline.appendChild(bar);
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
    renderDataFreshness(payload.data_freshness);
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

async function loadAnomalies() {
  try {
    const response = await fetch("/api/anomalies", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Unable to load anomalies");
    }
    anomalyUpdated.textContent = formatDateTime(payload.last_updated);
    renderSeveritySummary(payload.severity_counts);
    renderEvents(payload.events);
    renderAnomalyTimeline(payload.events);
  } catch (error) {
    anomalyUpdated.textContent = "Unavailable";
    renderSeveritySummary({});
    eventList.innerHTML = `<p class="helper-text">${error.message}</p>`;
  }
}

refreshButton.addEventListener("click", () => {
  loadForecast();
  loadAnomalies();
});
loadForecast();
loadAnomalies();
