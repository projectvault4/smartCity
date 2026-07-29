const summaryCards = document.getElementById("summaryCards");
const topDaysBody = document.getElementById("topDaysBody");
const eventPanel = document.getElementById("eventPanel");
const heatmap = document.getElementById("heatmap");
const monthlyStats = document.getElementById("monthlyStats");
const insightsList = document.getElementById("insightsList");
const recommendations = document.getElementById("recommendations");
const errorBox = document.getElementById("errorBox");
const dataMode = document.getElementById("dataMode");

const startDate = document.getElementById("startDate");
const endDate = document.getElementById("endDate");
const domainFilter = document.getElementById("domainFilter");
const severityFilter = document.getElementById("severityFilter");
const resetFilters = document.getElementById("resetFilters");

let anomalyPayload = null;

function severityClass(severity) {
  return `severity-${String(severity || "normal").toLowerCase()}`;
}

function formatDate(value) {
  return new Date(value).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function formatScore(value) {
  return Number(value || 0).toFixed(3);
}

function cleanFeatureName(value) {
  return String(value || "Urban signal")
    .replace(/\bAqi\b/g, "AQI")
    .replace(/\bDiff\b/g, "Change")
    .replace(/\bRoll\b/g, "Rolling")
    .replace(/\bStd\b/g, "Spread")
    .replace(/\b Z \b/g, " ")
    .replace(/\bLag\b/g, "Lag")
    .replace(/\s+/g, " ")
    .trim();
}

function cleanDescription(value) {
  return String(value || "")
    .replace(/ This is a model-generated explanation from uploaded historical data, not a confirmed real-world event\./g, "")
    .trim();
}

function cleanDisplayText(value) {
  return String(value || "")
    .replace(/Hybrid score above threshold/g, "Above alert threshold")
    .replace(/Top SHAP-style contribution/g, "Primary anomaly driver")
    .replace(/SHAP-style/g, "Driver")
    .replace(/autoencoder reconstruction/gi, "reconstruction")
    .replace(/Isolation Forest/gi, "outlier model")
    .trim();
}

function domainFromEvent(event) {
  const driver = event?.drivers?.[0]?.feature || event?.event_type || "Urban";
  const text = driver.toLowerCase();
  if (text.includes("traffic")) return "Traffic";
  if (text.includes("aqi") || text.includes("air")) return "AQI";
  if (text.includes("electricity") || text.includes("energy")) return "Electricity";
  if (text.includes("temperature")) return "Temperature";
  if (text.includes("humidity")) return "Humidity";
  return "Urban";
}

function inDateRange(iso) {
  const date = iso.slice(0, 10);
  return (!startDate.value || date >= startDate.value) && (!endDate.value || date <= endDate.value);
}

function filteredEvents() {
  const domain = domainFilter.value;
  const severity = severityFilter.value;
  return (anomalyPayload?.all_events || []).filter((event) => {
    const eventDomain = domainFromEvent(event);
    return inDateRange(event.timestamp) && (domain === "All" || eventDomain === domain) && (severity === "All" || event.severity === severity);
  });
}

function filteredTopDays() {
  const severity = severityFilter.value;
  const domain = domainFilter.value;
  return (anomalyPayload?.top_anomalous_days || []).filter((row) => {
    return inDateRange(`${row.date}T00:00:00`) && (severity === "All" || row.severity === severity) && (domain === "All" || row.category === domain);
  });
}

function renderSummary(cards) {
  summaryCards.innerHTML = "";
  cards.slice(0, 4).forEach((card) => {
    const item = document.createElement("article");
    item.className = "summary-card";
    item.innerHTML = `<span>${cleanDisplayText(card.label)}</span><strong>${card.value}</strong><span>${cleanDisplayText(card.detail)}</span>`;
    summaryCards.appendChild(item);
  });
}

function renderTopDays(rows) {
  topDaysBody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${formatDate(`${row.date}T00:00:00`)}</td>
      <td>${formatScore(row.score)}</td>
      <td><span class="severity ${severityClass(row.severity)}">${row.severity}</span></td>
      <td>${row.category}</td>
    `;
    topDaysBody.appendChild(tr);
  });
}

function renderEvents(events) {
  eventPanel.innerHTML = "";
  events.slice(0, 6).forEach((event) => {
    const card = document.createElement("div");
    card.className = "event-card";
    const drivers = (event.drivers || [])
      .slice(0, 3)
      .map((driver) => {
        const width = Math.min(100, Math.max(8, Number(driver.contribution || 0) * 20));
        return `<div class="driver-row"><span>${cleanFeatureName(driver.feature)}</span><strong>${Number(driver.contribution || 0).toFixed(1)}</strong><div class="driver-bar"><span style="width:${width}%"></span></div></div>`;
      })
      .join("");
    card.innerHTML = `
      <span class="severity ${severityClass(event.severity)}">${event.severity} · ${formatScore(event.anomaly_score)}</span>
      <h3>${event.event_type}</h3>
      <p class="helper">${formatDate(event.timestamp)}</p>
      <p>${cleanDescription(event.description)}</p>
      ${drivers}
    `;
    eventPanel.appendChild(card);
  });
}

function renderHeatmap(rows) {
  heatmap.innerHTML = "";
  rows.forEach((row) => {
    const scoreLevel = row.score >= 0.9 ? 4 : row.score >= 0.7 ? 3 : row.score >= 0.5 ? 2 : row.score >= 0.3 ? 1 : 0;
    const cell = document.createElement("span");
    cell.className = `day-cell score-${scoreLevel}`;
    cell.title = `${formatDate(`${row.date}T00:00:00`)} · ${row.severity} · ${formatScore(row.score)}`;
    heatmap.appendChild(cell);
  });
}

function renderMonthly(rows) {
  monthlyStats.innerHTML = "";
  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "month-item";
    item.innerHTML = `
      <strong>${row.month}</strong>
      <p class="helper">${row.anomalies} anomalies · max score ${formatScore(row.max_score)}</p>
      <div class="month-bar"><span style="width:${Math.max(4, row.max_score * 100)}%"></span></div>
    `;
    monthlyStats.appendChild(item);
  });
}

function renderInsights(lines) {
  insightsList.innerHTML = "";
  lines.forEach((line) => {
    const li = document.createElement("li");
    li.textContent = cleanDisplayText(line);
    insightsList.appendChild(li);
  });
}

function renderRecommendations(rows) {
  recommendations.innerHTML = "";
  rows.slice(0, 4).forEach((row) => {
    const item = document.createElement("div");
    item.className = "recommendation-card";
    item.innerHTML = `
      <span class="severity ${severityClass(row.priority)}">${row.priority}</span>
      <h3>${cleanDisplayText(row.title)}</h3>
      <p class="helper">${cleanDisplayText(row.reason)}</p>
      <ul>${row.actions.map((action) => `<li>${cleanDisplayText(action)}</li>`).join("")}</ul>
    `;
    recommendations.appendChild(item);
  });
}

function renderAll() {
  const events = filteredEvents();
  renderTopDays(filteredTopDays());
  renderEvents(events);
  renderHeatmap((anomalyPayload.heatmap || []).filter((row) => inDateRange(`${row.date}T00:00:00`)));
  renderMonthly(anomalyPayload.monthly_stats || []);
  renderInsights((anomalyPayload.dynamic_insights || []).slice(0, 5));
  renderRecommendations(anomalyPayload.recommendations || []);
}

function setError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

async function loadDashboard() {
  try {
    const response = await fetch("/api/anomalies", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Unable to load anomaly dashboard");
    anomalyPayload = payload;
    dataMode.textContent = "Historical CSV";
    renderSummary(payload.summary_cards || []);
    renderAll();
  } catch (error) {
    setError(error.message);
  }
}

[startDate, endDate, domainFilter, severityFilter].forEach((control) => {
  control.addEventListener("change", () => {
    if (anomalyPayload) renderAll();
  });
});

resetFilters.addEventListener("click", () => {
  startDate.value = "";
  endDate.value = "";
  domainFilter.value = "All";
  severityFilter.value = "All";
  if (anomalyPayload) renderAll();
});

loadDashboard();
