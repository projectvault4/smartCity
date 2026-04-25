const comparisonTableBody = document.getElementById("comparisonTableBody");
const literatureTableBody = document.getElementById("literatureTableBody");
const comparisonMessage = document.getElementById("comparisonMessage");
const adaptiveSwitcherLabel = document.getElementById("adaptiveSwitcherLabel");
const streamingMetrics = document.getElementById("streamingMetrics");
const streamingModel = document.getElementById("streamingModel");
const bestPerTarget = document.getElementById("bestPerTarget");
const adaptiveSwitcher = document.getElementById("adaptiveSwitcher");
const comparisonPearsonPairs = document.getElementById("comparisonPearsonPairs");
const comparisonSpearmanPairs = document.getElementById("comparisonSpearmanPairs");
const comparisonGrangerLinks = document.getElementById("comparisonGrangerLinks");
const comparisonErrorBox = document.getElementById("comparisonErrorBox");
const refreshComparisonButton = document.getElementById("refreshComparisonButton");

function setComparisonError(message) {
  comparisonErrorBox.textContent = message;
  comparisonErrorBox.classList.remove("hidden");
}

function clearComparisonError() {
  comparisonErrorBox.textContent = "";
  comparisonErrorBox.classList.add("hidden");
}

function renderComparisonTable(rows) {
  comparisonTableBody.innerHTML = "";
  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="5">No comparison data available yet.</td>`;
    comparisonTableBody.appendChild(tr);
    return;
  }

  const bestUps = Math.max(...rows.map((row) => row.ups));
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    if (row.ups === bestUps) {
      tr.classList.add("best-row");
    }
    tr.innerHTML = `
      <td>${row.model}</td>
      <td>${row.mae.toFixed(4)}</td>
      <td>${row.rmse.toFixed(4)}</td>
      <td>${row.nrmse.toFixed(4)}</td>
      <td>${row.ups.toFixed(2)}</td>
    `;
    comparisonTableBody.appendChild(tr);
  });
}

function renderLiteratureTable(rows) {
  literatureTableBody.innerHTML = "";
  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="6">No literature reference models available yet.</td>`;
    literatureTableBody.appendChild(tr);
    return;
  }

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const source = row.source_url
      ? `<a href="${row.source_url}" target="_blank" rel="noreferrer">${row.source_name}</a>`
      : row.source_name || "Project implementation";
    tr.innerHTML = `
      <td>${row.model}</td>
      <td>${row.family}</td>
      <td>${row.year ?? ""}</td>
      <td>${row.efficiency ?? ""}</td>
      <td>${row.comparison_scope ?? ""}</td>
      <td>${source}</td>
    `;
    literatureTableBody.appendChild(tr);
  });
}

function renderInsightList(root, rows, formatter) {
  root.innerHTML = "";
  if (!rows.length) {
    const item = document.createElement("div");
    item.className = "insight-item";
    item.textContent = "No data available yet.";
    root.appendChild(item);
    return;
  }
  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "insight-item";
    item.innerHTML = formatter(row);
    root.appendChild(item);
  });
}

function renderMetricCards(root, rows) {
  root.innerHTML = "";
  if (!rows.length) {
    const item = document.createElement("div");
    item.className = "insight-item";
    item.textContent = "No streaming metrics available yet.";
    root.appendChild(item);
    return;
  }
  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "insight-item";
    item.innerHTML = `<strong>${row.metric}</strong><span>${row.value.toFixed(4)}</span>`;
    root.appendChild(item);
  });
}

function renderPairChips(root, rows, label) {
  root.innerHTML = "";
  if (!rows.length) {
    const item = document.createElement("div");
    item.className = "chip-item";
    item.textContent = `No ${label.toLowerCase()} results available yet.`;
    root.appendChild(item);
    return;
  }
  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "chip-item";
    item.innerHTML = `<strong>${row.left} <-> ${row.right}</strong><span>${label}: ${row.value.toFixed(3)}</span>`;
    root.appendChild(item);
  });
}

function renderGranger(rows) {
  comparisonGrangerLinks.innerHTML = "";
  if (!rows.length) {
    const item = document.createElement("div");
    item.className = "insight-item";
    item.textContent = "No Granger results available yet.";
    comparisonGrangerLinks.appendChild(item);
    return;
  }
  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "insight-item";
    const lagText = row.lag ? `best lag ${row.lag}` : "lag unavailable";
    const significance = row.significant ? "significant at 5%" : "not significant at 5%";
    item.innerHTML = `<strong>${row.cause} -> ${row.effect}</strong><span>${lagText}, p-value ${row.p_value}, ${significance}</span>`;
    comparisonGrangerLinks.appendChild(item);
  });
}

async function loadComparison() {
  refreshComparisonButton.disabled = true;
  refreshComparisonButton.textContent = "Refreshing...";
  clearComparisonError();

  try {
    const response = await fetch("/api/comparison", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Unable to load comparison");
    }

    comparisonMessage.textContent = payload.message;
    adaptiveSwitcherLabel.textContent = payload.adaptive_switcher_label
      ? `AdaptiveSwitcher chose: ${payload.adaptive_switcher_label}`
      : "";
    streamingModel.textContent = payload.streaming_model
      ? `Streaming evaluation uses: ${payload.streaming_model}`
      : "";
    renderComparisonTable(payload.models);
    renderLiteratureTable(payload.literature_models);
    renderMetricCards(streamingMetrics, payload.streaming_metrics);
    renderInsightList(bestPerTarget, payload.per_target_best, (row) => {
      const score = row.best_nrmse == null ? "NRMSE unavailable" : `Best NRMSE ${row.best_nrmse.toFixed(4)}`;
      return `<strong>${row.target}</strong><span>${row.best_model} (${score})</span>`;
    });
    renderInsightList(adaptiveSwitcher, payload.adaptive_switcher, (row) => {
      return `<strong>${row.target}</strong><span>Selected base model: ${row.model}</span>`;
    });
    renderPairChips(comparisonPearsonPairs, payload.analytics.pearson_top_pairs, "Pearson");
    renderPairChips(comparisonSpearmanPairs, payload.analytics.spearman_top_pairs, "Spearman");
    renderGranger(payload.analytics.granger_top_links);
  } catch (error) {
    setComparisonError(error.message);
  } finally {
    refreshComparisonButton.disabled = false;
    refreshComparisonButton.textContent = "Refresh Comparison";
  }
}

refreshComparisonButton.addEventListener("click", loadComparison);
loadComparison();
