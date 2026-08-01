const anomalyModelService = require('./anomalyModel.service');

const toNumber = (value) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

const pearson = (a, b) => {
  const n = Math.min(a.length, b.length);
  if (n < 2) return 0;
  const mA = a.reduce((s, v) => s + v, 0) / n;
  const mB = b.reduce((s, v) => s + v, 0) / n;
  let num = 0;
  let dA = 0;
  let dB = 0;
  for (let i = 0; i < n; i += 1) {
    num += (a[i] - mA) * (b[i] - mB);
    dA += (a[i] - mA) ** 2;
    dB += (b[i] - mB) ** 2;
  }
  const den = Math.sqrt(dA * dB);
  return den === 0 ? 0 : num / den;
};

const crossCorrelationLag = (signal, reference, maxLag) => {
  let best = { lag: 0, corr: -Infinity };
  for (let lag = -maxLag; lag <= maxLag; lag += 1) {
    const shifted = [];
    const aligned = [];
    for (let i = 0; i < reference.length; i += 1) {
      const j = i + lag;
      if (j >= 0 && j < signal.length) {
        shifted.push(signal[j]);
        aligned.push(reference[i]);
      }
    }
    const corr = pearson(shifted, aligned);
    if (corr > best.corr) {
      best = { lag, corr };
    }
  }
  return best;
};

const normalize = (values) => {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  return values.map((v) => Math.round(((v - min) / span) * 100));
};

const getMultivariateAnalysis = ({ windowHours = 720 } = {}) => {
  const rows = anomalyModelService.readTimelineRows();
  const recent = rows.slice(-windowHours);

  const traffic = recent.map((r) => toNumber(r.traffic_flow) || 0);
  const aqi = recent.map((r) => toNumber(r.aqi) || 0);
  const energy = recent.map((r) => toNumber(r.electricity_demand) || 0);
  const temperature = recent.map((r) => toNumber(r.temperature) || 0);

  const phaseLag = crossCorrelationLag(traffic, aqi, 12);
  const syncFactor = pearson(traffic, energy);
  const coherence = pearson(aqi, traffic);
  const tempEnergyCorr = pearson(temperature, energy);
  const aqiEnergyCorr = pearson(aqi, energy);

  const last24 = rows.slice(-24);
  const t24 = last24.map((r) => toNumber(r.traffic_flow) || 0);
  const a24 = last24.map((r) => toNumber(r.aqi) || 0);
  const e24 = last24.map((r) => toNumber(r.electricity_demand) || 0);

  const nT = normalize(t24);
  const nA = normalize(a24);
  const nE = normalize(e24);

  const series = last24.map((r, i) => ({
    time: r.timestamp.slice(11, 16),
    traffic: nT[i],
    aqi: nA[i],
    energy: nE[i],
    trafficRaw: Math.round(t24[i]),
    aqiRaw: Math.round(a24[i]),
    energyRaw: Math.round(e24[i])
  }));

  return {
    source: 'trained_model_multivariate_outputs',
    window: {
      hours: windowHours,
      from: recent[0].timestamp,
      to: recent[recent.length - 1].timestamp
    },
    series,
    stats: {
      phaseLagHours: Math.abs(phaseLag.lag) <= 1 ? 0 : Math.abs(phaseLag.lag),
      phaseLagDirection: phaseLag.lag < 0 ? 'aqi_leads_traffic' : 'traffic_leads_aqi',
      phaseLagCorr: Math.round(phaseLag.corr * 100) / 100,
      syncFactor: Math.round(syncFactor * 100) / 100,
      coherence: Math.round(coherence * 100) / 100,
      tempEnergyCorr: Math.round(tempEnergyCorr * 100) / 100,
      aqiEnergyCorr: Math.round(aqiEnergyCorr * 100) / 100
    }
  };
};

module.exports = {
  getMultivariateAnalysis
};
