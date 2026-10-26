// One decision pipeline for every channel (dashboard, citizen page, SMS,
// in-app, voice, AI summary):
//
//   forecast (live 2026 outlook)
//     + uncertainty  (residuals of a long-range outlook, backtested on held-out 2025 data)
//     -> exceedance probability per event (policy.json thresholds, joint across targets)
//     -> alert when probability >= cutoff (cutoff tuned on the validation window)
//     -> one explanation object (SHAP drivers at that hour, interval, vs-typical)
//     -> per-profile message + action
//
// The thresholds, cutoffs and wording are the same ones scored in the
// historical replay (outputs/intelligence/replay.json).

const config = require('../config/env');
const intelligence = require('./intelligence.service');
const modelForecast = require('./modelForecast.service');
const liveState = require('./liveState');
const { cityParts, formatCityHour } = require('../utils/cityTime');

const TARGET_FIELDS = {
  traffic_flow: 'trafficFlow',
  aqi: 'aqi',
  temperature: 'temperature',
  electricity_demand: 'electricityDemand'
};
const NON_NEGATIVE = new Set(['traffic_flow', 'aqi', 'electricity_demand']);
const SEVERITY_TO_DB = { critical: 'critical', warning: 'warning', advisory: 'medium' };

const GROUP_TO_PROFILE = {
  resp: 'respiratory',
  respiratory: 'respiratory',
  elder: 'elderly',
  elderly: 'elderly',
  child: 'child',
  children: 'child',
  worker: 'outdoor_worker',
  outdoor_worker: 'outdoor_worker',
  commuter: 'commuter',
  general: 'general'
};
const HEALTH_TO_PROFILE = {
  asthma_copd: 'respiratory',
  heart: 'respiratory',
  works_outdoors: 'outdoor_worker'
};
const TAG_TO_PROFILE = {
  Respiratory: 'respiratory',
  Heart: 'respiratory',
  Elder: 'elderly',
  Child: 'child',
  Worker: 'outdoor_worker',
  Commuter: 'commuter'
};

// CPCB National AQI bands.
const cpcbCategory = (aqi) => {
  if (aqi === null || aqi === undefined || !Number.isFinite(Number(aqi))) return 'Unknown';
  const v = Number(aqi);
  if (v <= 50) return 'Good';
  if (v <= 100) return 'Satisfactory';
  if (v <= 200) return 'Moderate';
  if (v <= 300) return 'Poor';
  if (v <= 400) return 'Very Poor';
  return 'Severe';
};

const profilesForUser = (user = {}) => {
  const profiles = new Set(['general']);
  const add = (key, map) => {
    const mapped = map[String(key)] || map[String(key).toLowerCase()];
    if (mapped) profiles.add(mapped);
  };
  (user.groups || user.riskGroups || []).forEach((g) => add(typeof g === 'string' ? g : g?.group_key, GROUP_TO_PROFILE));
  (user.preferences?.health_conditions || []).forEach((h) => add(h, HEALTH_TO_PROFILE));
  (user.tags || []).forEach((t) => add(t, TAG_TO_PROFILE));
  const age = Number(user.age);
  if (Number.isFinite(age) && age > 60) profiles.add('elderly');
  if (Number.isFinite(age) && age > 0 && age < 12) profiles.add('child');
  return Array.from(profiles);
};

const quantile = (sorted, q) => {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
};

const fmtValue = (target, v) => {
  if (v === null || v === undefined || !Number.isFinite(v)) return '-';
  if (target === 'traffic_flow' || target === 'electricity_demand') return Math.round(v).toLocaleString('en-IN');
  if (target === 'temperature') return v.toFixed(1);
  return String(Math.round(v));
};

const fillTemplate = (template, values) => template.replace(/\{(\w+)\}/g, (_, key) => (values[key] ?? '')).replace(/\s{2,}/g, ' ').trim();

const actionFor = (policy, eventId, profile) => {
  const actions = policy.actions[eventId] || {};
  return actions[profile] || actions.general || Object.values(actions)[0] || '';
};

// Residual vectors observed at the same hour of day (+/- 1 h), so the spread
// reflects how uncertain THIS hour is (rush hours are less certain than 3 AM).
const residualsForHour = (unc, hour) => {
  const near = new Set([(hour + 23) % 24, hour, (hour + 1) % 24]);
  const rows = unc.residuals.filter((_, i) => near.has(unc.hour_of_day[i]));
  return rows.length >= 20 ? rows : unc.residuals;
};

const eventCutoffs = (policy) => {
  try {
    const replay = intelligence.loadArtifact('replay.json');
    return Object.fromEntries(Object.keys(policy.events).map((id) => [id, replay.events?.[id]?.cutoff ?? policy.default_probability_cutoff]));
  } catch {
    return Object.fromEntries(Object.keys(policy.events).map((id) => [id, policy.default_probability_cutoff]));
  }
};

const typicalDrivers = (shap, target, hour) => (
  shap?.targets?.[target]?.typical_drivers_by_hour?.[String(hour)] || []
);

const buildHours = ({ forecast, unc, policy, shap, liveBias = {} }) => {
  const targets = unc.targets;
  return forecast.map((rawRow) => {
    // Live correction: when recent observed readings show the outlook running
    // consistently high or low (AQI, temperature), shift it by that bias.
    const row = { ...rawRow };
    Object.entries(liveBias).forEach(([target, { bias }]) => {
      const field = TARGET_FIELDS[target];
      if (row[field] !== null && row[field] !== undefined) {
        row[field] = Math.max(NON_NEGATIVE.has(target) ? 0 : -Infinity, row[field] + bias);
      }
    });
    const ms = new Date(row.timestamp).getTime();
    const { hour } = cityParts(ms);
    const residuals = residualsForHour(unc, hour);
    const values = {};
    targets.forEach((target, ti) => {
      const value = row[TARGET_FIELDS[target]];
      if (value === null || value === undefined) return;
      const res = residuals.map((r) => r[ti]).sort((a, b) => a - b);
      let low = value + quantile(res, (1 - policy.interval_coverage) / 2);
      let high = value + quantile(res, 1 - (1 - policy.interval_coverage) / 2);
      if (NON_NEGATIVE.has(target)) {
        low = Math.max(0, low);
        high = Math.max(0, high);
      }
      const typical = unc.typical_by_hour?.[target]?.[String(hour)] ?? null;
      values[target] = {
        value,
        outlook: rawRow[TARGET_FIELDS[target]],
        liveAdjustment: liveBias[target]?.bias ?? 0,
        low,
        high,
        typical,
        vsTypicalPct: typical ? ((value - typical) / typical) * 100 : null,
        drivers: typicalDrivers(shap, target, hour)
      };
    });
    const probabilities = {};
    Object.entries(policy.events).forEach(([id, ev]) => {
      let hits = 0;
      residuals.forEach((r) => {
        const ok = ev.conditions.every((c) => {
          const ti = targets.indexOf(c.target);
          const v = row[TARGET_FIELDS[c.target]];
          return ti >= 0 && v !== null && v + r[ti] > c.threshold;
        });
        if (ok) hits += 1;
      });
      probabilities[id] = residuals.length ? hits / residuals.length : 0;
    });
    if (values.aqi) values.aqi.category = cpcbCategory(values.aqi.value);
    return { timestamp: row.timestamp, hour, values, probabilities };
  });
};

const buildAlerts = ({ hours, policy, cutoffs }) => {
  const alerts = [];
  Object.entries(policy.events).forEach(([id, ev]) => {
    const cutoff = cutoffs[id];
    const active = hours.filter((h) => h.probabilities[id] >= cutoff);
    if (!active.length) return;
    const peak = active.reduce((best, h) => (h.probabilities[id] > best.probabilities[id] ? h : best), active[0]);
    const first = active[0];
    const lead = ev.conditions[0].target;
    const v = first.values[lead];
    const drivers = v?.drivers || [];
    const driverText = drivers.length
      ? drivers[0].label + (drivers[1] ? `, then ${drivers[1].label.charAt(0).toLowerCase()}${drivers[1].label.slice(1)}` : '')
      : 'recent pattern';
    const firstMs = new Date(first.timestamp).getTime();
    const units = policy.units || {};
    const valueText = ev.conditions
      .map((c) => `${fmtValue(c.target, first.values[c.target]?.value)} ${units[c.target] || ''}`.trim())
      .join(' and ');
    const base = {
      label: ev.label,
      time: formatCityHour(firstMs),
      probability: Math.round(first.probabilities[id] * 100),
      value: valueText,
      low: fmtValue(lead, v?.low),
      high: `${fmtValue(lead, v?.high)} ${units[lead] || ''}`.trim(),
      driver: driverText
    };
    const messages = {};
    const sms = {};
    ev.audience.forEach((profile) => {
      const action = actionFor(policy, id, profile);
      messages[profile] = fillTemplate(policy.message_template, { ...base, action });
      sms[profile] = fillTemplate(policy.sms_template, { ...base, action });
    });
    alerts.push({
      event: id,
      label: ev.label,
      severity: ev.severity,
      cutoff,
      startsAt: first.timestamp,
      endsAt: active[active.length - 1].timestamp,
      activeHours: active.length,
      peakAt: peak.timestamp,
      probability: first.probabilities[id],
      peakProbability: peak.probabilities[id],
      leadTarget: lead,
      forecast: v ? { value: v.value, low: v.low, high: v.high, typical: v.typical, vsTypicalPct: v.vsTypicalPct } : null,
      drivers,
      audience: ev.audience,
      messages,
      sms
    });
  });
  const rank = { critical: 3, warning: 2, advisory: 1 };
  return alerts.sort((a, b) => (rank[b.severity] - rank[a.severity]) || (b.probability - a.probability));
};

const getDecision = ({ city = config.modelForecast.defaultCity, hours = 12, profiles } = {}) => {
  const policy = intelligence.getPolicy();
  const unc = intelligence.getOutlookUncertainty();
  let shap = null;
  try {
    shap = intelligence.getShapSummary();
  } catch {
    shap = null;
  }
  const steps = Math.min(48, Math.max(1, Number(hours) || 12));
  const forecast = modelForecast.getForecastSeries({ city, steps });
  const liveBias = liveState.getLiveBias();
  const hourRows = buildHours({ forecast, unc, policy, shap, liveBias });
  const cutoffs = eventCutoffs(policy);
  let alerts = buildAlerts({ hours: hourRows, policy, cutoffs });
  const wanted = profiles && profiles.length ? new Set(profiles) : null;
  if (wanted) {
    alerts = alerts
      .filter((a) => a.audience.some((p) => wanted.has(p)))
      .map((a) => {
        const profile = a.audience.find((p) => wanted.has(p) && p !== 'general') || a.audience.find((p) => wanted.has(p));
        return { ...a, profile, message: a.messages[profile], smsMessage: a.sms[profile] };
      });
  }
  const mix = {};
  Object.entries(shap?.targets || {}).forEach(([t, v]) => { mix[t] = v.final_mix; });
  return {
    generatedAt: new Date().toISOString(),
    city,
    horizonHours: steps,
    profiles: profiles || null,
    source: 'trained_model_2026_outlook',
    uncertainty: {
      coverage: policy.interval_coverage,
      basis: unc.method,
      note: 'Ranges and probabilities come from how far a long-range outlook was off on held-out 2025 data at the same hour of day.'
    },
    explanationBasis: shap?.method ? `${shap.method}; drivers shown are typical for that hour on held-out data` : null,
    liveAdjustments: liveBias,
    forecastMix: mix,
    cutoffs,
    hours: hourRows,
    alerts,
    quiet: alerts.length === 0
  };
};

// 90% interval for one outlook value at a given city hour (used by the live check).
const outlookInterval = (target, value, hour) => {
  const policy = intelligence.getPolicy();
  const unc = intelligence.getOutlookUncertainty();
  const ti = unc.targets.indexOf(target);
  if (ti < 0 || value === null || value === undefined) return null;
  const res = residualsForHour(unc, hour).map((r) => r[ti]).sort((a, b) => a - b);
  const tail = (1 - policy.interval_coverage) / 2;
  let low = value + quantile(res, tail);
  const high = value + quantile(res, 1 - tail);
  if (NON_NEGATIVE.has(target)) low = Math.max(0, low);
  return { low, high, expectedMae: unc.per_target?.[target]?.mae ?? null };
};

module.exports = {
  getDecision,
  profilesForUser,
  cpcbCategory,
  outlookInterval,
  SEVERITY_TO_DB
};
