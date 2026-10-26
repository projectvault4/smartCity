// Evidence-based briefings for voice and the AI summary.
//
// Every sentence is built from the decision object, so the voice, the AI
// summary, the in-app alert and the SMS all say the same thing: what is
// different from a normal hour, why (SHAP drivers), how sure (interval /
// probability) and what to do for this person's profile.
//
// Kannada strings are template translations; have a native speaker review
// them before relying on them publicly.

const decisionEngine = require('./decisionEngine.service');
const { cityParts, formatCityHour } = require('../utils/cityTime');

const KN_EVENT = {
  traffic_congestion: 'ತೀವ್ರ ಸಂಚಾರ ದಟ್ಟಣೆ',
  grid_peak: 'ವಿದ್ಯುತ್ ಬೇಡಿಕೆಯ ಗರಿಷ್ಠ ಮಟ್ಟ',
  compound_stress: 'ಸಂಚಾರ ಮತ್ತು ವಿದ್ಯುತ್ ಬೇಡಿಕೆ ಎರಡೂ ಗರಿಷ್ಠ',
  air_quality_sensitive: 'ವಾಯು ಗುಣಮಟ್ಟ ಮಧ್ಯಮ',
  air_quality_poor: 'ವಾಯು ಗುಣಮಟ್ಟ ಕಳಪೆ',
  heat_stress: 'ಬಿಸಿಲಿನ ತಾಪ'
};
const KN_ACTION = {
  traffic_congestion: 'ಸಾಧ್ಯವಾದರೆ ಪ್ರಯಾಣದ ಸಮಯ ಬದಲಿಸಿ ಅಥವಾ ನಮ್ಮ ಮೆಟ್ರೋ ಬಳಸಿ.',
  grid_peak: 'ಭಾರೀ ವಿದ್ಯುತ್ ಉಪಕರಣಗಳನ್ನು ಈ ಸಮಯದ ಹೊರಗೆ ಬಳಸಿ.',
  compound_stress: 'ಈ ಸಮಯದಲ್ಲಿ ಪ್ರಯಾಣ ಮತ್ತು ಭಾರೀ ವಿದ್ಯುತ್ ಬಳಕೆ ಕಡಿಮೆ ಮಾಡಿ.',
  air_quality_sensitive: 'ಹೊರಾಂಗಣ ಶ್ರಮದ ಚಟುವಟಿಕೆ ಕಡಿಮೆ ಮಾಡಿ; ಅಗತ್ಯವಿದ್ದರೆ ಮಾಸ್ಕ್ ಧರಿಸಿ.',
  air_quality_poor: 'ಸಾಧ್ಯವಾದಷ್ಟು ಮನೆಯೊಳಗೆ ಇರಿ; ಹೊರಗೆ ಹೋದರೆ ಮಾಸ್ಕ್ ಧರಿಸಿ.',
  heat_stress: 'ಸಾಕಷ್ಟು ನೀರು ಕುಡಿಯಿರಿ ಮತ್ತು ಮಧ್ಯಾಹ್ನ ನೆರಳಿನಲ್ಲಿರಿ.'
};
const KN_AQI = {
  Good: 'ಉತ್ತಮ',
  Satisfactory: 'ತೃಪ್ತಿಕರ',
  Moderate: 'ಮಧ್ಯಮ',
  Poor: 'ಕಳಪೆ',
  'Very Poor': 'ಅತಿ ಕಳಪೆ',
  Severe: 'ತೀವ್ರ',
  Unknown: 'ಲಭ್ಯವಿಲ್ಲ'
};

const knTime = (ms) => {
  const { hour } = cityParts(ms);
  const h12 = hour % 12 || 12;
  let period = 'ರಾತ್ರಿ';
  if (hour >= 5 && hour < 12) period = 'ಬೆಳಿಗ್ಗೆ';
  else if (hour >= 12 && hour < 16) period = 'ಮಧ್ಯಾಹ್ನ';
  else if (hour >= 16 && hour < 20) period = 'ಸಂಜೆ';
  return `${period} ${h12} ಗಂಟೆಯ ಸುಮಾರಿಗೆ`;
};

const round = (v, step = 1) => Math.round(v / step) * step;
const n = (v) => Math.round(v).toLocaleString('en-IN');

const compareEn = (pct, typicalText) => {
  if (pct === null || pct === undefined || !Number.isFinite(pct)) return '';
  if (Math.abs(pct) < 5) return `, close to a normal ${typicalText}`;
  return `, ${Math.abs(Math.round(pct))}% ${pct > 0 ? 'above' : 'below'} a normal ${typicalText}`;
};
const compareKn = (pct) => {
  if (pct === null || pct === undefined || !Number.isFinite(pct)) return '';
  if (Math.abs(pct) < 5) return ' (ಸಾಮಾನ್ಯ ಮಟ್ಟ)';
  return ` (ಸಾಮಾನ್ಯಕ್ಕಿಂತ ${Math.abs(Math.round(pct))}% ${pct > 0 ? 'ಹೆಚ್ಚು' : 'ಕಡಿಮೆ'})`;
};

const buildBriefing = ({ profiles = ['general'], hours = 12, lang = 'en', city } = {}) => {
  const decision = decisionEngine.getDecision({ city, hours, profiles });
  const next = decision.hours[0];
  const ms = new Date(next.timestamp).getTime();
  const t = next.values;
  const timeEn = formatCityHour(ms);
  const alerts = decision.alerts;
  const trafficDriver = t.traffic_flow?.drivers?.[0]?.label;

  const en = [];
  en.push(alerts.length
    ? `${alerts.length} alert${alerts.length > 1 ? 's' : ''} for you in the next ${decision.horizonHours} hours.`
    : `No alerts for you in the next ${decision.horizonHours} hours.`);
  if (t.traffic_flow) {
    en.push(`Around ${timeEn}, traffic about ${n(round(t.traffic_flow.value, 100))} vehicles an hour${compareEn(t.traffic_flow.vsTypicalPct, `${timeEn} hour`)}; likely between ${n(round(t.traffic_flow.low, 100))} and ${n(round(t.traffic_flow.high, 100))}.`);
  }
  if (t.aqi) {
    const adj = t.aqi.liveAdjustment ? ` (adjusted ${Math.round(t.aqi.liveAdjustment) > 0 ? 'up' : 'down'} by ${Math.abs(Math.round(t.aqi.liveAdjustment))} from live readings)` : '';
    en.push(`Air quality index about ${Math.round(t.aqi.value)}, ${t.aqi.category} on the CPCB scale${adj}.`);
  }
  if (t.temperature) {
    en.push(`Temperature near ${t.temperature.value.toFixed(0)} degrees.`);
  }
  if (t.electricity_demand) {
    en.push(`Electricity demand about ${n(round(t.electricity_demand.value, 10))} megawatts${compareEn(t.electricity_demand.vsTypicalPct, 'hour')}.`);
  }
  if (trafficDriver) {
    en.push(`Main reason: ${trafficDriver.replace(' (SARIMA)', '')}.`);
  }
  alerts.forEach((a) => en.push(a.message || a.messages[a.audience[0]]));
  en.push('Ranges come from how far past forecasts were off at this hour.');

  const kn = [];
  kn.push(alerts.length
    ? `ಮುಂದಿನ ${decision.horizonHours} ಗಂಟೆಗಳಲ್ಲಿ ನಿಮಗೆ ${alerts.length} ಎಚ್ಚರಿಕೆ ಇದೆ.`
    : `ಮುಂದಿನ ${decision.horizonHours} ಗಂಟೆಗಳಲ್ಲಿ ನಿಮಗೆ ಯಾವುದೇ ಎಚ್ಚರಿಕೆ ಇಲ್ಲ.`);
  if (t.traffic_flow) {
    kn.push(`${knTime(ms)} ಸಂಚಾರ ಗಂಟೆಗೆ ಸುಮಾರು ${n(round(t.traffic_flow.value, 100))} ವಾಹನಗಳು${compareKn(t.traffic_flow.vsTypicalPct)}.`);
  }
  if (t.aqi) {
    kn.push(`ವಾಯು ಗುಣಮಟ್ಟ ಸೂಚ್ಯಂಕ ಸುಮಾರು ${Math.round(t.aqi.value)}, ${KN_AQI[t.aqi.category] || ''}.`);
  }
  if (t.temperature) {
    kn.push(`ತಾಪಮಾನ ಸುಮಾರು ${t.temperature.value.toFixed(0)} ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್.`);
  }
  if (t.electricity_demand) {
    kn.push(`ವಿದ್ಯುತ್ ಬೇಡಿಕೆ ಸುಮಾರು ${n(round(t.electricity_demand.value, 10))} ಮೆಗಾವ್ಯಾಟ್${compareKn(t.electricity_demand.vsTypicalPct)}.`);
  }
  alerts.forEach((a) => {
    const ams = new Date(a.startsAt).getTime();
    kn.push(`${knTime(ams)} ${KN_EVENT[a.event] || a.label} ಸಾಧ್ಯತೆ ${Math.round(a.probability * 100)}%. ${KN_ACTION[a.event] || ''}`.trim());
  });

  return {
    generatedAt: decision.generatedAt,
    forecastFor: next.timestamp,
    profiles: decision.profiles,
    lang,
    sentences: lang === 'kn' ? kn : en,
    english: en,
    kannada: kn,
    kannadaNote: 'Template translation; please have a native Kannada speaker review before public use.',
    alerts: alerts.map((a) => ({ event: a.event, label: a.label, severity: a.severity, probability: a.probability, startsAt: a.startsAt })),
    evidence: {
      next: next.values,
      uncertainty: decision.uncertainty,
      explanationBasis: decision.explanationBasis,
      forecastMix: decision.forecastMix
    }
  };
};

module.exports = { buildBriefing };
