// Server-side Gemini proxy plus an evidence-grounded summary.
//
// The browser no longer holds a Gemini key. When GEMINI_API_KEY is not set the
// summary falls back to the deterministic evidence text, so the feature still
// works (and never invents numbers).

const axios = require('axios');

const config = require('../config/env');
const briefingService = require('./briefing.service');

const GEMINI_URL = (model) => `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;

const callGemini = async ({ system, prompt, temperature = 0.4, maxOutputTokens = 900 }) => {
  if (!config.ai.geminiApiKey) {
    const error = new Error('AI summaries are not configured on the server (GEMINI_API_KEY missing).');
    error.statusCode = 503;
    throw error;
  }
  const body = {
    systemInstruction: { parts: [{ text: String(system || '').slice(0, 4000) }] },
    contents: [{ role: 'user', parts: [{ text: String(prompt || '').slice(0, config.ai.maxPromptChars) }] }],
    generationConfig: { temperature, maxOutputTokens: Math.min(2048, Number(maxOutputTokens) || 900), thinkingConfig: { thinkingBudget: 0 } }
  };
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const response = await axios.post(GEMINI_URL(config.ai.geminiModel), body, {
        headers: { 'Content-Type': 'application/json', 'x-goog-api-key': config.ai.geminiApiKey },
        timeout: 20000
      });
      const parts = response.data?.candidates?.[0]?.content?.parts || [];
      return parts.map((p) => p.text || '').join('');
    } catch (error) {
      const status = error.response?.status;
      if (attempt === 0 && (status === 429 || status >= 500)) {
        await new Promise((r) => setTimeout(r, 1500));
        continue;
      }
      const wrapped = new Error(`Gemini request failed (${status || 'network'})`);
      wrapped.statusCode = 502;
      throw wrapped;
    }
  }
  return '';
};

const summarize = async ({ profiles, lang = 'en' } = {}) => {
  const briefing = briefingService.buildBriefing({ profiles, lang: 'en' });
  const evidence = briefing.english.join(' ');
  if (!config.ai.geminiApiKey || lang === 'kn') {
    return { source: 'evidence_template', text: (lang === 'kn' ? briefing.kannada : briefing.english).join(' '), evidence };
  }
  try {
    const text = await callGemini({
      system: 'You rewrite a city forecast for a regular Bengaluru citizen. Use ONLY facts and numbers in the evidence. Do not add causes, numbers or advice that are not in the evidence. 4-5 short sentences, no jargon.',
      prompt: `Evidence:\n${evidence}`,
      temperature: 0.3,
      maxOutputTokens: 400
    });
    return { source: 'gemini_grounded', text: text.trim() || evidence, evidence };
  } catch (error) {
    return { source: 'evidence_template', text: evidence, evidence, aiError: error.message };
  }
};

// ── Query Urban Intelligence ────────────────────────────────────────────────
// Answers a free-text question from the model's real evidence: the forecast
// and 90% range for the relevant signal, SHAP drivers, alerts, and measured
// cross-domain links. Gemini (when configured) only rephrases that evidence;
// the chart values and drivers always come from the model, never from the LLM.

const decisionEngine = require('./decisionEngine.service');
const intelligence = require('./intelligence.service');
const { formatCityHour } = require('../utils/cityTime');

const TOPICS = [
  { target: 'aqi', words: /\b(aqi|air|pollut|smog|breath|asthma|school|pm2|pm10|mask)/i },
  { target: 'electricity_demand', words: /\b(energy|electric|power|grid|demand|load|bescom|mw)/i },
  { target: 'temperature', words: /\b(temperat|heat|hot|cool|weather|warm|degree)/i },
  { target: 'traffic_flow', words: /\b(traffic|congest|commut|vehicle|road|jam|metro)/i }
];
const LABEL = { traffic_flow: 'Traffic', aqi: 'AQI', temperature: 'Temperature', electricity_demand: 'Electricity demand' };
const UNIT = { traffic_flow: 'vehicles/hr', aqi: '', temperature: '°C', electricity_demand: 'MW' };
const fmt = (t, v) => (v === null || v === undefined ? '-' : t === 'temperature' ? v.toFixed(1) : Math.round(v).toLocaleString('en-IN'));

const topicsFor = (question) => {
  const hits = TOPICS
    .map((t) => ({ target: t.target, at: question.search(t.words) }))
    .filter((t) => t.at >= 0)
    .sort((a, b) => a.at - b.at)
    .map((t) => t.target);
  return hits.length ? hits : ['traffic_flow'];
};

const buildEvidence = (question) => {
  const targets = topicsFor(question);
  const lead = targets[targets.length > 1 && /\b(on|affect|impact|effect|drive|cause)/i.test(question) ? 1 : 0];
  const school = /school|child|kid/i.test(question);
  const decision = decisionEngine.getDecision({ hours: 12, profiles: school ? ['child'] : undefined });
  const hours = decision.hours.slice(0, 6);
  const series = hours.map((h) => ({ time: formatCityHour(new Date(h.timestamp).getTime()), value: h.values[lead]?.value ?? null }));
  const first = hours[0]?.values[lead];
  const last = hours[hours.length - 1]?.values[lead];
  const peak = hours.reduce((best, h) => ((h.values[lead]?.value ?? -Infinity) > (best.values[lead]?.value ?? -Infinity) ? h : best), hours[0]);
  const unit = UNIT[lead] ? ` ${UNIT[lead]}` : '';
  const lines = [];

  const direction = last && first ? (last.value - first.value) / (Math.abs(first.value) || 1) : 0;
  const trend = Math.abs(direction) < 0.03 ? 'stays about level' : direction > 0 ? 'rises' : 'falls';
  lines.push(`${LABEL[lead]} ${trend} over the next 6 hours: about ${fmt(lead, first?.value)}${unit} at ${series[0]?.time}, ${fmt(lead, last?.value)}${unit} by ${series[series.length - 1]?.time}${lead === 'aqi' && first ? ` (CPCB ${first.category})` : ''}.`);
  if (peak && peak !== hours[0]) {
    const pv = peak.values[lead];
    lines.push(`The highest point is around ${formatCityHour(new Date(peak.timestamp).getTime())} at ${fmt(lead, pv.value)}${unit}, likely between ${fmt(lead, pv.low)} and ${fmt(lead, pv.high)}.`);
  } else if (first) {
    lines.push(`The likely range for the next hour is ${fmt(lead, first.low)} to ${fmt(lead, first.high)}${unit}.`);
  }
  if (first && first.vsTypicalPct !== null && Number.isFinite(first.vsTypicalPct)) {
    lines.push(Math.abs(first.vsTypicalPct) < 5
      ? 'That is close to a normal day at this hour.'
      : `That is ${Math.abs(Math.round(first.vsTypicalPct))}% ${first.vsTypicalPct > 0 ? 'above' : 'below'} a normal day at this hour.`);
  }
  const drivers = first?.drivers || [];
  const lower = (s) => {
    const t = s.replace(' (SARIMA)', '');
    return t.startsWith('AQI') ? t : t.charAt(0).toLowerCase() + t.slice(1);
  };
  if (drivers.length) {
    lines.push(`Main reason: ${lower(drivers[0].label)}${drivers[1] ? `, then ${lower(drivers[1].label)}` : ''}.`);
  }

  // Cross-domain question: report what the data actually shows.
  try {
    const whatIf = intelligence.getWhatIf();
    if (targets.length > 1) {
      const [a, b] = [targets[0], targets[1]];
      const assoc = whatIf.daily_associations.find((x) => (x.x === a && x.y === b) || (x.x === b && x.y === a));
      if (assoc) lines.push(assoc.sentence.replace('Observed daily data (Feb-Jun 2025): ', 'In observed Bengaluru data, '));
      if ((a === 'traffic_flow' && b === 'aqi') || (a === 'aqi' && b === 'traffic_flow')) {
        lines.push('Over the next hour the model barely moves AQI when traffic changes: cutting traffic 20% shifts the AQI forecast by less than 0.01 points.');
      }
    }
  } catch {
    // evidence optional
  }

  const relevantAlerts = decision.alerts.filter((al) => al.leadTarget === lead || al.event.startsWith(lead === 'aqi' ? 'air' : lead === 'temperature' ? 'heat' : 'x'));
  if (relevantAlerts.length) {
    const al = relevantAlerts[0];
    lines.push(al.message || al.messages[al.audience[0]]);
  } else {
    lines.push(`No ${LABEL[lead] === 'AQI' ? 'AQI' : LABEL[lead].toLowerCase()} alert is expected in the next 12 hours.`);
  }
  if (school && lead === 'aqi' && peak) {
    const pv = peak.values.aqi?.value ?? 0;
    lines.push(pv > 100
      ? `For school zones: AQI reaches the CPCB Moderate band (above 100) around ${formatCityHour(new Date(peak.timestamp).getTime())}, so schedule outdoor PE earlier and avoid roadside waiting for children then.`
      : 'For school zones: AQI stays below the CPCB Moderate band (100), so normal outdoor activity is fine for children.');
  }

  return {
    lead,
    text: lines.join(' '),
    forecast: series,
    shap: drivers.map((d) => ({ name: d.label.replace(' (SARIMA)', ''), val: Math.round(d.share * 1000) / 10 }))
  };
};

const ask = async ({ question }) => {
  const q = String(question || '').slice(0, 500);
  const evidence = buildEvidence(q);
  if (!config.ai.geminiApiKey) return { ...evidence, source: 'model_evidence' };
  try {
    const text = await callGemini({
      system: 'You answer a Bengaluru citizen\'s question about a city forecast. Use ONLY the facts and numbers in the evidence; do not add causes, numbers or advice that are not in it. If the evidence says a link is not supported, say so. 4-5 short sentences, plain words.',
      prompt: `Question: ${q}\n\nEvidence:\n${evidence.text}`,
      temperature: 0.3,
      maxOutputTokens: 400
    });
    return { ...evidence, text: text.trim() || evidence.text, source: 'gemini_grounded' };
  } catch (error) {
    return { ...evidence, source: 'model_evidence', aiError: error.message };
  }
};

// Tiny per-IP fixed-window limiter for the AI proxy (no extra dependency).
const hits = new Map();
const allow = (ip) => {
  const now = Date.now();
  const entry = hits.get(ip) || { start: now, count: 0 };
  if (now - entry.start > config.ai.rateLimitWindowMs) {
    entry.start = now;
    entry.count = 0;
  }
  entry.count += 1;
  hits.set(ip, entry);
  if (hits.size > 5000) hits.clear();
  return entry.count <= config.ai.rateLimitPerWindow;
};

module.exports = { callGemini, summarize, ask, allow };
