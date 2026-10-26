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

module.exports = { callGemini, summarize, allow };
