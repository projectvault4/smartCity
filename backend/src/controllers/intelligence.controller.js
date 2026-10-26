const asyncHandler = require('../utils/asyncHandler');
const intelligence = require('../services/intelligence.service');
const decisionEngine = require('../services/decisionEngine.service');
const briefingService = require('../services/briefing.service');
const liveCheck = require('../services/liveCheck.service');
const aiSummary = require('../services/aiSummary.service');

const KNOWN_PROFILES = new Set(['general', 'commuter', 'outdoor_worker', 'respiratory', 'elderly', 'child']);

const parseProfiles = (value) => {
  if (!value) return null;
  const list = String(value).split(',').map((p) => p.trim()).filter((p) => KNOWN_PROFILES.has(p));
  return list.length ? Array.from(new Set(['general', ...list])) : null;
};

const send = (fn) => asyncHandler(async (req, res) => {
  res.set('Cache-Control', 'public, max-age=300');
  res.status(200).json({ data: await fn(req) });
});

module.exports = {
  manifest: send(() => intelligence.getManifest()),
  proof: send(() => intelligence.getProof()),
  benchmark: send(() => intelligence.getBenchmark()),
  conformal: send(() => intelligence.getConformal()),
  explain: send(() => intelligence.getShapSummary()),
  whatIf: send(() => intelligence.getWhatIf()),
  anomalyEval: send(() => intelligence.getAnomalyEval()),
  replay: send((req) => intelligence.getReplay({ offset: req.query.offset, limit: req.query.limit })),
  policy: send(() => intelligence.getPolicy()),

  decision: asyncHandler(async (req, res) => {
    res.set('Cache-Control', 'no-store');
    const data = decisionEngine.getDecision({
      city: req.query.city,
      hours: req.query.hours ? Number(req.query.hours) : 12,
      profiles: parseProfiles(req.query.profiles)
    });
    res.status(200).json({ data });
  }),

  briefing: asyncHandler(async (req, res) => {
    res.set('Cache-Control', 'no-store');
    const data = briefingService.buildBriefing({
      city: req.query.city,
      hours: req.query.hours ? Number(req.query.hours) : 12,
      profiles: parseProfiles(req.query.profiles) || ['general'],
      lang: req.query.lang === 'kn' ? 'kn' : 'en'
    });
    res.status(200).json({ data });
  }),

  summary: asyncHandler(async (req, res) => {
    res.set('Cache-Control', 'no-store');
    if (!aiSummary.allow(req.ip)) {
      res.status(429).json({ error: { message: 'Too many AI requests, try again in a few minutes.' } });
      return;
    }
    const data = await aiSummary.summarize({
      profiles: parseProfiles(req.query.profiles) || ['general'],
      lang: req.query.lang === 'kn' ? 'kn' : 'en'
    });
    res.status(200).json({ data });
  }),

  ai: asyncHandler(async (req, res) => {
    if (!aiSummary.allow(req.ip)) {
      res.status(429).json({ error: { message: 'Too many AI requests, try again in a few minutes.' } });
      return;
    }
    const { system, prompt, temperature, maxOutputTokens } = req.body || {};
    if (!prompt || typeof prompt !== 'string') {
      res.status(400).json({ error: { message: 'prompt is required' } });
      return;
    }
    const text = await aiSummary.callGemini({ system, prompt, temperature, maxOutputTokens });
    res.status(200).json({ data: { text } });
  }),

  liveCheck: asyncHandler(async (req, res) => {
    res.set('Cache-Control', 'no-store');
    const data = await liveCheck.getLiveCheck({ refresh: req.query.refresh !== 'false' });
    res.status(200).json({ data });
  })
};
