const express = require('express');

const controller = require('../controllers/intelligence.controller');

const router = express.Router();

// Read-only evidence artifacts (built offline by build_intelligence.py).
router.get('/manifest', controller.manifest);
router.get('/proof', controller.proof);
router.get('/benchmark', controller.benchmark);
router.get('/conformal', controller.conformal);
router.get('/explain', controller.explain);
router.get('/whatif', controller.whatIf);
router.get('/anomaly-eval', controller.anomalyEval);
router.get('/replay', controller.replay);
router.get('/policy', controller.policy);

// Live decision layer.
router.get('/decision', controller.decision);
router.get('/briefing', controller.briefing);
router.get('/summary', controller.summary);
router.get('/live-check', controller.liveCheck);

// Server-side Gemini proxy (rate limited; key stays on the server).
router.post('/ai', controller.ai);
// Evidence-grounded Q&A (works without a Gemini key).
router.post('/ask', controller.ask);

module.exports = router;
