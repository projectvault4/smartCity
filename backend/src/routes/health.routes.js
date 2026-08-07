const express = require('express');
const { getHealth, getMetrics, getReadiness } = require('../controllers/health.controller');

const router = express.Router();

router.get('/', getHealth);
router.get('/metrics', getMetrics);
router.get('/ready', getReadiness);

// Temporary debug endpoint
router.get('/time', (req, res) => {
  res.json({
    serverTime: new Date().toString(),
    isoTime: new Date().toISOString(),
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    offsetMinutes: new Date().getTimezoneOffset()
  });
});
router.get('/chatgpt-test-123', (req, res) => {
  res.send('HELLO FROM RENDER');
});
module.exports = router;