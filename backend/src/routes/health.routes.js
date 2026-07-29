const express = require('express');
const { getHealth, getMetrics, getReadiness } = require('../controllers/health.controller');

const router = express.Router();

router.get('/', getHealth);
router.get('/metrics', getMetrics);
router.get('/ready', getReadiness);

module.exports = router;
