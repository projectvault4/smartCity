const express = require('express');

const jobController = require('../controllers/job.controller');

const router = express.Router();

router.post('/advisories/run', jobController.runAdvisoryBatch);

module.exports = router;
