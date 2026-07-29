const express = require('express');

const riskController = require('../controllers/risk.controller');
const { validateRiskAssessment } = require('../validators/risk.validator');

const router = express.Router();

router.post('/assess', validateRiskAssessment, riskController.assessRisk);

module.exports = router;
