const express = require('express');

const modelController = require('../controllers/model.controller');

const router = express.Router();

router.get('/conditions', modelController.getModelConditions);
router.get('/forecast', modelController.getModelForecast);
router.get('/forecast-yearly', modelController.getModelYearlyForecast);
router.get('/anomalies', modelController.getModelAnomalies);
router.get('/multivariate', modelController.getModelMultivariate);

module.exports = router;
