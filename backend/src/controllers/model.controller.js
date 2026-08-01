const modelForecastService = require('../services/modelForecast.service');
const anomalyModelService = require('../services/anomalyModel.service');
const multivariateModelService = require('../services/multivariateModel.service');
const asyncHandler = require('../utils/asyncHandler');

const getModelConditions = asyncHandler(async (req, res) => {
  const conditions = modelForecastService.getLatestModelConditions({
    city: req.query.city,
    stepAhead: req.query.stepAhead ? Number(req.query.stepAhead) : 1
  });

  res.status(200).json({ data: conditions });
});

const getModelForecast = asyncHandler(async (req, res) => {
  const steps = req.query.steps ? Number(req.query.steps) : 24;

  const forecast = modelForecastService.getForecastSeries({
    city: req.query.city,
    steps
  });

  res.status(200).json({ data: forecast });
});

const getModelAnomalies = asyncHandler(async (req, res) => {
  const dashboard = anomalyModelService.getAnomalyDashboard();

  res.status(200).json({ data: dashboard });
});

const getModelMultivariate = asyncHandler(async (req, res) => {
  const analysis = multivariateModelService.getMultivariateAnalysis({
    windowHours: req.query.window ? Number(req.query.window) : 720
  });

  res.status(200).json({ data: analysis });
});

module.exports = {
  getModelConditions,
  getModelForecast,
  getModelAnomalies,
  getModelMultivariate
};
