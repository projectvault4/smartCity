const modelForecastService = require('../services/modelForecast.service');
const asyncHandler = require('../utils/asyncHandler');

const getModelConditions = asyncHandler(async (req, res) => {
  const conditions = modelForecastService.getLatestModelConditions({
    city: req.query.city,
    stepAhead: req.query.stepAhead ? Number(req.query.stepAhead) : 1
  });

  res.status(200).json({ data: conditions });
});

module.exports = {
  getModelConditions
};
