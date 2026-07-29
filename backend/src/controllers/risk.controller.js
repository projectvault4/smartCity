const riskEngine = require('../services/riskEngine.service');
const asyncHandler = require('../utils/asyncHandler');

const assessRisk = asyncHandler(async (req, res) => {
  const assessment = riskEngine.calculateRisk(req.body);

  res.status(200).json({ data: assessment });
});

module.exports = {
  assessRisk
};
