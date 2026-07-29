const advisoryGenerator = require('../services/advisoryGenerator.service');
const asyncHandler = require('../utils/asyncHandler');

const generateAdvisory = asyncHandler(async (req, res) => {
  const result = advisoryGenerator.generateForUser(req.body);

  res.status(200).json({ data: result });
});

const generateAdvisories = asyncHandler(async (req, res) => {
  const result = advisoryGenerator.generateForUsers(req.body);

  res.status(200).json({ data: result });
});

module.exports = {
  generateAdvisory,
  generateAdvisories
};
