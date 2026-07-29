const advisoryBatchJob = require('../services/advisoryBatchJob.service');
const asyncHandler = require('../utils/asyncHandler');

const runAdvisoryBatch = asyncHandler(async (req, res) => {
  const result = await advisoryBatchJob.processAllUsers({
    startedBy: req.body?.startedBy || 'manual',
    channels: req.body?.channels,
    sharedConditions: req.body?.sharedConditions
  });

  res.status(200).json({ data: result });
});

module.exports = {
  runAdvisoryBatch
};
