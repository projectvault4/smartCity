const cron = require('node-cron');

const config = require('../config/env');
const advisoryBatchJob = require('../services/advisoryBatchJob.service');
const logger = require('../utils/logger');

let task;
let running = false;

const runOnce = async () => {
  if (running) {
    logger.warn('Advisory cron skipped because a previous run is still active');
    return null;
  }

  running = true;

  try {
    const result = await advisoryBatchJob.processAllUsers();
    logger.info('Advisory cron completed', { status: result.status, executionLogId: result.id });
    return result;
  } catch (error) {
    logger.error('Advisory cron failed', { message: error.message, stack: error.stack });
    throw error;
  } finally {
    running = false;
  }
};

const start = () => {
  if (task) {
    return task;
  }

  if (!cron.validate(config.jobs.advisoryCronSchedule)) {
    throw new Error(`Invalid advisory cron schedule: ${config.jobs.advisoryCronSchedule}`);
  }

  task = cron.schedule(config.jobs.advisoryCronSchedule, runOnce, {
    scheduled: true
  });

  logger.info('Advisory cron scheduled', { schedule: config.jobs.advisoryCronSchedule });
  return task;
};

const stop = () => {
  if (task) {
    task.stop();
    task = null;
  }
};

module.exports = {
  start,
  stop,
  runOnce
};
