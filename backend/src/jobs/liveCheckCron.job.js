const cron = require('node-cron');

const config = require('../config/env');
const liveCheck = require('../services/liveCheck.service');
const logger = require('../utils/logger');

let task;

const runOnce = async () => {
  try {
    // getLiveCheck records the reading and refreshes the drift/bias stats the
    // decision engine uses.
    const result = await liveCheck.getLiveCheck({ refresh: true });
    const row = result.latest || {};
    logger.info('Live check recorded', { observedHour: row.observedHour, storage: row.storage, status: result.status });
  } catch (error) {
    logger.warn('Live check failed', { message: error.message });
  }
};

const start = () => {
  if (task) return task;
  if (!cron.validate(config.jobs.liveCheckCronSchedule)) {
    throw new Error(`Invalid live check schedule: ${config.jobs.liveCheckCronSchedule}`);
  }
  task = cron.schedule(config.jobs.liveCheckCronSchedule, runOnce);
  // Render's free tier sleeps; take a reading as soon as the server wakes.
  setTimeout(runOnce, 5000);
  logger.info('Live check cron scheduled', { schedule: config.jobs.liveCheckCronSchedule });
  return task;
};

const stop = () => {
  if (task) {
    task.stop();
    task = null;
  }
};

module.exports = { start, stop, runOnce };
