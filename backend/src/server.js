const app = require('./app');
const config = require('./config/env');
const { testConnection } = require('./config/db');
const advisoryCron = require('./jobs/advisoryCron.job');
const logger = require('./utils/logger');

const startServer = async () => {
  try {
    await testConnection();

    logger.info('SERVER STARTUP', {
      serverTime: new Date().toString(),
      isoTime: new Date().toISOString(),
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
    });

    app.listen(config.port, () => {
      logger.info('ForeSightX backend running', {
        port: config.port,
        baseUrl: `http://localhost:${config.port}`
      });

      if (config.jobs.advisoryCronEnabled) {
        advisoryCron.start();
      }
    });
  } catch (error) {
    logger.error('Failed to start server', {
      message: error.message,
      stack: error.stack
    });
    process.exit(1);
  }
};

startServer();