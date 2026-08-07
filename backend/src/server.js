const app = require('./app');
const config = require('./config/env');
const { testConnection } = require('./config/db');
const advisoryCron = require('./jobs/advisoryCron.job');
const logger = require('./utils/logger');

console.log("====================================");
console.log("Server time:", new Date().toString());
console.log("ISO time   :", new Date().toISOString());
console.log("Timezone   :", Intl.DateTimeFormat().resolvedOptions().timeZone);
console.log("====================================");

const startServer = async () => {
  try {
    await testConnection();

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