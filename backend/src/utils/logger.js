const winston = require('winston');

const config = require('../config/env');

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || (config.nodeEnv === 'production' ? 'info' : 'debug'),
  defaultMeta: {
    service: 'foresightx-backend'
  },
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.splat(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console({
      silent: config.nodeEnv === 'test'
    })
  ]
});

module.exports = logger;
