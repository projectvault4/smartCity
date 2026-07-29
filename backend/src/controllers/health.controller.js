const monitoringService = require('../services/monitoring.service');

const getHealth = (_req, res) => {
  res.status(200).json(monitoringService.getHealth());
};

const getMetrics = (_req, res) => {
  res.status(200).json(monitoringService.getMetrics());
};

const getReadiness = async (_req, res) => {
  const readiness = await monitoringService.getReadiness();
  const statusCode = readiness.status === 'ready' ? 200 : 503;

  res.status(statusCode).json(readiness);
};

module.exports = {
  getHealth,
  getMetrics,
  getReadiness
};
