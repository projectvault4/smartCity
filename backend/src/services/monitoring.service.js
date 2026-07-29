const db = require('../config/db');
const { testRedisConnection } = require('../config/redis');

const startedAt = Date.now();

const getHealth = () => ({
  status: 'ok',
  service: 'foresightx-backend',
  uptimeSeconds: Math.floor(process.uptime()),
  startedAt: new Date(startedAt).toISOString()
});

const getMetrics = () => {
  const memory = process.memoryUsage();

  return {
    process: {
      uptimeSeconds: Math.floor(process.uptime()),
      pid: process.pid,
      nodeVersion: process.version,
      environment: process.env.NODE_ENV || 'development'
    },
    memory: {
      rss: memory.rss,
      heapTotal: memory.heapTotal,
      heapUsed: memory.heapUsed,
      external: memory.external,
      arrayBuffers: memory.arrayBuffers
    }
  };
};

const checkDependency = async (name, check) => {
  const started = Date.now();

  try {
    await check();
    return {
      name,
      status: 'ok',
      latencyMs: Date.now() - started
    };
  } catch (error) {
    return {
      name,
      status: 'error',
      latencyMs: Date.now() - started,
      message: error.message
    };
  }
};

const getReadiness = async () => {
  const dependencies = await Promise.all([
    checkDependency('postgresql', () => db.query('SELECT 1')),
    checkDependency('redis', () => testRedisConnection())
  ]);
  const ready = dependencies.every((dependency) => dependency.status === 'ok');

  return {
    status: ready ? 'ready' : 'degraded',
    dependencies
  };
};

module.exports = {
  getHealth,
  getMetrics,
  getReadiness
};
