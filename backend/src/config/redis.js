const Redis = require('ioredis');

const config = require('./env');

let client;

const getRedisClient = () => {
  if (!client) {
    client = new Redis(config.redis.url, {
      keyPrefix: config.redis.keyPrefix,
      connectTimeout: config.redis.connectTimeoutMs,
      maxRetriesPerRequest: 2,
      lazyConnect: true
    });

    client.on('error', () => {});
  }

  return client;
};

const testRedisConnection = async () => {
  const redis = getRedisClient();

  if (redis.status === 'wait') {
    await redis.connect();
  }

  return redis.ping();
};

const closeRedisConnection = async () => {
  if (client) {
    await client.quit();
    client = null;
  }
};

module.exports = {
  getRedisClient,
  testRedisConnection,
  closeRedisConnection
};
