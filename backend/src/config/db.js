const { Pool } = require('pg');
const config = require('./env');

const pool = new Pool(config.db);

const testConnection = async () => {
  const client = await pool.connect();

  try {
    await client.query('SELECT 1');
    console.log('PostgreSQL connection ready');
  } finally {
    client.release();
  }
};

module.exports = {
  pool,
  query: (text, params) => pool.query(text, params),
  testConnection
};
