const fs = require('fs');
const path = require('path');

const { pool } = require('../config/db');
const logger = require('../utils/logger');

const migrationsDir = path.resolve(__dirname, '../migrations');

const ensureMigrationsTable = async (client) => {
  await client.query(`
    CREATE TABLE IF NOT EXISTS schema_migrations (
      id SERIAL PRIMARY KEY,
      filename TEXT NOT NULL UNIQUE,
      applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
};

const getAppliedMigrations = async (client) => {
  const result = await client.query('SELECT filename FROM schema_migrations');

  return new Set(result.rows.map((row) => row.filename));
};

const run = async () => {
  const client = await pool.connect();

  try {
    await ensureMigrationsTable(client);
    const applied = await getAppliedMigrations(client);
    const files = fs
      .readdirSync(migrationsDir)
      .filter((file) => file.endsWith('.sql'))
      .sort();

    for (const file of files) {
      if (applied.has(file)) {
        logger.info('Skipping applied migration', { file });
        continue;
      }

      const sql = fs.readFileSync(path.join(migrationsDir, file), 'utf8');

      logger.info('Applying migration', { file });
      await client.query(sql);
      await client.query('INSERT INTO schema_migrations (filename) VALUES ($1)', [file]);
      logger.info('Migration applied', { file });
    }

    logger.info('Database migrations complete');
  } finally {
    client.release();
    await pool.end();
  }
};

run().catch(async (error) => {
  logger.error('Database migration failed', { message: error.message, stack: error.stack });
  await pool.end();
  process.exit(1);
});
