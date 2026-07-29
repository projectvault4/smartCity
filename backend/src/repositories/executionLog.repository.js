const db = require('../config/db');

const selectableColumns = `
  id,
  job_name,
  status,
  started_at,
  completed_at,
  duration_ms,
  users_processed,
  advisories_generated,
  notifications_sent,
  error_message,
  metadata,
  created_at
`;

const create = async ({ jobName, metadata = {} }) => {
  const result = await db.query(
    `
      INSERT INTO execution_log (job_name, status, metadata)
      VALUES ($1, 'running', $2)
      RETURNING ${selectableColumns}
    `,
    [jobName, metadata]
  );

  return result.rows[0];
};

const complete = async ({
  id,
  status,
  startedAt,
  usersProcessed,
  advisoriesGenerated,
  notificationsSent,
  errorMessage = null,
  metadata = {}
}) => {
  const durationMs = Math.max(Date.now() - new Date(startedAt).getTime(), 0);
  const result = await db.query(
    `
      UPDATE execution_log
      SET status = $2,
          completed_at = NOW(),
          duration_ms = $3,
          users_processed = $4,
          advisories_generated = $5,
          notifications_sent = $6,
          error_message = $7,
          metadata = metadata || $8::jsonb
      WHERE id = $1
      RETURNING ${selectableColumns}
    `,
    [
      id,
      status,
      durationMs,
      usersProcessed,
      advisoriesGenerated,
      notificationsSent,
      errorMessage,
      metadata
    ]
  );

  return result.rows[0];
};

module.exports = {
  create,
  complete
};
