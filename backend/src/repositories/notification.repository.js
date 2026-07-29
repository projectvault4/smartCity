const db = require('../config/db');

const selectableColumns = `
  id,
  user_id,
  advisory_sent_id,
  title,
  message,
  channel,
  status,
  is_read,
  metadata,
  read_at,
  created_at,
  updated_at
`;

const create = async ({
  userId,
  advisorySentId = null,
  title,
  message,
  channel = 'in_app',
  status = 'pending',
  isRead = false,
  metadata = {},
  readAt = null
}) => {
  const result = await db.query(
    `
      INSERT INTO notifications (
        user_id,
        advisory_sent_id,
        title,
        message,
        channel,
        status,
        is_read,
        metadata,
        read_at
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
      RETURNING ${selectableColumns}
    `,
    [userId, advisorySentId, title, message, channel, status, isRead, metadata, readAt]
  );

  return result.rows[0];
};

const findByUserId = async ({ userId, unreadOnly = false, limit = 50, offset = 0 }) => {
  const filters = ['user_id = $1'];
  const values = [userId];

  if (unreadOnly) {
    filters.push('is_read = false');
  }

  values.push(limit);
  const limitIndex = values.length;
  values.push(offset);
  const offsetIndex = values.length;

  const result = await db.query(
    `
      SELECT ${selectableColumns}
      FROM notifications
      WHERE ${filters.join(' AND ')}
      ORDER BY created_at DESC
      LIMIT $${limitIndex}
      OFFSET $${offsetIndex}
    `,
    values
  );

  return result.rows;
};

const markRead = async ({ id, userId }) => {
  const result = await db.query(
    `
      UPDATE notifications
      SET is_read = true,
          status = 'read',
          read_at = COALESCE(read_at, NOW())
      WHERE id = $1
        AND user_id = $2
      RETURNING ${selectableColumns}
    `,
    [id, userId]
  );

  return result.rows[0] || null;
};

module.exports = {
  create,
  findByUserId,
  markRead
};
