const db = require('../config/db');

const selectableColumns = `
  id,
  user_id,
  name,
  email,
  phone,
  age,
  city,
  ward,
  latitude,
  longitude,
  preferences,
  status,
  created_at,
  updated_at
`;

const writableColumns = [
  'user_id',
  'name',
  'email',
  'phone',
  'age',
  'city',
  'ward',
  'latitude',
  'longitude',
  'preferences',
  'status'
];

const activeUserColumns = `
  users.id,
  users.user_id,
  users.name,
  users.email,
  users.phone,
  users.age,
  users.city,
  users.ward,
  users.latitude,
  users.longitude,
  users.preferences,
  users.status,
  users.created_at,
  users.updated_at
`;

const findAll = async ({ limit = 50, offset = 0, status, city, ward, search }) => {
  const values = [];
  const filters = [];

  if (status) {
    values.push(status);
    filters.push(`status = $${values.length}`);
  }

  if (city) {
    values.push(city);
    filters.push(`city ILIKE $${values.length}`);
  }

  if (ward) {
    values.push(ward);
    filters.push(`ward ILIKE $${values.length}`);
  }

  if (search) {
    values.push(`%${search}%`);
    filters.push(`(name ILIKE $${values.length} OR email ILIKE $${values.length} OR user_id ILIKE $${values.length})`);
  }

  const whereClause = filters.length ? `WHERE ${filters.join(' AND ')}` : '';
  const countResult = await db.query(`SELECT COUNT(*)::int AS total FROM users ${whereClause}`, values);

  values.push(limit);
  const limitIndex = values.length;
  values.push(offset);
  const offsetIndex = values.length;

  const result = await db.query(
    `
      SELECT ${selectableColumns}
      FROM users
      ${whereClause}
      ORDER BY created_at DESC
      LIMIT $${limitIndex}
      OFFSET $${offsetIndex}
    `,
    values
  );

  return {
    data: result.rows,
    meta: {
      total: countResult.rows[0].total,
      limit,
      offset
    }
  };
};

const findById = async (id) => {
  const result = await db.query(
    `SELECT ${selectableColumns} FROM users WHERE id = $1`,
    [id]
  );

  return result.rows[0] || null;
};

const create = async (user) => {
  const columns = writableColumns.filter((column) => Object.prototype.hasOwnProperty.call(user, column));
  const values = columns.map((column) => user[column]);
  const placeholders = columns.map((_, index) => `$${index + 1}`);

  const result = await db.query(
    `
      INSERT INTO users (${columns.join(', ')})
      VALUES (${placeholders.join(', ')})
      RETURNING ${selectableColumns}
    `,
    values
  );

  return result.rows[0];
};

const updateById = async (id, user) => {
  const columns = writableColumns.filter((column) => Object.prototype.hasOwnProperty.call(user, column));
  const values = columns.map((column) => user[column]);
  const assignments = columns.map((column, index) => `${column} = $${index + 1}`);

  values.push(id);

  const result = await db.query(
    `
      UPDATE users
      SET ${assignments.join(', ')}
      WHERE id = $${values.length}
      RETURNING ${selectableColumns}
    `,
    values
  );

  return result.rows[0] || null;
};

const deleteById = async (id) => {
  const result = await db.query('DELETE FROM users WHERE id = $1 RETURNING id', [id]);

  return result.rowCount > 0;
};

const findActiveWithRiskGroups = async ({ limit = 100, offset = 0 } = {}) => {
  const result = await db.query(
    `
      SELECT
        ${activeUserColumns},
        COALESCE(
          ARRAY_AGG(urg.group_key) FILTER (WHERE urg.group_key IS NOT NULL),
          ARRAY[]::risk_group_key[]
        ) AS groups
      FROM users
      LEFT JOIN user_risk_groups urg ON urg.user_id = users.id
      WHERE users.status = 'active'
      GROUP BY users.id
      ORDER BY users.created_at ASC
      LIMIT $1
      OFFSET $2
    `,
    [limit, offset]
  );

  return result.rows;
};

module.exports = {
  findAll,
  findById,
  create,
  updateById,
  deleteById,
  findActiveWithRiskGroups
};
