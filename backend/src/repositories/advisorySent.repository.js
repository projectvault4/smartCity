const db = require('../config/db');

const selectableColumns = `
  id,
  user_id,
  advisory_rule_id,
  title,
  message,
  severity,
  risk_score,
  risk_level,
  context,
  channel,
  delivery_status,
  email_sent,
  sms_sent,
  error_message,
  sent_at,
  delivered_at,
  created_at
`;

const create = async ({
  userId,
  advisoryRuleId = null,
  title,
  message,
  severity,
  riskScore = null,
  riskLevel = null,
  context = {},
  channel,
  deliveryStatus = 'pending',
  emailSent = false,
  smsSent = false,
  errorMessage = null,
  deliveredAt = null
}) => {
  const result = await db.query(
    `
      INSERT INTO advisories_sent (
        user_id,
        advisory_rule_id,
        title,
        message,
        severity,
        risk_score,
        risk_level,
        context,
        channel,
        delivery_status,
        email_sent,
        sms_sent,
        error_message,
        delivered_at
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
      RETURNING ${selectableColumns}
    `,
    [
      userId,
      advisoryRuleId,
      title,
      message,
      severity,
      riskScore,
      riskLevel,
      context,
      channel,
      deliveryStatus,
      emailSent,
      smsSent,
      errorMessage,
      deliveredAt
    ]
  );

  return result.rows[0];
};

const findSmsHistory = async ({ limit = 50, offset = 0 } = {}) => {
  const result = await db.query(
    `
      SELECT
        a.id,
        a.user_id,
        u.name AS member_name,
        u.phone AS member_phone,
        a.title,
        a.message,
        a.severity,
        a.risk_score,
        a.channel,
        a.delivery_status,
        a.sms_sent,
        a.error_message,
        a.sent_at
      FROM advisories_sent a
      LEFT JOIN users u ON u.id = a.user_id
      WHERE a.channel = 'sms'
      ORDER BY a.created_at DESC
      LIMIT $1
      OFFSET $2
    `,
    [limit, offset]
  );

  return result.rows;
};

// Event ids already delivered to a user recently (cooldown / dedupe for the
// decision engine, so the 15-minute cron cannot resend the same alert).
const findRecentEventIds = async ({ userId, sinceHours }) => {
  const result = await db.query(
    `
      SELECT DISTINCT context->>'event' AS event
      FROM advisories_sent
      WHERE user_id = $1
        AND context ? 'event'
        AND delivery_status IN ('sent', 'delivered')
        AND sent_at >= NOW() - ($2 || ' hours')::interval
    `,
    [userId, String(sinceHours)]
  );
  return new Set(result.rows.map((row) => row.event));
};

module.exports = {
  findRecentEventIds,
  create,
  findSmsHistory
};
