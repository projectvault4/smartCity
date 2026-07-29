BEGIN;

CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_city_ward ON users(city, ward);
CREATE INDEX idx_users_email_lower ON users(LOWER(email)) WHERE email IS NOT NULL;
CREATE INDEX idx_users_phone ON users(phone) WHERE phone IS NOT NULL;
CREATE INDEX idx_users_preferences_gin ON users USING GIN (preferences);

CREATE INDEX idx_user_risk_groups_group_key ON user_risk_groups(group_key);
CREATE INDEX idx_user_risk_groups_user_id ON user_risk_groups(user_id);

CREATE INDEX idx_advisory_rules_active_group ON advisory_rules(group_key, is_active);
CREATE INDEX idx_advisory_rules_condition_type ON advisory_rules(condition_type);
CREATE INDEX idx_advisory_rules_severity ON advisory_rules(severity);
CREATE INDEX idx_advisory_rules_condition_json_gin ON advisory_rules USING GIN (condition_json);
CREATE INDEX idx_advisory_rules_active_window
  ON advisory_rules(starts_at, ends_at)
  WHERE is_active = true;

CREATE INDEX idx_advisories_sent_user_sent_at ON advisories_sent(user_id, sent_at DESC);
CREATE INDEX idx_advisories_sent_rule_sent_at ON advisories_sent(advisory_rule_id, sent_at DESC);
CREATE INDEX idx_advisories_sent_severity_sent_at ON advisories_sent(severity, sent_at DESC);
CREATE INDEX idx_advisories_sent_delivery_status ON advisories_sent(delivery_status);
CREATE INDEX idx_advisories_sent_channel_sent_at ON advisories_sent(channel, sent_at DESC);
CREATE INDEX idx_advisories_sent_context_gin ON advisories_sent USING GIN (context);

CREATE INDEX idx_notifications_user_created_at ON notifications(user_id, created_at DESC);
CREATE INDEX idx_notifications_user_unread
  ON notifications(user_id, created_at DESC)
  WHERE is_read = false;
CREATE INDEX idx_notifications_status ON notifications(status);
CREATE INDEX idx_notifications_channel_status ON notifications(channel, status);
CREATE INDEX idx_notifications_advisory_sent_id ON notifications(advisory_sent_id);
CREATE INDEX idx_notifications_metadata_gin ON notifications USING GIN (metadata);

CREATE INDEX idx_execution_log_job_started_at ON execution_log(job_name, started_at DESC);
CREATE INDEX idx_execution_log_status_started_at ON execution_log(status, started_at DESC);
CREATE INDEX idx_execution_log_completed_at ON execution_log(completed_at DESC);
CREATE INDEX idx_execution_log_metadata_gin ON execution_log USING GIN (metadata);

COMMIT;
