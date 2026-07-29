BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE user_status AS ENUM ('active', 'inactive', 'suspended');
CREATE TYPE risk_group_key AS ENUM ('resp', 'elder', 'child', 'worker', 'commuter', 'general');
CREATE TYPE advisory_severity AS ENUM ('info', 'low', 'medium', 'warning', 'critical');
CREATE TYPE notification_channel AS ENUM ('in_app', 'email', 'sms', 'push');
CREATE TYPE notification_status AS ENUM ('pending', 'sent', 'delivered', 'failed', 'read');
CREATE TYPE execution_status AS ENUM ('running', 'success', 'failed', 'partial');

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id VARCHAR(80) NOT NULL UNIQUE,
  name VARCHAR(160) NOT NULL,
  email VARCHAR(255) UNIQUE,
  phone VARCHAR(32),
  age INTEGER CHECK (age IS NULL OR age BETWEEN 0 AND 130),
  city VARCHAR(120),
  ward VARCHAR(120),
  latitude NUMERIC(9, 6),
  longitude NUMERIC(9, 6),
  preferences JSONB NOT NULL DEFAULT '{}'::jsonb,
  status user_status NOT NULL DEFAULT 'active',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TRIGGER users_set_updated_at
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

CREATE TABLE user_risk_groups (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  group_key risk_group_key NOT NULL,
  source VARCHAR(80) NOT NULL DEFAULT 'manual',
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (user_id, group_key)
);

CREATE TRIGGER user_risk_groups_set_updated_at
BEFORE UPDATE ON user_risk_groups
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

CREATE TABLE advisory_rules (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  rule_name VARCHAR(160) NOT NULL UNIQUE,
  group_key risk_group_key NOT NULL,
  condition_type VARCHAR(80) NOT NULL,
  condition_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  advisory_title VARCHAR(220) NOT NULL,
  advisory_description TEXT NOT NULL,
  severity advisory_severity NOT NULL,
  score_weight NUMERIC(6, 2) NOT NULL DEFAULT 1 CHECK (score_weight >= 0),
  priority INTEGER NOT NULL DEFAULT 100 CHECK (priority >= 0),
  is_active BOOLEAN NOT NULL DEFAULT true,
  starts_at TIMESTAMPTZ,
  ends_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (ends_at IS NULL OR starts_at IS NULL OR ends_at > starts_at)
);

CREATE TRIGGER advisory_rules_set_updated_at
BEFORE UPDATE ON advisory_rules
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

CREATE TABLE advisories_sent (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  advisory_rule_id UUID REFERENCES advisory_rules(id) ON DELETE SET NULL,
  title VARCHAR(220) NOT NULL,
  message TEXT NOT NULL,
  severity advisory_severity NOT NULL,
  risk_score NUMERIC(8, 2),
  risk_level VARCHAR(40),
  context JSONB NOT NULL DEFAULT '{}'::jsonb,
  channel notification_channel NOT NULL DEFAULT 'in_app',
  delivery_status notification_status NOT NULL DEFAULT 'pending',
  email_sent BOOLEAN NOT NULL DEFAULT false,
  sms_sent BOOLEAN NOT NULL DEFAULT false,
  error_message TEXT,
  sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  delivered_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE notifications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  advisory_sent_id UUID REFERENCES advisories_sent(id) ON DELETE SET NULL,
  title VARCHAR(220) NOT NULL,
  message TEXT NOT NULL,
  channel notification_channel NOT NULL DEFAULT 'in_app',
  status notification_status NOT NULL DEFAULT 'pending',
  is_read BOOLEAN NOT NULL DEFAULT false,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  read_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK ((is_read = false AND read_at IS NULL) OR (is_read = true AND read_at IS NOT NULL))
);

CREATE TRIGGER notifications_set_updated_at
BEFORE UPDATE ON notifications
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

CREATE TABLE execution_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_name VARCHAR(140) NOT NULL,
  status execution_status NOT NULL DEFAULT 'running',
  started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at TIMESTAMPTZ,
  duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
  users_processed INTEGER NOT NULL DEFAULT 0 CHECK (users_processed >= 0),
  advisories_generated INTEGER NOT NULL DEFAULT 0 CHECK (advisories_generated >= 0),
  notifications_sent INTEGER NOT NULL DEFAULT 0 CHECK (notifications_sent >= 0),
  error_message TEXT,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (completed_at IS NULL OR completed_at >= started_at)
);

COMMIT;
