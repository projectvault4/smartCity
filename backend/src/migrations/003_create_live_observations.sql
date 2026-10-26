BEGIN;

-- Observed conditions (WAQI + OpenWeather) stored next to the outlook value for
-- the same hour, so forecast-vs-reality error and drift can be measured live.
CREATE TABLE IF NOT EXISTS live_observations (
  id BIGSERIAL PRIMARY KEY,
  city VARCHAR(80) NOT NULL DEFAULT 'bangalore',
  observed_hour TIMESTAMPTZ NOT NULL,
  observed_at TIMESTAMPTZ NOT NULL,
  aqi_cpcb NUMERIC(8, 2),
  aqi_us NUMERIC(8, 2),
  temperature NUMERIC(6, 2),
  humidity NUMERIC(6, 2),
  forecast_aqi NUMERIC(8, 2),
  forecast_temperature NUMERIC(6, 2),
  forecast_humidity NUMERIC(6, 2),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (city, observed_hour)
);

CREATE INDEX IF NOT EXISTS live_observations_city_hour_idx ON live_observations (city, observed_hour DESC);

COMMIT;
