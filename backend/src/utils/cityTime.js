const config = require('../config/env');

// Forecast CSVs store naive local timestamps ("2026-10-26 07:00:00") that mean
// city time (IST for Bengaluru). `new Date(naive)` would read them in the
// SERVER's timezone (UTC on Render), shifting every value by 5.5 hours, so all
// timestamp parsing goes through these helpers.

const offsetMs = () => (config.modelForecast.tzOffsetMinutes ?? 330) * 60 * 1000;

const NAIVE = /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?$/;

const parseCityTimestamp = (value) => {
  if (value instanceof Date) return value.getTime();
  const text = String(value || '').trim();
  const match = text.match(NAIVE);
  if (match) {
    const [, y, mo, d, h, mi, s] = match.map((part) => (part === undefined ? '0' : part));
    return Date.UTC(Number(y), Number(mo) - 1, Number(d), Number(h), Number(mi), Number(s)) - offsetMs();
  }
  return new Date(text).getTime();
};

// Calendar parts of an instant, read on the city clock.
const cityParts = (ms) => {
  const local = new Date(ms + offsetMs());
  return {
    year: local.getUTCFullYear(),
    month: local.getUTCMonth() + 1,
    day: local.getUTCDate(),
    hour: local.getUTCHours(),
    minute: local.getUTCMinutes(),
    dayOfWeek: (local.getUTCDay() + 6) % 7 // Monday = 0, matches pandas dayofweek
  };
};

const floorToCityHour = (ms) => {
  const hourMs = 60 * 60 * 1000;
  return Math.floor((ms + offsetMs()) / hourMs) * hourMs - offsetMs();
};

const formatCityHour = (ms) => {
  const { hour } = cityParts(ms);
  const h12 = hour % 12 || 12;
  return `${h12} ${hour < 12 ? 'AM' : 'PM'}`;
};

module.exports = {
  parseCityTimestamp,
  cityParts,
  floorToCityHour,
  formatCityHour
};
