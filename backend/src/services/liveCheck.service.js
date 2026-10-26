// Live reality check: compare the outlook with what is observed right now.
//
// Sources: WAQI (station PM2.5 / PM10) and OpenWeather (temperature, humidity)
// for central Bengaluru. Traffic counts and BESCOM load have no public live
// feed, so those two domains cannot be checked live.
//
// WAQI reports US-EPA AQI sub-indices. The model is trained on CPCB AQI, so the
// PM sub-indices are converted back to concentrations and re-indexed on the
// CPCB scale (approximate: CPCB uses more pollutants and 24 h averages).

const axios = require('axios');

const config = require('../config/env');
const db = require('../config/db');
const logger = require('../utils/logger');
const dataProviders = require('./dataProviders.service');
const modelForecast = require('./modelForecast.service');
const decisionEngine = require('./decisionEngine.service');
const liveState = require('./liveState');
const { cityParts, floorToCityHour } = require('../utils/cityTime');

const CITY = { name: 'bangalore', latitude: 12.9716, longitude: 77.5946 };
const DRIFT_RATIO = 1.5;
const MIN_POINTS = 6;
const MAX_STATION_KM = 60;
const OPEN_METEO_AQ = 'https://air-quality-api.open-meteo.com/v1/air-quality';
const OPEN_METEO_WX = 'https://api.open-meteo.com/v1/forecast';

const distanceKm = (a, b) => {
  const rad = (d) => (d * Math.PI) / 180;
  const dLat = rad(b.latitude - a.latitude);
  const dLon = rad(b.longitude - a.longitude);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a.latitude)) * Math.cos(rad(b.latitude)) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(h));
};

// Station AQI from WAQI. Rejected if the station is not in Bengaluru (the
// public "demo" token always answers with Shanghai).
const fetchWaqi = async () => {
  const aqi = await dataProviders.fetchAqi({ latitude: CITY.latitude, longitude: CITY.longitude });
  const loc = aqi.location || {};
  if (!Number.isFinite(loc.latitude) || distanceKm(CITY, loc) > MAX_STATION_KM) {
    throw new Error(`WAQI returned a station outside Bengaluru (${loc.name || 'unknown'}); check WAQI_API_KEY (the "demo" token only serves Shanghai).`);
  }
  return {
    source: 'WAQI station',
    station: loc.name,
    stationTime: aqi.observedAt,
    aqiCpcb: toCpcbAqi(aqi.pollutants),
    aqiUs: aqi.aqi,
    pm25UsAqi: aqi.pollutants?.pm25 ?? null,
    pm10UsAqi: aqi.pollutants?.pm10 ?? null,
    dominantPollutant: aqi.dominantPollutant || null
  };
};

// Keyless fallback: Open-Meteo air quality (CAMS model concentrations, ug/m3).
const fetchOpenMeteoAir = async () => {
  const { data } = await axios.get(OPEN_METEO_AQ, {
    params: { latitude: CITY.latitude, longitude: CITY.longitude, current: 'pm2_5,pm10,us_aqi', timezone: 'Asia/Kolkata' },
    timeout: 10000
  });
  const c = data.current || {};
  const pm25 = concToCpcb(Number(c.pm2_5), CPCB_PM25);
  const pm10 = concToCpcb(Number(c.pm10), CPCB_PM10);
  const values = [pm25, pm10].filter((v) => Number.isFinite(v));
  return {
    source: 'Open-Meteo air quality (CAMS model, not a station)',
    station: 'Bengaluru grid cell',
    stationTime: c.time || null,
    aqiCpcb: values.length ? Math.max(...values) : null,
    aqiUs: c.us_aqi ?? null,
    pm25Concentration: c.pm2_5 ?? null,
    pm10Concentration: c.pm10 ?? null
  };
};

const fetchOpenWeather = async () => {
  const w = await dataProviders.fetchCurrentWeather({ latitude: CITY.latitude, longitude: CITY.longitude });
  return { source: 'OpenWeather', temperature: w.temperature?.value ?? null, humidity: w.humidity ?? null, description: w.weather?.description || null };
};

const fetchOpenMeteoWeather = async () => {
  const { data } = await axios.get(OPEN_METEO_WX, {
    params: { latitude: CITY.latitude, longitude: CITY.longitude, current: 'temperature_2m,relative_humidity_2m,precipitation', timezone: 'Asia/Kolkata' },
    timeout: 10000
  });
  const c = data.current || {};
  return { source: 'Open-Meteo', temperature: c.temperature_2m ?? null, humidity: c.relative_humidity_2m ?? null, precipitationMm: c.precipitation ?? null };
};

const firstAvailable = async (fetchers, errors) => {
  for (const [name, fn] of fetchers) {
    try {
      return await fn();
    } catch (error) {
      errors.push({ provider: name, message: error.message });
    }
  }
  return null;
};

// [aqiLo, aqiHi, concLo, concHi]
const US_PM25 = [[0, 50, 0, 12], [51, 100, 12.1, 35.4], [101, 150, 35.5, 55.4], [151, 200, 55.5, 150.4], [201, 300, 150.5, 250.4], [301, 400, 250.5, 350.4], [401, 500, 350.5, 500.4]];
const US_PM10 = [[0, 50, 0, 54], [51, 100, 55, 154], [101, 150, 155, 254], [151, 200, 255, 354], [201, 300, 355, 424], [301, 400, 425, 504], [401, 500, 505, 604]];
// [concLo, concHi, aqiLo, aqiHi]
const CPCB_PM25 = [[0, 30, 0, 50], [31, 60, 51, 100], [61, 90, 101, 200], [91, 120, 201, 300], [121, 250, 301, 400], [251, 500, 401, 500]];
const CPCB_PM10 = [[0, 50, 0, 50], [51, 100, 51, 100], [101, 250, 101, 200], [251, 350, 201, 300], [351, 430, 301, 400], [431, 800, 401, 500]];

const usAqiToConc = (aqi, table) => {
  if (!Number.isFinite(aqi)) return null;
  const band = table.find(([lo, hi]) => aqi <= hi) || table[table.length - 1];
  const [iLo, iHi, cLo, cHi] = band;
  return ((Math.max(aqi, iLo) - iLo) / (iHi - iLo)) * (cHi - cLo) + cLo;
};

const concToCpcb = (conc, table) => {
  if (!Number.isFinite(conc)) return null;
  const band = table.find(([, hi]) => conc <= hi) || table[table.length - 1];
  const [cLo, cHi, iLo, iHi] = band;
  return ((Math.max(conc, cLo) - cLo) / (cHi - cLo)) * (iHi - iLo) + iLo;
};

const toCpcbAqi = (pollutants = {}) => {
  const pm25 = concToCpcb(usAqiToConc(Number(pollutants.pm25), US_PM25), CPCB_PM25);
  const pm10 = concToCpcb(usAqiToConc(Number(pollutants.pm10), US_PM10), CPCB_PM10);
  const values = [pm25, pm10].filter((v) => Number.isFinite(v));
  return values.length ? Math.max(...values) : null;
};

// In-memory fallback when the database is unavailable (e.g. local dev without
// Postgres). Render's free tier sleeps, so stored history has gaps.
const memory = [];

const save = async (row) => {
  try {
    await db.query(
      `INSERT INTO live_observations
        (city, observed_hour, observed_at, aqi_cpcb, aqi_us, temperature, humidity,
         forecast_aqi, forecast_temperature, forecast_humidity, metadata)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
       ON CONFLICT (city, observed_hour) DO UPDATE SET
         observed_at = EXCLUDED.observed_at, aqi_cpcb = EXCLUDED.aqi_cpcb, aqi_us = EXCLUDED.aqi_us,
         temperature = EXCLUDED.temperature, humidity = EXCLUDED.humidity,
         forecast_aqi = EXCLUDED.forecast_aqi, forecast_temperature = EXCLUDED.forecast_temperature,
         forecast_humidity = EXCLUDED.forecast_humidity, metadata = EXCLUDED.metadata`,
      [row.city, row.observedHour, row.observedAt, row.aqiCpcb, row.aqiUs, row.temperature, row.humidity,
        row.forecastAqi, row.forecastTemperature, row.forecastHumidity, row.metadata]
    );
    return 'database';
  } catch (error) {
    logger.warn('live_observations insert failed; keeping in memory', { message: error.message });
    const i = memory.findIndex((m) => m.observedHour === row.observedHour);
    if (i >= 0) memory[i] = row; else memory.push(row);
    while (memory.length > 168) memory.shift();
    return 'memory';
  }
};

const history = async (hours = 48) => {
  try {
    const result = await db.query(
      `SELECT observed_hour, observed_at, aqi_cpcb, aqi_us, temperature, humidity,
              forecast_aqi, forecast_temperature, forecast_humidity, metadata
         FROM live_observations
        WHERE city = $1 AND observed_hour >= NOW() - ($2 || ' hours')::interval
        ORDER BY observed_hour ASC`,
      [CITY.name, String(hours)]
    );
    return result.rows.map((r) => ({
      observedHour: new Date(r.observed_hour).toISOString(),
      observedAt: new Date(r.observed_at).toISOString(),
      aqiCpcb: r.aqi_cpcb === null ? null : Number(r.aqi_cpcb),
      aqiUs: r.aqi_us === null ? null : Number(r.aqi_us),
      temperature: r.temperature === null ? null : Number(r.temperature),
      humidity: r.humidity === null ? null : Number(r.humidity),
      forecastAqi: r.forecast_aqi === null ? null : Number(r.forecast_aqi),
      forecastTemperature: r.forecast_temperature === null ? null : Number(r.forecast_temperature),
      forecastHumidity: r.forecast_humidity === null ? null : Number(r.forecast_humidity),
      metadata: r.metadata
    }));
  } catch {
    const cutoff = Date.now() - hours * 3600 * 1000;
    return memory.filter((m) => new Date(m.observedHour).getTime() >= cutoff);
  }
};

const record = async () => {
  const now = Date.now();
  const hourMs = floorToCityHour(now);
  const errors = [];
  const [air, weather] = await Promise.all([
    firstAvailable([['WAQI', fetchWaqi], ['Open-Meteo air quality', fetchOpenMeteoAir]], errors),
    firstAvailable([['OpenWeather', fetchOpenWeather], ['Open-Meteo', fetchOpenMeteoWeather]], errors)
  ]);
  if (!air && !weather) {
    const error = new Error('No live provider responded');
    error.statusCode = 503;
    error.details = errors;
    throw error;
  }
  const outlook = modelForecast.getOutlookAt({ city: CITY.name, ms: now });
  const row = {
    city: CITY.name,
    observedHour: new Date(hourMs).toISOString(),
    observedAt: new Date(now).toISOString(),
    aqiCpcb: air?.aqiCpcb ?? null,
    aqiUs: air?.aqiUs ?? null,
    temperature: weather?.temperature ?? null,
    humidity: weather?.humidity ?? null,
    forecastAqi: outlook?.aqi ?? null,
    forecastTemperature: outlook?.temperature ?? null,
    forecastHumidity: outlook?.humidity ?? null,
    metadata: { air, weather, errors }
  };
  row.storage = await save(row);
  return row;
};

const err = (obs, fc) => (Number.isFinite(obs) && Number.isFinite(fc) ? obs - fc : null);

const summarize = (rows) => {
  const out = {};
  const pairs = [
    ['aqi', 'aqiCpcb', 'forecastAqi'],
    ['temperature', 'temperature', 'forecastTemperature'],
    ['humidity', 'humidity', 'forecastHumidity']
  ];
  pairs.forEach(([target, obsKey, fcKey]) => {
    const points = rows
      .map((r) => {
        const e = err(r[obsKey], r[fcKey]);
        if (e === null) return null;
        const hour = cityParts(new Date(r.observedHour).getTime()).hour;
        const interval = target === 'humidity' ? null : decisionEngine.outlookInterval(target, r[fcKey], hour);
        return { e, inside: interval ? r[obsKey] >= interval.low && r[obsKey] <= interval.high : null, expected: interval?.expectedMae ?? null };
      })
      .filter(Boolean);
    if (!points.length) {
      out[target] = { points: 0, status: 'no_data' };
      return;
    }
    const mae = points.reduce((s, p) => s + Math.abs(p.e), 0) / points.length;
    const bias = points.reduce((s, p) => s + p.e, 0) / points.length;
    const withInterval = points.filter((p) => p.inside !== null);
    const coverage = withInterval.length ? withInterval.filter((p) => p.inside).length / withInterval.length : null;
    const expected = points[0].expected;
    const ratio = expected ? mae / expected : null;
    let status = 'insufficient_data';
    if (points.length >= MIN_POINTS) status = ratio !== null && ratio > DRIFT_RATIO ? 'drift' : 'ok';
    out[target] = { points: points.length, mae, bias, coverage, expectedMae: expected, errorRatio: ratio, status };
  });
  return out;
};

const getLiveCheck = async ({ refresh = true, hours = 48 } = {}) => {
  let latest = null;
  let refreshError = null;
  if (refresh) {
    try {
      latest = await record();
    } catch (error) {
      refreshError = { message: error.message, details: error.details };
    }
  }
  const rows = await history(hours);
  if (!latest && rows.length) latest = rows[rows.length - 1];
  const stats = summarize(rows.filter((r) => Date.now() - new Date(r.observedHour).getTime() <= 24 * 3600 * 1000));
  liveState.setLiveStats(stats);
  const drift = Object.entries(stats).filter(([, s]) => s.status === 'drift').map(([t]) => t);
  if (latest) {
    const hour = cityParts(new Date(latest.observedHour).getTime()).hour;
    latest.intervals = {
      aqi: decisionEngine.outlookInterval('aqi', latest.forecastAqi, hour),
      temperature: decisionEngine.outlookInterval('temperature', latest.forecastTemperature, hour)
    };
    latest.aqiCategory = decisionEngine.cpcbCategory(latest.aqiCpcb);
  }
  return {
    generatedAt: new Date().toISOString(),
    city: CITY.name,
    latest,
    history: rows,
    stats,
    drift,
    status: drift.length ? 'drift' : Object.values(stats).some((s) => s.status === 'ok') ? 'ok' : 'collecting',
    rule: `Drift when the last 24 h error is more than ${DRIFT_RATIO}x the error measured on held-out data (needs ${MIN_POINTS}+ hourly readings).`,
    notes: [
      'AQI from a WAQI Bengaluru station (US-EPA PM sub-indices) or, without a WAQI key, Open-Meteo PM concentrations; both converted to the CPCB scale (approximate, instantaneous rather than 24 h average).',
      'Weather from OpenWeather or, without a key, Open-Meteo.',
      'No public live feed exists for traffic counts or BESCOM load, so those are not checked live.',
      config.jobs.liveCheckCronEnabled ? 'Readings are recorded hourly while the server is awake.' : 'Readings are recorded when this panel is opened.'
    ],
    refreshError
  };
};

module.exports = { getLiveCheck, record, toCpcbAqi };
