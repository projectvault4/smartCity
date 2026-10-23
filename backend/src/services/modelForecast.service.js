const fs = require('fs');
const path = require('path');

const config = require('../config/env');

const parseCsvLine = (line) => {
  const values = [];
  let current = '';
  let inQuotes = false;

  for (const char of line) {
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      values.push(current);
      current = '';
    } else {
      current += char;
    }
  }

  values.push(current);
  return values;
};

const forecastFileName = (year) => {
  if (year === '2026') {
    return 'forecast_2026.csv';
  }
  return 'past_present_future_forecast.csv';
};

const readForecastRows = (city = config.modelForecast.defaultCity, year = 'ppf') => {
  const cityKey = String(city || '').toLowerCase();
  // The default-city training run writes to the repository root outputs/ dir
  // (shared with the anomaly + multivariate pipelines). Only cities with a
  // dedicated subdirectory (e.g. delhi) resolve into outputs/<city>/.
  const forecastPath = ['delhi'].includes(cityKey)
    ? path.join(
        config.modelForecast.projectRoot,
        'outputs',
        cityKey,
        forecastFileName(year)
      )
    : path.join(
        config.modelForecast.projectRoot,
        'outputs',
        forecastFileName(year)
      );

  if (!fs.existsSync(forecastPath)) {
    throw new Error(`Model forecast file not found: ${forecastPath}`);
  }

  const [headerLine, ...lines] = fs.readFileSync(forecastPath, 'utf8').trim().split(/\r?\n/);
  const headers = parseCsvLine(headerLine);

  return lines
    .filter(Boolean)
    .map((line) => {
      const values = parseCsvLine(line);
      return headers.reduce((row, header, index) => {
        row[header] = values[index];
        return row;
      }, {});
    });
};

const toNumber = (value) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

const pad = (n) => String(n).padStart(2, '0');

const formatTimestamp = (date) => date.toISOString();

const toTimestampMs = (timestamp) => {
  return new Date(timestamp).getTime();
};

// Re-anchor future rows to the live server clock. The model forecasts N steps
// ahead of its last training timestamp; we keep the relative spacing and map
// step_ahead k onto the k-th step boundary after "now", so the dashboard reads
// like a rolling next-24h outlook instead of a frozen 2025 date.
// Floor to the top of the hour in the CITY's timezone (IST by default), not the
// server's. Render runs in UTC, and IST is UTC+5:30, so flooring in UTC lands on
// :30 IST and can publish an hour that has already passed.
const floorToLocalHour = (date) => {
  const offsetMs = (config.modelForecast.tzOffsetMinutes ?? 330) * 60 * 1000;
  const hourMs = 60 * 60 * 1000;
  return Math.floor((date.getTime() + offsetMs) / hourMs) * hourMs - offsetMs;
};

const rebaseFutureTimestamps = (rows) => {
  const stepMs = (config.modelForecast.stepMinutes || 60) * 60 * 1000;
  const now = Date.now();
  const anchorMs = floorToLocalHour(new Date(now));

  return rows.map((row) => {
    if (row.time_segment !== 'future') {
      return row;
    }
    const step = Math.max(1, Number(row.step_ahead) || 1);
    return {
      ...row,
      timestamp: formatTimestamp(new Date(anchorMs + step * stepMs))
    };
  });
};

// The full-year 2026 forecast file already contains real 2026 timestamps. We
// anchor it to the live server clock by finding the hourly row for the FIRST
// forecast hour (so the forecast always reflects the CURRENT month/season
// instead of always restarting from the file's first row on January 1st) and
// then rebase the selected window onto the live clock so the dashboard reads
// like a rolling real-time forecast for the current 2026 date.
const anchor2026Rows = (rows) => {
  const stepMs = (config.modelForecast.stepMinutes || 60) * 60 * 1000;

  // step_ahead k means "k steps after the current step boundary", so the first
  // row must land on the hour AFTER the current one. Anchoring on the floored
  // current hour would publish a T+1H row that is already in the past (e.g. at
  // 10:48 it would read "forecast from 10 AM" instead of 11 AM).
  const anchorMs = floorToLocalHour(new Date());
  const targetMs = anchorMs + stepMs;

  // Pick the earliest row at or after the first forecast hour so the seasonal
  // values line up with the horizon we publish. Choosing purely on absolute
  // distance could snap backwards onto an already-elapsed hour.
  let startIndex = -1;
  let earliestForwardTs = Infinity;
  let fallbackIndex = 0;
  let fallbackDiff = Infinity;

  rows.forEach((row, index) => {
    const ts = toTimestampMs(row.timestamp);
    if (!Number.isFinite(ts)) {
      return;
    }

    if (ts >= targetMs && ts < earliestForwardTs) {
      earliestForwardTs = ts;
      startIndex = index;
    }

    const diff = Math.abs(ts - targetMs);
    if (diff < fallbackDiff) {
      fallbackDiff = diff;
      fallbackIndex = index;
    }
  });

  // At the very end of the file there is no forward row left; fall back to the
  // closest one so the endpoint still returns a usable window.
  const window = rows.slice(startIndex >= 0 ? startIndex : fallbackIndex);

  return window.map((row, index) => ({
    ...row,
    time_segment: 'future',
    step_ahead: index + 1,
    // step_ahead k maps onto anchor + k * step, matching rebaseFutureTimestamps.
    timestamp: formatTimestamp(new Date(targetMs + index * stepMs))
  }));
};

const readForecastRowsLive = (city) => {
  const year = config.modelForecast.forecastYear || '2026';
  const rows = readForecastRows(city, year);
  if (year === '2026') {
    return anchor2026Rows(rows);
  }
  return rebaseFutureTimestamps(rows);
};

const classifyTraffic = (trafficFlow) => {
  if (trafficFlow === null) {
    return 'unknown';
  }

  if (trafficFlow >= 20000) {
    return 'severe';
  }

  if (trafficFlow >= 12000) {
    return 'heavy';
  }

  if (trafficFlow >= 6000) {
    return 'moderate';
  }

  return 'light';
};

const classifyWeather = ({ temperature, humidity }) => {
  if (temperature !== null && temperature >= 38) {
    return { main: 'Heatwave', description: 'model forecast heatwave' };
  }

  if (humidity !== null && humidity >= 82) {
    return { main: 'Rain', description: 'model forecast high humidity rain risk' };
  }

  return { main: 'Clear', description: 'model forecast stable weather' };
};

const getLatestModelConditions = ({ city = config.modelForecast.defaultCity, stepAhead = 1 } = {}) => {
  const rows = config.modelForecast.liveRebase
    ? readForecastRowsLive(city)
    : readForecastRows(city);
  const futureRows = rows.filter((row) => row.time_segment === 'future');
  const selected = futureRows.find((row) => Number(row.step_ahead) === stepAhead) || futureRows[0] || rows[rows.length - 1];

  if (!selected) {
    throw new Error(`No model forecast rows available for ${city}`);
  }

  const aqi = toNumber(selected.aqi);
  const temperature = toNumber(selected.temperature);
  const humidity = toNumber(selected.humidity);
  const trafficFlow = toNumber(selected.traffic_flow);
  const trafficLevel = classifyTraffic(trafficFlow);
  const weather = classifyWeather({ temperature, humidity });

  return {
    city,
    source: 'trained_model_forecast',
    forecastFor: selected.timestamp,
    stepAhead: Number(selected.step_ahead || stepAhead),
    aqi: {
      provider: 'ForeSightXModel',
      aqi,
      observedAt: selected.timestamp
    },
    weather: {
      provider: 'ForeSightXModel',
      weather,
      temperature: {
        value: temperature,
        units: 'metric'
      },
      humidity,
      rainLastHourMm: weather.main === 'Rain' ? 8 : 0,
      observedAt: selected.timestamp
    },
    traffic: {
      provider: 'ForeSightXModel',
      congestionLevel: trafficLevel,
      flow: trafficFlow,
      observedAt: selected.timestamp
    },
    raw: selected
  };
};

const getForecastSeries = ({ city = config.modelForecast.defaultCity, steps = 24 } = {}) => {
  const rows = config.modelForecast.liveRebase
    ? readForecastRowsLive(city)
    : readForecastRows(city);
  const futureRows = rows
    .filter((row) => row.time_segment === 'future')
    .sort((a, b) => Number(a.step_ahead) - Number(b.step_ahead))
    .slice(0, steps);

  return futureRows.map((row) => ({
    timestamp: row.timestamp,
    stepAhead: Number(row.step_ahead || 0),
    trafficFlow: toNumber(row.traffic_flow),
    aqi: toNumber(row.aqi),
    temperature: toNumber(row.temperature),
    humidity: toNumber(row.humidity),
    electricityDemand: toNumber(row.electricity_demand),
    weather: classifyWeather({
      temperature: toNumber(row.temperature),
      humidity: toNumber(row.humidity)
    }),
    traffic: {
      congestionLevel: classifyTraffic(toNumber(row.traffic_flow))
    }
  }));
};

module.exports = {
  getLatestModelConditions,
  getForecastSeries,
  readForecastRows
};
