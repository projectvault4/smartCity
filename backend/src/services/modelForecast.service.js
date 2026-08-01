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

const readForecastRows = (city = config.modelForecast.defaultCity) => {
  const cityKey = String(city || '').toLowerCase();
  // The default-city training run writes to the repository root outputs/ dir
  // (shared with the anomaly + multivariate pipelines). Only cities with a
  // dedicated subdirectory (e.g. delhi) resolve into outputs/<city>/.
  const forecastPath = ['delhi'].includes(cityKey)
    ? path.join(
        config.modelForecast.projectRoot,
        'outputs',
        cityKey,
        'past_present_future_forecast.csv'
      )
    : path.join(
        config.modelForecast.projectRoot,
        'outputs',
        'past_present_future_forecast.csv'
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
  const rows = readForecastRows(city);
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
  const rows = readForecastRows(city);
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
