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

const readYearlyForecastRows = (city = config.modelForecast.defaultCity, year = 2026) => {
  const cityKey = String(city || '').toLowerCase();
  const forecastPath = ['delhi'].includes(cityKey)
    ? path.join(
        config.modelForecast.projectRoot,
        'outputs',
        cityKey,
        `forecast_${year}.csv`
      )
    : path.join(
        config.modelForecast.projectRoot,
        'outputs',
        `forecast_${year}.csv`
      );

  if (!fs.existsSync(forecastPath)) {
    throw new Error(`2026 model forecast file not found: ${forecastPath}`);
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

const round = (value, decimals = 2) => {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return null;
  const factor = 10 ** decimals;
  return Math.round(Number(value) * factor) / factor;
};

const monthLabel = (month) => {
  const names = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];
  return names[month - 1] || `Month ${month}`;
};

const mean = (values) => {
  const clean = values.filter((v) => v !== null && Number.isFinite(Number(v)));
  if (clean.length === 0) return null;
  return clean.reduce((s, v) => s + Number(v), 0) / clean.length;
};

const median = (values) => {
  const clean = values.filter((v) => v !== null && Number.isFinite(Number(v))).map(Number).sort((a, b) => a - b);
  if (clean.length === 0) return null;
  const mid = Math.floor(clean.length / 2);
  return clean.length % 2 === 0 ? (clean[mid - 1] + clean[mid]) / 2 : clean[mid];
};

const getYearlyForecast = ({ city = config.modelForecast.defaultCity, year = 2026, granularity = 'monthly' } = {}) => {
  const rows = readYearlyForecastRows(city, year);

  if (rows.length === 0) {
    throw new Error(`No forecast rows found for ${year}`);
  }

  const monthlyAgg = Array.from({ length: 12 }, (_, i) => i + 1).map((month) => {
    const monthRows = rows.filter((row) => {
      const ts = new Date(String(row.timestamp).replace(' ', 'T'));
      return ts.getMonth() === month - 1;
    });

    const pick = (col) => monthRows.map((row) => toNumber(row[col]));
    return {
      month,
      label: monthLabel(month),
      days: monthRows.length / 24,
      trafficFlow: round(mean(pick('traffic_flow'))),
      aqi: round(mean(pick('aqi'))),
      temperature: round(mean(pick('temperature'))),
      humidity: round(mean(pick('humidity'))),
      electricityDemand: round(mean(pick('electricity_demand'))),
      maxTrafficFlow: round(Math.max(...pick('traffic_flow').filter((v) => v !== null)), 0),
      maxAqi: round(Math.max(...pick('aqi').filter((v) => v !== null)), 0),
      maxTemperature: round(Math.max(...pick('temperature').filter((v) => v !== null)), 0)
    };
  });

  const annualPick = (col) => rows.map((row) => toNumber(row[col]));
  const annual = {
    trafficFlow: round(mean(annualPick('traffic_flow'))),
    aqi: round(mean(annualPick('aqi'))),
    temperature: round(mean(annualPick('temperature'))),
    humidity: round(mean(annualPick('humidity'))),
    electricityDemand: round(mean(annualPick('electricity_demand')))
  };

  const peak = (col, pickMax = true) => {
    const values = monthlyAgg.map((m) => ({ ...m, value: m[col] }));
    return values.reduce((best, m) => (m.value !== null && (pickMax ? m.value > best.value : m.value < best.value) ? m : best), values[0]);
  };

  const series = rows.map((row) => ({
    timestamp: row.timestamp,
    stepAhead: toNumber(row.step_ahead),
    trafficFlow: round(toNumber(row.traffic_flow), 0),
    aqi: round(toNumber(row.aqi), 1),
    temperature: round(toNumber(row.temperature), 1),
    humidity: round(toNumber(row.humidity), 1),
    electricityDemand: round(toNumber(row.electricity_demand), 1)
  }));

  return {
    year,
    city,
    source: 'trained_model_2026_forecast',
    generatedFrom: rows[0].timestamp,
    through: rows[rows.length - 1].timestamp,
    totalHours: rows.length,
    annual,
    peaks: {
      peakTrafficMonth: peak('trafficFlow'),
      peakAqiMonth: peak('aqi'),
      peakTemperatureMonth: peak('maxTemperature'),
      hottestMonth: peak('temperature'),
      coolestMonth: peak('temperature', false)
    },
    monthly: granularity === 'monthly' ? monthlyAgg : series,
    series
  };
};

module.exports = {
  getYearlyForecast,
  readYearlyForecastRows
};
