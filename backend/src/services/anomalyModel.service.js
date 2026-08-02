const fs = require('fs');
const path = require('path');

const config = require('../config/env');

const artifactSuffix = () => {
  const year = config.modelForecast.forecastYear || '2026';
  return year === '2026' ? '2026' : '';
};

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

const readEvents = () => {
  const suffix = artifactSuffix();
  const eventsPath = path.join(
    config.modelForecast.projectRoot,
    'outputs',
    `urban_events${suffix ? `_${suffix}` : ''}.json`
  );

  if (!fs.existsSync(eventsPath)) {
    throw new Error(`Model anomaly events file not found: ${eventsPath}`);
  }

  return JSON.parse(fs.readFileSync(eventsPath, 'utf8'));
};

const readTimelineRows = () => {
  const suffix = artifactSuffix();
  const timelinePath = path.join(
    config.modelForecast.projectRoot,
    'outputs',
    `urban_anomaly_timeline${suffix ? `_${suffix}` : ''}.csv`
  );

  if (!fs.existsSync(timelinePath)) {
    throw new Error(`Model anomaly timeline file not found: ${timelinePath}`);
  }

  const [headerLine, ...lines] = fs.readFileSync(timelinePath, 'utf8').trim().split(/\r?\n/);
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

const keyOfTimestamp = (timestamp) => String(timestamp).slice(0, 10);

const formatLabel = (timestamp) => {
  const [year, month, day] = String(timestamp).slice(0, 10).split('-');
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${months[Number(month) - 1]} ${Number(day)}, ${year}`;
};

const mapSeverity = (severity) => {
  const normalized = String(severity || '').toLowerCase();
  if (normalized === 'critical' || normalized === 'high') {
    return 'critical';
  }
  if (normalized === 'medium') {
    return 'medium';
  }
  return 'normal';
};

const scoreSeverity = (score) => {
  if (score >= 0.85) return 'critical';
  if (score >= 0.6) return 'medium';
  return 'normal';
};

const featureNameFor = (feature) => {
  const name = String(feature || '');
  if (/aqi/i.test(name)) return 'AQI';
  if (/traffic/i.test(name)) return 'Traffic';
  if (/electricity|energy|power/i.test(name)) return 'Electricity';
  if (/temperature|temp/i.test(name)) return 'Temperature';
  if (/humid/i.test(name)) return 'Humidity';
  return name.split(' ').slice(0, 2).join(' ');
};

const getDailySeries = (rows) => {
  const daily = new Map();

  rows.forEach((row) => {
    const key = keyOfTimestamp(row.timestamp);
    const score = toNumber(row.anomaly_score);
    if (score === null) {
      return;
    }

    const existing = daily.get(key);
    if (!existing) {
      daily.set(key, {
        key,
        date: key,
        score,
        severity: scoreSeverity(score)
      });
      return;
    }

    if (score > existing.score) {
      existing.score = score;
      existing.severity = scoreSeverity(score);
    }
  });

  return Array.from(daily.values())
    .sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0))
    .map((entry, index) => {
      const [year, month] = entry.key.split('-');
      return {
        ...entry,
        month: Number(month) - 1,
        year: Number(year),
        z: Math.round(((Math.sin(index * 12.9898) * 43758.5453) % 1) * 72 - 36)
      };
    });
};

const buildAnomalies = (events) => {
  // Collapse per-hour events to one per day (highest score) so the list,
  // the calendar heatmap and the 3D day-sphere model all describe the same
  // daily series instead of disagreeing on granularity.
  const byDay = new Map();
  events.forEach((event) => {
    const key = keyOfTimestamp(event.timestamp);
    const score = toNumber(event.anomaly_score) || 0;
    const existing = byDay.get(key);
    if (!existing || score > (toNumber(existing.anomaly_score) || 0)) {
      byDay.set(key, { ...event, _key: key });
    }
  });

  return Array.from(byDay.values())
    .sort((a, b) => (toNumber(b.anomaly_score) || 0) - (toNumber(a.anomaly_score) || 0))
    .map((event) => {
      const features = (event.drivers || [])
        .slice(0, 3)
        .map((driver) => ({
          name: featureNameFor(driver.feature),
          dir: 'up',
          pct: Math.min(100, Math.round(Math.abs(toNumber(driver.contribution) || 0) * 22))
        }));

      return {
        key: event._key,
        label: formatLabel(event.timestamp),
        score: Math.round((toNumber(event.anomaly_score) || 0) * 100) / 100,
        severity: mapSeverity(event.severity),
        features,
        tags: [event.event_type, 'Model-detected'],
        shap: (event.drivers || [])
          .slice(0, 5)
          .map((driver) => ({
            f: featureNameFor(driver.feature),
            v: Math.round((toNumber(driver.contribution) || 0) * 100) / 100
          })),
        interp: event.description || `Model detected a ${event.severity} anomaly (score ${event.anomaly_score}).`
      };
    });
};

const getAnomalyDashboard = () => {
  const events = readEvents();
  const rows = readTimelineRows();

  return {
    source: 'trained_model_anomaly_outputs',
    anomalies: buildAnomalies(events),
    daily: getDailySeries(rows)
  };
};

module.exports = {
  getAnomalyDashboard,
  readEvents,
  readTimelineRows
};
