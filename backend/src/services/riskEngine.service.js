const groupAliases = {
  respiratory: 'resp',
  resp: 'resp',
  elderly: 'elder',
  elder: 'elder',
  children: 'child',
  child: 'child',
  outdoor_workers: 'worker',
  outdoor_worker: 'worker',
  worker: 'worker',
  commuters: 'commuter',
  commuter: 'commuter',
  general: 'general'
};

const normalizeGroupKey = (group) => {
  if (typeof group === 'string') {
    return groupAliases[group.toLowerCase()] || group.toLowerCase();
  }

  if (group?.group_key) {
    return normalizeGroupKey(group.group_key);
  }

  return null;
};

const normalizeRiskGroups = (groups = []) => new Set(
  groups.map(normalizeGroupKey).filter(Boolean)
);

const getAqiValue = (aqi) => {
  if (typeof aqi === 'number') {
    return aqi;
  }

  return Number.isFinite(Number(aqi?.aqi)) ? Number(aqi.aqi) : null;
};

const getAqiCategory = (aqi) => {
  const value = getAqiValue(aqi);

  if (value === null) {
    return 'unknown';
  }

  if (value <= 50) {
    return 'good';
  }

  if (value <= 100) {
    return 'moderate';
  }

  if (value <= 200) {
    return 'poor';
  }

  if (value <= 300) {
    return 'very_poor';
  }

  return 'severe';
};

const getTemperatureCelsius = (weather, temperature) => {
  if (typeof temperature === 'number') {
    return temperature;
  }

  if (typeof weather?.temperature?.value === 'number') {
    return weather.temperature.value;
  }

  if (typeof weather?.temp === 'number') {
    return weather.temp;
  }

  return null;
};

const textIncludesAny = (value, keywords) => {
  const text = String(value || '').toLowerCase();
  return keywords.some((keyword) => text.includes(keyword));
};

const getWeatherCondition = (weather = {}) => {
  const main = weather.weather?.main || weather.main || weather.condition || '';
  const description = weather.weather?.description || weather.description || '';
  const rainLastHourMm = weather.rainLastHourMm ?? weather.rain?.['1h'] ?? 0;
  const temperature = getTemperatureCelsius(weather);

  if (temperature !== null && temperature >= 38) {
    return 'heatwave';
  }

  if (temperature !== null && temperature <= 5) {
    return 'coldwave';
  }

  if (Number(rainLastHourMm) >= 7.6 || textIncludesAny(`${main} ${description}`, ['heavy rain', 'thunderstorm', 'storm'])) {
    return 'heavy_rain';
  }

  if (textIncludesAny(`${main} ${description}`, ['rain', 'drizzle'])) {
    return 'rain';
  }

  if (textIncludesAny(`${main} ${description}`, ['fog', 'mist', 'haze', 'smoke'])) {
    return 'low_visibility';
  }

  return main ? main.toLowerCase() : 'normal';
};

const hasRainHazard = (weather = {}) => {
  const main = weather.weather?.main || weather.main || weather.condition || '';
  const description = weather.weather?.description || weather.description || '';
  const rainLastHourMm = weather.rainLastHourMm ?? weather.rain?.['1h'] ?? 0;

  return Number(rainLastHourMm) > 0 || textIncludesAny(`${main} ${description}`, ['rain', 'drizzle', 'thunderstorm', 'storm']);
};

const getTrafficLevel = (traffic = {}) => {
  if (typeof traffic === 'string') {
    return traffic.toLowerCase();
  }

  return (traffic.congestionLevel || traffic.level || 'unknown').toLowerCase();
};

const getRiskLevel = (score) => {
  if (score >= 8) {
    return 'High';
  }

  if (score >= 5) {
    return 'Medium';
  }

  return 'Low';
};

const getSeverity = (riskLevel, factors) => {
  if (factors.some((factor) => factor.severity === 'critical') || riskLevel === 'High') {
    return 'critical';
  }

  if (riskLevel === 'Medium' || factors.some((factor) => factor.severity === 'warning')) {
    return 'warning';
  }

  if (factors.length) {
    return 'low';
  }

  return 'info';
};

const addFactor = (factors, { code, score, severity, title, description }) => {
  factors.push({
    code,
    score,
    severity,
    title,
    description
  });
};

const calculateRisk = ({ user = {}, riskGroups = [], aqi, weather = {}, temperature, traffic = {} } = {}) => {
  const providedGroups = Array.isArray(riskGroups) ? riskGroups : [];
  const groups = normalizeRiskGroups(providedGroups.length ? providedGroups : user.groups);
  const aqiValue = getAqiValue(aqi);
  const aqiCategory = getAqiCategory(aqi);
  const temperatureCelsius = getTemperatureCelsius(weather, temperature);
  const weatherCondition = getWeatherCondition({ ...weather, temperature: { ...weather.temperature, value: temperatureCelsius } });
  const rainHazard = hasRainHazard(weather);
  const trafficLevel = getTrafficLevel(traffic);
  const numericAge = Number(user.age);
  const age = Number.isInteger(numericAge) ? numericAge : null;
  const factors = [];

  if (aqiCategory === 'very_poor' || aqiCategory === 'severe') {
    addFactor(factors, {
      code: 'aqi_very_poor',
      score: 3,
      severity: 'critical',
      title: 'Very Poor AQI',
      description: 'Air quality is very poor; outdoor exposure should be reduced.'
    });
  } else if (aqiCategory === 'poor') {
    addFactor(factors, {
      code: 'aqi_poor',
      score: 1.5,
      severity: 'warning',
      title: 'Poor AQI',
      description: 'Air quality is poor and may affect sensitive groups.'
    });
  }

  if ((weatherCondition === 'heatwave' || weatherCondition === 'coldwave') && (groups.has('elder') || age > 60)) {
    addFactor(factors, {
      code: 'elder_extreme_temperature',
      score: 2,
      severity: 'critical',
      title: 'Extreme Weather Risk',
      description: 'Extreme temperature conditions are elevated risk for elderly citizens.'
    });
  }

  if ((rainHazard || weatherCondition === 'heavy_rain' || weatherCondition === 'rain') && trafficLevel === 'severe') {
    addFactor(factors, {
      code: 'rain_severe_traffic',
      score: 3,
      severity: 'critical',
      title: 'Severe Traffic With Rain',
      description: 'Severe congestion and rain can cause unsafe or delayed commutes.'
    });
  }

  if (groups.has('resp') && (aqiCategory === 'very_poor' || aqiCategory === 'severe')) {
    addFactor(factors, {
      code: 'respiratory_very_poor_aqi',
      score: 2,
      severity: 'critical',
      title: 'Respiratory AQI Alert',
      description: 'Very poor AQI is high risk for users with respiratory sensitivity.'
    });
  }

  if (groups.has('child') && (aqiCategory === 'very_poor' || aqiCategory === 'severe')) {
    addFactor(factors, {
      code: 'child_very_poor_aqi',
      score: 2,
      severity: 'critical',
      title: 'School Indoor Recess Alert',
      description: 'Children should avoid outdoor activity while AQI is very poor.'
    });
  }

  if (groups.has('worker') && (aqiCategory === 'very_poor' || aqiCategory === 'severe')) {
    addFactor(factors, {
      code: 'worker_very_poor_aqi',
      score: 2,
      severity: 'critical',
      title: 'Field Safety Alert',
      description: 'Outdoor workers should use PPE and take frequent breaks.'
    });
  }

  if (groups.has('commuter') && trafficLevel === 'severe' && (rainHazard || weatherCondition === 'heavy_rain' || weatherCondition === 'rain')) {
    addFactor(factors, {
      code: 'commuter_rain_severe_traffic',
      score: 2,
      severity: 'critical',
      title: 'Commute Alert',
      description: 'Severe traffic and rain may require extra travel time or route changes.'
    });
  }

  const score = Number(factors.reduce((sum, factor) => sum + factor.score, 0).toFixed(2));
  const riskLevel = getRiskLevel(score);
  const severity = getSeverity(riskLevel, factors);

  return {
    userId: user.id || user.user_id || null,
    score,
    riskLevel,
    severity,
    inputs: {
      aqi: {
        value: aqiValue,
        category: aqiCategory
      },
      weather: {
        condition: weatherCondition,
        temperatureCelsius,
        rainHazard
      },
      traffic: {
        level: trafficLevel
      },
      riskGroups: Array.from(groups)
    },
    factors,
    advisories: factors.map((factor) => ({
      title: factor.title,
      description: factor.description,
      severity: factor.severity,
      sourceFactor: factor.code
    }))
  };
};

module.exports = {
  calculateRisk,
  getAqiCategory,
  getRiskLevel,
  getTrafficLevel,
  getWeatherCondition,
  normalizeRiskGroups
};
