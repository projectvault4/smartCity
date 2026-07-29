const config = require('../config/env');
const { createHttpClient } = require('../utils/httpClient');
const ProviderError = require('../utils/ProviderError');

const provider = 'OpenWeather';
const client = createHttpClient({
  provider,
  baseURL: config.providers.openWeather.baseUrl
});

const requireApiKey = () => {
  if (!config.providers.openWeather.apiKey) {
    throw new ProviderError(provider, 'OPENWEATHER_API_KEY is required');
  }
};

const normalizeWeatherResponse = (data) => ({
  provider,
  location: {
    id: data.id || null,
    name: data.name || null,
    country: data.sys?.country || null,
    latitude: data.coord?.lat ?? null,
    longitude: data.coord?.lon ?? null,
    timezoneSeconds: data.timezone ?? null
  },
  weather: {
    conditionId: data.weather?.[0]?.id ?? null,
    main: data.weather?.[0]?.main || null,
    description: data.weather?.[0]?.description || null,
    icon: data.weather?.[0]?.icon || null
  },
  temperature: {
    value: data.main?.temp ?? null,
    feelsLike: data.main?.feels_like ?? null,
    min: data.main?.temp_min ?? null,
    max: data.main?.temp_max ?? null,
    units: config.providers.openWeather.units
  },
  pressure: data.main?.pressure ?? null,
  humidity: data.main?.humidity ?? null,
  visibility: data.visibility ?? null,
  wind: {
    speed: data.wind?.speed ?? null,
    degrees: data.wind?.deg ?? null,
    gust: data.wind?.gust ?? null
  },
  rainLastHourMm: data.rain?.['1h'] ?? null,
  snowLastHourMm: data.snow?.['1h'] ?? null,
  cloudinessPercent: data.clouds?.all ?? null,
  observedAt: data.dt ? new Date(data.dt * 1000).toISOString() : null,
  sunriseAt: data.sys?.sunrise ? new Date(data.sys.sunrise * 1000).toISOString() : null,
  sunsetAt: data.sys?.sunset ? new Date(data.sys.sunset * 1000).toISOString() : null,
  raw: data
});

const fetchCurrentWeather = async ({ latitude, longitude, city, units, lang } = {}) => {
  requireApiKey();

  if ((latitude === undefined || longitude === undefined) && !city) {
    throw new ProviderError(provider, 'latitude and longitude or city is required');
  }

  const params = {
    appid: config.providers.openWeather.apiKey,
    units: units || config.providers.openWeather.units
  };

  if (lang) {
    params.lang = lang;
  }

  if (latitude !== undefined && longitude !== undefined) {
    params.lat = latitude;
    params.lon = longitude;
  } else {
    params.q = city;
  }

  const response = await client.get('/data/2.5/weather', { params });

  return normalizeWeatherResponse(response.data);
};

module.exports = {
  fetchCurrentWeather,
  normalizeWeatherResponse
};
