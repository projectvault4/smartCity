const config = require('../config/env');
const { createHttpClient } = require('../utils/httpClient');
const ProviderError = require('../utils/ProviderError');

const provider = 'WAQI';
const client = createHttpClient({
  provider,
  baseURL: config.providers.waqi.baseUrl
});

const requireApiKey = () => {
  if (!config.providers.waqi.apiKey) {
    throw new ProviderError(provider, 'WAQI_API_KEY is required');
  }
};

const normalizePollutants = (iaqi = {}) => Object.entries(iaqi).reduce((pollutants, [key, value]) => {
  pollutants[key] = value?.v ?? null;
  return pollutants;
}, {});

const normalizeAqiResponse = (payload) => {
  if (payload.status !== 'ok') {
    throw new ProviderError(provider, 'WAQI returned an unsuccessful response', {
      status: payload.status,
      reason: payload.reason || payload.data
    });
  }

  const data = payload.data || {};
  const [latitude, longitude] = data.city?.geo || [];

  return {
    provider,
    location: {
      name: data.city?.name || null,
      url: data.city?.url || null,
      latitude: latitude !== undefined ? Number(latitude) : null,
      longitude: longitude !== undefined ? Number(longitude) : null
    },
    aqi: Number.isFinite(Number(data.aqi)) ? Number(data.aqi) : null,
    dominantPollutant: data.dominentpol || data.dominantpol || null,
    pollutants: normalizePollutants(data.iaqi),
    observedAt: data.time?.iso || data.time?.s || null,
    forecast: data.forecast || null,
    attribution: data.attributions || [],
    raw: data
  };
};

const buildFeedPath = ({ city, latitude, longitude, stationId }) => {
  if (stationId) {
    return `/feed/@${encodeURIComponent(stationId)}/`;
  }

  if (latitude !== undefined && longitude !== undefined) {
    return `/feed/geo:${latitude};${longitude}/`;
  }

  if (city) {
    return `/feed/${encodeURIComponent(city)}/`;
  }

  return '/feed/here/';
};

const fetchAqi = async ({ city, latitude, longitude, stationId } = {}) => {
  requireApiKey();

  const response = await client.get(buildFeedPath({ city, latitude, longitude, stationId }), {
    params: {
      token: config.providers.waqi.apiKey
    }
  });

  return normalizeAqiResponse(response.data);
};

module.exports = {
  fetchAqi,
  normalizeAqiResponse
};
