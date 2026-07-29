const config = require('../config/env');
const { createHttpClient } = require('../utils/httpClient');
const ProviderError = require('../utils/ProviderError');

const provider = 'GoogleTraffic';
const client = createHttpClient({
  provider,
  baseURL: config.providers.googleTraffic.baseUrl
});

const requireApiKey = () => {
  if (!config.providers.googleTraffic.apiKey) {
    throw new ProviderError(provider, 'GOOGLE_MAPS_API_KEY is required');
  }
};

const parseDurationSeconds = (duration) => {
  if (!duration || typeof duration !== 'string') {
    return null;
  }

  const seconds = Number(duration.replace(/s$/, ''));
  return Number.isFinite(seconds) ? seconds : null;
};

const normalizeWaypoint = (point) => {
  if (point.address) {
    return { address: point.address };
  }

  if (point.placeId) {
    return { placeId: point.placeId };
  }

  return {
    location: {
      latLng: {
        latitude: point.latitude,
        longitude: point.longitude
      }
    }
  };
};

const getCongestionLevel = (trafficDelaySeconds) => {
  if (trafficDelaySeconds === null || trafficDelaySeconds <= 0) {
    return 'unknown';
  }

  if (trafficDelaySeconds < 300) {
    return 'light';
  }

  if (trafficDelaySeconds < 900) {
    return 'moderate';
  }

  if (trafficDelaySeconds < 1800) {
    return 'heavy';
  }

  return 'severe';
};

const normalizeTrafficResponse = (data) => {
  const route = data.routes?.[0] || {};
  const durationSeconds = parseDurationSeconds(route.duration);
  const staticDurationSeconds = parseDurationSeconds(route.staticDuration);
  const trafficDelaySeconds = durationSeconds !== null && staticDurationSeconds !== null
    ? Math.max(durationSeconds - staticDurationSeconds, 0)
    : null;

  return {
    provider,
    distanceMeters: route.distanceMeters ?? null,
    durationSeconds,
    staticDurationSeconds,
    trafficDelaySeconds,
    congestionLevel: getCongestionLevel(trafficDelaySeconds),
    legs: (route.legs || []).map((leg) => {
      const legDurationSeconds = parseDurationSeconds(leg.duration);
      const legStaticDurationSeconds = parseDurationSeconds(leg.staticDuration);
      const legTrafficDelaySeconds = legDurationSeconds !== null && legStaticDurationSeconds !== null
        ? Math.max(legDurationSeconds - legStaticDurationSeconds, 0)
        : null;

      return {
        distanceMeters: leg.distanceMeters ?? null,
        durationSeconds: legDurationSeconds,
        staticDurationSeconds: legStaticDurationSeconds,
        trafficDelaySeconds: legTrafficDelaySeconds,
        congestionLevel: getCongestionLevel(legTrafficDelaySeconds)
      };
    }),
    raw: data
  };
};

const fetchTrafficRoute = async ({
  origin,
  destination,
  intermediates,
  routingPreference = 'TRAFFIC_AWARE',
  trafficModel,
  departureTime
}) => {
  requireApiKey();

  if (!origin || !destination) {
    throw new ProviderError(provider, 'origin and destination are required');
  }

  const payload = {
    origin: normalizeWaypoint(origin),
    destination: normalizeWaypoint(destination),
    travelMode: 'DRIVE',
    routingPreference
  };

  if (Array.isArray(intermediates) && intermediates.length) {
    payload.intermediates = intermediates.map(normalizeWaypoint);
  }

  if (trafficModel) {
    payload.trafficModel = trafficModel;
  }

  if (departureTime) {
    payload.departureTime = departureTime;
  }

  const response = await client.post('/directions/v2:computeRoutes', payload, {
    headers: {
      'X-Goog-Api-Key': config.providers.googleTraffic.apiKey,
      'X-Goog-FieldMask': config.providers.googleTraffic.fieldMask
    }
  });

  return normalizeTrafficResponse(response.data);
};

module.exports = {
  fetchTrafficRoute,
  normalizeTrafficResponse
};
