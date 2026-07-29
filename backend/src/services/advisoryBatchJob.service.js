const config = require('../config/env');
const userRepository = require('../repositories/user.repository');
const executionLogRepository = require('../repositories/executionLog.repository');
const dataProviders = require('./dataProviders.service');
const advisoryGenerator = require('./advisoryGenerator.service');
const notificationService = require('./notification.service');
const modelForecastService = require('./modelForecast.service');

const jobName = 'advisory_batch';

const getUserCoordinates = (user) => {
  const latitude = user.latitude === null || user.latitude === undefined ? null : Number(user.latitude);
  const longitude = user.longitude === null || user.longitude === undefined ? null : Number(user.longitude);

  if (Number.isFinite(latitude) && Number.isFinite(longitude)) {
    return { latitude, longitude };
  }

  return null;
};

const getTrafficDestination = (user) => {
  const commute = user.preferences?.commuteDestination || user.preferences?.trafficDestination;

  if (commute?.latitude !== undefined && commute?.longitude !== undefined) {
    return {
      latitude: Number(commute.latitude),
      longitude: Number(commute.longitude)
    };
  }

  if (commute?.address) {
    return { address: commute.address };
  }

  const fallback = config.jobs.defaultTrafficDestination;

  if (fallback.address) {
    return { address: fallback.address };
  }

  if (Number.isFinite(fallback.latitude) && Number.isFinite(fallback.longitude)) {
    return {
      latitude: fallback.latitude,
      longitude: fallback.longitude
    };
  }

  return null;
};

const fetchLiveConditions = async (user) => {
  const coordinates = getUserCoordinates(user);
  const conditions = {};
  const errors = [];

  if (config.modelForecast.enabled) {
    try {
      const modelConditions = modelForecastService.getLatestModelConditions({
        city: String(user.city || config.modelForecast.defaultCity).toLowerCase()
      });

      return {
        conditions: {
          aqi: modelConditions.aqi,
          weather: modelConditions.weather,
          traffic: modelConditions.traffic
        },
        errors,
        source: modelConditions.source,
        forecastFor: modelConditions.forecastFor
      };
    } catch (error) {
      errors.push({ provider: 'trained_model_forecast', message: error.message });
    }
  }

  try {
    conditions.aqi = coordinates
      ? await dataProviders.fetchAqi(coordinates)
      : await dataProviders.fetchAqi({ city: user.city });
  } catch (error) {
    errors.push({ provider: 'aqi', message: error.message, details: error.details });
  }

  try {
    conditions.weather = coordinates
      ? await dataProviders.fetchCurrentWeather(coordinates)
      : await dataProviders.fetchCurrentWeather({ city: user.city });
  } catch (error) {
    errors.push({ provider: 'weather', message: error.message, details: error.details });
  }

  const destination = getTrafficDestination(user);

  if (coordinates && destination) {
    try {
      conditions.traffic = await dataProviders.fetchTrafficRoute({
        origin: coordinates,
        destination,
        routingPreference: 'TRAFFIC_AWARE'
      });
    } catch (error) {
      errors.push({ provider: 'traffic', message: error.message, details: error.details });
    }
  }

  return { conditions, errors, source: 'external_live_apis' };
};

const countSentNotifications = (deliveryResult) => (
  deliveryResult.results || []
).reduce((total, advisoryResult) => (
  total + (advisoryResult.results || []).filter((channelResult) => (
    channelResult.status === 'sent' || channelResult.status === 'delivered'
  )).length
), 0);

const processUser = async (user, { channels, sharedConditions } = {}) => {
  const liveResult = sharedConditions
    ? {
        conditions: sharedConditions,
        errors: [],
        source: 'manual_console_conditions',
        forecastFor: null
      }
    : await fetchLiveConditions(user);
  const { conditions, errors, source, forecastFor } = liveResult;
  const generated = advisoryGenerator.generateForUser({
    user,
    riskGroups: user.groups || [],
    ...conditions
  });

  if (!generated.advisories.length) {
    return {
      userId: user.id,
      advisoriesGenerated: 0,
      notificationsSent: 0,
      providerErrors: errors,
      source,
      forecastFor,
      delivery: null
    };
  }

  const delivery = await notificationService.deliverAdvisories({
    user,
    advisories: generated.advisories,
    channels
  });

  return {
    userId: user.id,
    advisoriesGenerated: generated.advisories.length,
    notificationsSent: countSentNotifications(delivery),
    providerErrors: errors,
    source,
    forecastFor,
    delivery
  };
};

const processAllUsers = async ({
  batchSize = config.jobs.advisoryBatchSize,
  startedBy = 'node-cron',
  channels,
  sharedConditions
} = {}) => {
  const execution = await executionLogRepository.create({
    jobName,
    metadata: {
      batchSize,
      startedBy,
      channels,
      sharedConditions
    }
  });

  let offset = 0;
  let usersProcessed = 0;
  let advisoriesGenerated = 0;
  let notificationsSent = 0;
  const failures = [];
  const providerErrors = [];

  try {
    while (true) {
      const users = await userRepository.findActiveWithRiskGroups({ limit: batchSize, offset });

      if (!users.length) {
        break;
      }

      for (const user of users) {
        try {
          const result = await processUser(user, { channels, sharedConditions });
          usersProcessed += 1;
          advisoriesGenerated += result.advisoriesGenerated;
          notificationsSent += result.notificationsSent;

          if (result.providerErrors.length) {
            providerErrors.push({
              userId: user.id,
              errors: result.providerErrors
            });
          }
        } catch (error) {
          usersProcessed += 1;
          failures.push({
            userId: user.id,
            message: error.message,
            details: error.details
          });
        }
      }

      offset += users.length;
    }

    const status = failures.length ? 'partial' : 'success';

    return executionLogRepository.complete({
      id: execution.id,
      startedAt: execution.started_at,
      status,
      usersProcessed,
      advisoriesGenerated,
      notificationsSent,
      errorMessage: failures.length ? `${failures.length} users failed` : null,
      metadata: {
        failures,
        providerErrors
      }
    });
  } catch (error) {
    return executionLogRepository.complete({
      id: execution.id,
      startedAt: execution.started_at,
      status: 'failed',
      usersProcessed,
      advisoriesGenerated,
      notificationsSent,
      errorMessage: error.message,
      metadata: {
        failures,
        providerErrors,
        fatal: {
          message: error.message,
          details: error.details
        }
      }
    });
  }
};

module.exports = {
  processAllUsers,
  processUser,
  fetchLiveConditions
};
