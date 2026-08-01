const path = require('path');
const dotenv = require('dotenv');

dotenv.config({ path: path.resolve(__dirname, '../../.env') });

const config = {
  nodeEnv: process.env.NODE_ENV || 'development',
  port: Number(process.env.PORT) || 5000,
  clientOrigin: (process.env.CLIENT_ORIGIN || 'http://localhost:3000,http://localhost:3001')
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean),
  http: {
    timeoutMs: Number(process.env.HTTP_TIMEOUT_MS) || 8000,
    retries: Number(process.env.HTTP_RETRIES) || 2,
    retryDelayMs: Number(process.env.HTTP_RETRY_DELAY_MS) || 500
  },
  providers: {
    waqi: {
      baseUrl: process.env.WAQI_BASE_URL || 'https://api.waqi.info',
      apiKey: process.env.WAQI_API_KEY || ''
    },
    openWeather: {
      baseUrl: process.env.OPENWEATHER_BASE_URL || 'https://api.openweathermap.org',
      apiKey: process.env.OPENWEATHER_API_KEY || '',
      units: process.env.OPENWEATHER_UNITS || 'metric'
    },
    googleTraffic: {
      baseUrl: process.env.GOOGLE_ROUTES_BASE_URL || 'https://routes.googleapis.com',
      apiKey: process.env.GOOGLE_MAPS_API_KEY || '',
      fieldMask: process.env.GOOGLE_ROUTES_FIELD_MASK || 'routes.duration,routes.staticDuration,routes.distanceMeters,routes.legs.duration,routes.legs.staticDuration,routes.legs.distanceMeters'
    }
  },
  notifications: {
    sendGrid: {
      apiKey: process.env.SENDGRID_API_KEY || '',
      fromEmail: process.env.SENDGRID_FROM_EMAIL || '',
      fromName: process.env.SENDGRID_FROM_NAME || 'ForeSightX'
    },
    twilio: {
      accountSid: process.env.TWILIO_ACCOUNT_SID || '',
      authToken: process.env.TWILIO_AUTH_TOKEN || '',
      phoneNumber: process.env.TWILIO_PHONE_NUMBER || '',
      messagingServiceSid: process.env.TWILIO_MESSAGING_SERVICE_SID || ''
    }
  },
  jobs: {
    advisoryCronEnabled: process.env.ADVISORY_CRON_ENABLED === 'true',
    advisoryCronSchedule: process.env.ADVISORY_CRON_SCHEDULE || '*/15 * * * *',
    advisoryBatchSize: Number(process.env.ADVISORY_BATCH_SIZE) || 100,
    defaultTrafficDestination: {
      latitude: process.env.DEFAULT_TRAFFIC_DESTINATION_LAT
        ? Number(process.env.DEFAULT_TRAFFIC_DESTINATION_LAT)
        : null,
      longitude: process.env.DEFAULT_TRAFFIC_DESTINATION_LON
        ? Number(process.env.DEFAULT_TRAFFIC_DESTINATION_LON)
        : null,
      address: process.env.DEFAULT_TRAFFIC_DESTINATION_ADDRESS || null
    }
  },
  modelForecast: {
    enabled: process.env.MODEL_FORECAST_ENABLED !== 'false',
    defaultCity: process.env.MODEL_FORECAST_CITY || 'bangalore',
    projectRoot: process.env.PROJECT_ROOT || path.resolve(__dirname, '../../..')
  },
  redis: {
    url: process.env.REDIS_URL || 'redis://localhost:6379',
    keyPrefix: process.env.REDIS_KEY_PREFIX || 'foresightx:',
    connectTimeoutMs: Number(process.env.REDIS_CONNECT_TIMEOUT_MS) || 5000
  },
  db: {
    host: process.env.DB_HOST || 'localhost',
    port: Number(process.env.DB_PORT) || 5432,
    database: process.env.DB_NAME || 'foresightx_db',
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || '',
    ssl: process.env.DB_SSL === 'true' ? { rejectUnauthorized: false } : false
  }
};

module.exports = config;
