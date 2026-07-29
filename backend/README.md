# ForeSightX Backend

Node.js Express backend scaffold for ForeSightX with PostgreSQL configuration.

## Setup

```bash
cd backend
npm install
cp .env.example .env
npm run dev
```

## Structure

```text
src/
  config/        Environment and PostgreSQL setup
  controllers/   Request handlers
  middleware/    Express middleware
  migrations/    Database migration files
  models/        Data models
  repositories/  Database access modules
  routes/        API route definitions
  services/      Business services
  utils/         Shared helpers
```

## Users API

```text
GET    /api/users
GET    /api/users/:id
POST   /api/users
PUT    /api/users/:id
PATCH  /api/users/:id
DELETE /api/users/:id
```

`GET /api/users` supports `limit`, `offset`, `status`, `city`, `ward`, and `search` query parameters.

Create and update payload fields:

```json
{
  "user_id": "user_001",
  "name": "Ravi Kumar",
  "email": "ravi@example.com",
  "phone": "+919876543210",
  "age": 34,
  "city": "Bangalore",
  "ward": "Indiranagar",
  "latitude": 12.9716,
  "longitude": 77.5946,
  "preferences": {},
  "status": "active"
}
```

Business logic for advisories and notifications can be added inside the feature folders later.

## Data Provider Services

Provider adapters live in `src/services` and return normalized objects:

```js
const {
  fetchAqi,
  fetchCurrentWeather,
  fetchTrafficRoute
} = require('./src/services/dataProviders.service');
```

Examples:

```js
await fetchAqi({ city: 'bangalore' });
await fetchAqi({ latitude: 12.9716, longitude: 77.5946 });

await fetchCurrentWeather({ latitude: 12.9716, longitude: 77.5946 });

await fetchTrafficRoute({
  origin: { latitude: 12.9716, longitude: 77.5946 },
  destination: { latitude: 12.9352, longitude: 77.6245 }
});
```

All provider calls use Axios with configurable timeout and retry settings from `.env`.

## Risk Engine

Use the risk engine directly from backend jobs or call the assessment endpoint:

```text
POST /api/risk/assess
```

Example payload:

```json
{
  "user": {
    "id": "user_001",
    "name": "Ravi Kumar",
    "age": 68
  },
  "riskGroups": ["elder", "commuter"],
  "aqi": { "aqi": 240 },
  "weather": {
    "weather": { "main": "Rain", "description": "heavy rain" },
    "temperature": { "value": 40 },
    "rainLastHourMm": 8
  },
  "traffic": { "congestionLevel": "severe" }
}
```

The response includes `score`, `riskLevel`, `severity`, normalized inputs, scoring factors, and advisory candidates.

## Advisory Generator

Generate personalized advisories from a risk assessment, or let the service calculate risk from the supplied conditions:

```text
POST /api/advisories/generate
POST /api/advisories/generate-batch
```

Single-user example:

```json
{
  "user": {
    "id": "user_001",
    "name": "Ravi Kumar",
    "age": 68,
    "ward": "Anekal Ward",
    "city": "Bangalore",
    "email": "ravi@example.com",
    "preferences": {
      "inApp": true,
      "email": true
    }
  },
  "riskGroups": ["elder", "commuter"],
  "aqi": { "aqi": 240 },
  "weather": {
    "weather": { "main": "Rain", "description": "heavy rain" },
    "temperature": { "value": 40 },
    "rainLastHourMm": 8
  },
  "traffic": { "congestionLevel": "severe" }
}
```

Generated advisories include personalized title, message, severity, risk score, source factors, recommended actions, delivery channels, and context ready for later persistence or notification delivery.

## Notification Services

Notification delivery supports SendGrid email, Twilio SMS, and PostgreSQL-backed in-app notifications.

```text
POST  /api/notifications/deliver
POST  /api/notifications/deliver-batch
GET   /api/notifications/users/:userId
PATCH /api/notifications/users/:userId/:id/read
```

Single delivery payload:

```json
{
  "user": {
    "id": "00000000-0000-4000-8000-000000000001",
    "email": "ravi@example.com",
    "phone": "+919876543210"
  },
  "advisory": {
    "userId": "00000000-0000-4000-8000-000000000001",
    "title": "Commute Alert - Severe Delays Expected",
    "message": "Dear Ravi, severe traffic and rain are expected.",
    "severity": "critical",
    "riskScore": 8,
    "riskLevel": "High",
    "deliveryChannels": ["in_app", "email", "sms"],
    "sourceFactors": ["commuter_rain_severe_traffic"],
    "recommendedActions": ["Delay non-essential travel."]
  }
}
```

Each delivery attempt is recorded in `advisories_sent`. In-app deliveries also create a row in `notifications`.

## Advisory Cron Job

The backend includes a `node-cron` batch job that processes active users every 15 minutes.

```env
ADVISORY_CRON_ENABLED=true
ADVISORY_CRON_SCHEDULE=*/15 * * * *
ADVISORY_BATCH_SIZE=100
```

When enabled, the job:

1. Loads active users with their `user_risk_groups`.
2. Fetches live WAQI AQI and OpenWeather data for each user location or city.
3. Fetches Google traffic data when the user has a commute destination, or a default traffic destination is configured.
4. Calculates risk and generates personalized advisories.
5. Stores delivery attempts in `advisories_sent`.
6. Creates in-app notifications and sends email/SMS according to user preferences.
7. Records run status, counts, provider errors, and failures in `execution_log`.

Manual trigger:

```text
POST /api/jobs/advisories/run
```

Traffic destinations can be provided per user in `preferences.commuteDestination`:

```json
{
  "commuteDestination": {
    "latitude": 12.9352,
    "longitude": 77.6245
  }
}
```

Or globally with `DEFAULT_TRAFFIC_DESTINATION_LAT`, `DEFAULT_TRAFFIC_DESTINATION_LON`, or `DEFAULT_TRAFFIC_DESTINATION_ADDRESS`.

## Logging, Monitoring, and Tests

Structured logs use Winston and include request method, path, status, duration, and request id. Every response includes `X-Request-Id`; callers can also provide their own `x-request-id`.

Monitoring endpoints:

```text
GET /api/health
GET /api/health/metrics
```

Run tests:

```bash
npm test
```

The Jest suite covers API validation/response behavior and Risk Engine scoring scenarios.

## Docker Production Stack

The production stack runs the Node backend with PostgreSQL and Redis:

```bash
cp backend/.env.production.example backend/.env.production
docker compose up --build -d
```

The app container runs database migrations before starting:

```text
npm run migrate && npm start
```

Services:

```text
app       Node.js Express API on port 5000
postgres  PostgreSQL 16 with persistent volume
redis     Redis 7 with append-only persistence
```

Production defaults include non-root app execution, dependency health checks, restart policies, internal-only PostgreSQL/Redis networking, log rotation, and `no-new-privileges`.

Useful commands:

```bash
docker compose ps
docker compose logs -f app
docker compose exec app npm run migrate
docker compose down
```

Readiness check:

```text
GET /api/health/ready
```
