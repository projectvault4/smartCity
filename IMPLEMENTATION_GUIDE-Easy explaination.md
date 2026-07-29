# ForeSightX v2 — Automated Advisory Engine
## Complete Implementation Guide

---

## 📋 Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Installation & Setup](#installation--setup)
4. [Configuration](#configuration)
5. [Deployment](#deployment)
6. [API Reference](#api-reference)
7. [Personalization Logic](#personalization-logic)
8. [Notification Channels](#notification-channels)
9. [Monitoring & Logs](#monitoring--logs)
10. [Troubleshooting](#troubleshooting)

---

## 🏗️ Architecture Overview

### **Problem Statement (Old System)**
- ❌ Manual parameter entry via UI dropdowns
- ❌ No real data integration
- ❌ No user personalization
- ❌ No automated delivery
- ❌ No scheduled execution

### **Solution (New System v2)**
- ✅ Automated data fetching from external APIs
- ✅ Real-time user-specific risk assessment
- ✅ Multi-channel notification delivery (Email, SMS, In-App)
- ✅ Scheduled batch processing (every 15 minutes)
- ✅ Configurable rules engine
- ✅ Audit trail & compliance logging

### **Data Flow Diagram**
```
┌─────────────────────────────────────────────────────────────────┐
│                     EXTERNAL DATA SOURCES                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  WAQI (AQI)  │  │   Weather    │  │   Traffic    │          │
│  │              │  │   OpenWeather│  │ Google Maps  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└──────────────────┬──────────────────────┬───────────────────────┘
                   │                      │
                   ▼                      ▼
        ┌──────────────────────────┐
        │  BACKEND API (Node.js)   │
        │  - Fetch real-time data  │
        │  - Compute risk scores   │
        │  - Determine advisories  │
        └──────────────┬───────────┘
                       │
        ┌──────────────┴─────────────────┬─────────────────┐
        │                                │                 │
        ▼                                ▼                 ▼
  ┌──────────────┐             ┌──────────────┐    ┌─────────────┐
  │ PostgreSQL   │             │ Notification │    │   Frontend  │
  │ Database     │             │   Services   │    │   (HTML/JS) │
  │ (User data,  │             │ SendGrid,    │    │  Real-time  │
  │  Advisories, │             │ Twilio, etc  │    │  Dashboard  │
  │  Rules)      │             │              │    │             │
  └──────────────┘             └──────────────┘    └─────────────┘
        │                           │
        │         Email/SMS/       │
        └─────────────────────────►│
                                   ▼
                          ┌──────────────────┐
                          │  CITIZEN INBOX   │
                          │  (Personalized)  │
                          └──────────────────┘
```

---

## 📦 Prerequisites

### **System Requirements**
- Node.js v16+ (for backend)
- PostgreSQL 12+ (for database)
- Redis 6+ (optional, for caching)
- npm v8+ (for package management)

### **API Keys Required**
1. **Air Quality Data**
   - WAQI (World Air Quality Index): https://waqi.info/api/
   - OR IQAir: https://www.iqair.com/air-quality-api

2. **Weather Data**
   - OpenWeatherMap: https://openweathermap.org/api

3. **Traffic Data** (optional)
   - Google Maps API: https://developers.google.com/maps

4. **Email Service**
   - SendGrid: https://sendgrid.com/
   - OR AWS SES: https://aws.amazon.com/ses/

5. **SMS Service**
   - Twilio: https://www.twilio.com/
   - OR AWS SNS: https://aws.amazon.com/sns/

---

## ⚙️ Installation & Setup

### **Step 1: Clone & Install Dependencies**
```bash
# Clone the repository
git clone https://github.com/your-org/foresightx.git
cd foresightx

# Install backend dependencies
npm install

# Install database tools (PostgreSQL client)
# macOS
brew install postgresql

# Ubuntu/Debian
sudo apt-get install postgresql-client

# Windows
# Download from: https://www.postgresql.org/download/windows/
```

### **Step 2: Set Up Database**
```bash
# Create PostgreSQL database
createdb foresightx_db

# Load schema
psql -U postgres -d foresightx_db -f database-schema.sql

# Verify installation
psql -U postgres -d foresightx_db -c "SELECT COUNT(*) FROM users;"
```

### **Step 3: Environment Configuration**
```bash
# Copy example config
cp .env.example .env

# Edit .env with your API keys
nano .env
```

**Sample .env:**
```
PORT=5000
NODE_ENV=development

# APIs
WAQI_API_KEY=demo
OPENWEATHER_API_KEY=your_key_here
GOOGLE_MAPS_API_KEY=your_key_here

# Email
SENDGRID_API_KEY=SG.xxxxx
SENDGRID_FROM_EMAIL=noreply@foresightx.app

# SMS
TWILIO_ACCOUNT_SID=ACxxxxx
TWILIO_AUTH_TOKEN=xxxxx
TWILIO_PHONE_NUMBER=+1234567890

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=foresightx_db
DB_USER=postgres
DB_PASSWORD=your_password
```

### **Step 4: Start the Backend**
```bash
# Development mode (with auto-reload)
npm run dev

# Production mode
npm start

# Output:
# 🚀 ForeSightX Advisory Backend running on port 5000
# 📊 Base URL: http://localhost:5000
```

### **Step 5: Verify API**
```bash
# Test endpoint
curl http://localhost:5000/api/users

# Expected response:
# [
#   {
#     "id": 1,
#     "user_id": "user_001",
#     "name": "Ravi Kumar",
#     ...
#   }
# ]
```

---

## 🎯 Configuration

### **Advisory Rules Engine**
Rules determine which advisories are triggered for which groups.

#### **Default Rules:**

| Group | Condition | Severity | Advisory |
|-------|-----------|----------|----------|
| Respiratory | AQI = "Very Poor" | 🔴 Critical | Stay indoors, use air purifiers |
| Elderly | Temp ≥38°C + Weather≠Normal | 🟠 Warning | Extreme weather precaution |
| Outdoor Workers | AQI = "Very Poor" | 🔴 Critical | Mandatory mask & frequent breaks |
| Children | AQI = "Very Poor" | 🔴 Critical | Indoor recess at schools |
| Commuters | Traffic=Severe + Rain | 🔴 Critical | Defer non-essential travel |

#### **How to Add Custom Rules:**
```sql
-- Insert new rule into database
INSERT INTO advisory_rules (
  rule_name,
  group_key,
  condition_type,
  condition_json,
  advisory_title,
  advisory_description,
  severity,
  score_weight
) VALUES (
  'Elderly Heatwave Alert',
  'elder',
  'temperature',
  '{"temp_threshold": 38, "duration_hours": 2}',
  'Extreme Heat Advisory',
  'Temperature >38°C. Avoid outdoor activities, stay hydrated.',
  'critical',
  2
);
```

### **Risk Score Calculation**
```javascript
// Pseudo-code
riskScore = 0;

if (aqi === 'Very Poor') riskScore += 3;
else if (aqi === 'Poor') riskScore += 1.5;

if (weather === 'Heatwave' && age > 60) riskScore += 2;
if (weather === 'Heavy Rain' && traffic === 'Severe') riskScore += 3;

const riskLevel = 
  riskScore >= 8 ? 'High' :
  riskScore >= 5 ? 'Medium' :
  'Low';
```

### **User Segmentation**
```sql
-- View user segments
SELECT 
  group_key,
  COUNT(*) as user_count,
  ARRAY_AGG(name) as users
FROM users u
JOIN user_risk_groups urg ON u.id = urg.user_id
GROUP BY urg.group_key;

-- Result:
-- group_key | user_count | users
-- ----------+------------+-----------------------
-- resp      |     1,234  | [Ravi, Anita, ...]
-- elder     |     5,678  | [Priya, Raj, ...]
-- child     |     9,456  | [Aman, Zara, ...]
-- worker    |     3,210  | [Amit, Sanjay, ...]
-- commuter  |    18,945  | [...]
```

---

## 🚀 Deployment

### **Option 1: Local/Development**
```bash
npm run dev
# Runs on http://localhost:5000
```

### **Option 2: Docker Deployment**
```dockerfile
# Dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 5000
CMD ["node", "advisory-backend.js"]
```

```bash
# Build & run
docker build -t foresightx-backend .
docker run -p 5000:5000 --env-file .env foresightx-backend
```

### **Option 3: Cloud Deployment (Heroku)**
```bash
# Install Heroku CLI
npm install -g heroku

# Login & create app
heroku login
heroku create foresightx-backend

# Set environment variables
heroku config:set WAQI_API_KEY=xxx SENDGRID_API_KEY=xxx

# Push to Heroku
git push heroku main

# View logs
heroku logs --tail
```

### **Option 4: AWS/GCP/Azure**
See `CLOUD_DEPLOYMENT.md` for detailed guides.

---

## 📡 API Reference

### **1. Compute Advisories**
```http
GET /api/compute-advisories?userId=user_001
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "userId": "user_001",
      "name": "Ravi Kumar",
      "ward": "Anekal Ward — Bangalore South",
      "advisories": [
        {
          "key": "commuter",
          "tag": "crit",
          "title": "Commute Alert — Severe Delays",
          "desc": "Heavy traffic + rainfall expected...",
          "personalizedFor": "Ravi Kumar",
          "severity": "critical"
        }
      ],
      "conditions": {
        "aqi": "Poor",
        "weather": "Rain",
        "temp": 28,
        "traffic": "Severe"
      },
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### **2. Get All Users**
```http
GET /api/users
```

### **3. Add New User**
```http
POST /api/users
Content-Type: application/json

{
  "name": "New Citizen",
  "email": "new@example.com",
  "phone": "+91-9000000000",
  "ward": "Your Ward — City",
  "age": 35,
  "groups": ["elder", "commuter"],
  "location": {
    "lat": 12.7139,
    "lng": 77.6245
  },
  "preferences": {
    "email": true,
    "sms": true,
    "inApp": true
  }
}
```

### **4. Advisory History**
```http
GET /api/advisories-history?limit=50&userId=user_001
```

---

## 🎨 Personalization Logic

### **How Personalization Works:**

#### **Step 1: User Profile**
```javascript
{
  "id": "user_001",
  "name": "Ravi Kumar",
  "age": 45,
  "groups": ["elder", "commuter"],
  "ward": "Anekal Ward",
  "location": { lat: 12.7139, lng: 77.6245 }
}
```

#### **Step 2: Fetch Real-Time Data for User's Location**
```javascript
const aqi = await fetchAQIData(12.7139, 77.6245); // "Poor"
const { weather, temp } = await fetchWeatherData(12.7139, 77.6245);
// { weather: "Rain", temp: 28 }
const traffic = await fetchTrafficData("Anekal Ward"); // "Severe"
```

#### **Step 3: Apply Risk Group Rules**
```javascript
// Elderly group rules apply to Ravi
if (user.groups.includes('elder')) {
  if (temp >= 38 || temp <= 5) {
    // Trigger extreme weather advisory
  }
}

// Commuter group rules apply to Ravi
if (user.groups.includes('commuter')) {
  if (traffic === 'Severe' && weather === 'Rain') {
    // Trigger travel delay advisory
  }
}
```

#### **Step 4: Generate Personalized Message**
```javascript
const advisory = {
  title: "Commute Alert — Severe Delays Expected",
  desc: `Dear ${user.name}, severe traffic congestion + heavy rain
         expected on your commute route in ${user.ward.split('—')[1]}.
         Plan extra time or consider public transit.`,
  personalizedFor: user.name,
  severity: "critical"
};
```

#### **Step 5: Send via Preferred Channels**
```javascript
if (user.preferences.email) {
  await sendEmail(user.email, advisory);
}
if (user.preferences.sms) {
  await sendSMS(user.phone, advisory);
}
if (user.preferences.inApp) {
  await recordInAppNotification(user.id, advisory);
}
```

### **Personalization Examples:**

**Scenario 1: Elderly Citizen in Heat**
- Profile: Age 68, Risk Groups: ['elder']
- Data: Temp 40°C, AQI Good, Traffic Low, Weather Normal
- Advisory: "🌡️ Extreme Heat Alert — Hydration & Rest Reminder"

**Scenario 2: School Child in Poor AQI**
- Profile: Age 12, Risk Groups: ['child', 'commuter']
- Data: AQI Very Poor, Weather Fog, Temp 15°C, Traffic High
- Advisory: "🫁 School Indoor Recess + Commute Delay Alert"

**Scenario 3: Construction Worker in Pollution**
- Profile: Age 32, Risk Groups: ['worker']
- Data: AQI Very Poor, Weather Normal, Temp 32°C
- Advisory: "🏗️ Field Safety Alert — Mandatory PPE & Frequent Breaks"

---

## 📮 Notification Channels

### **1. Email Notifications**
```javascript
// Using SendGrid
const sgMail = require('@sendgrid/mail');
sgMail.setApiKey(process.env.SENDGRID_API_KEY);

const msg = {
  to: user.email,
  from: 'alerts@foresightx.app',
  subject: `ForeSightX Alert: ${advisory.title}`,
  html: `
    <h2>${advisory.title}</h2>
    <p>${advisory.desc}</p>
    <p style="color: gray; font-size: 12px;">
      Personalized for: ${user.name}<br/>
      Ward: ${user.ward}
    </p>
  `
};

await sgMail.send(msg);
```

### **2. SMS Notifications**
```javascript
// Using Twilio
const twilio = require('twilio');
const client = twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN);

await client.messages.create({
  body: `ForeSightX Alert: ${advisory.title}. ${advisory.desc.substring(0, 100)}...`,
  from: process.env.TWILIO_PHONE_NUMBER,
  to: user.phone
});
```

### **3. In-App Notifications**
```javascript
// Store in database
await db.query(`
  INSERT INTO notifications (user_id, advisory_id, title, message, is_read)
  VALUES ($1, $2, $3, $4, false)
`, [user.id, advisory.id, advisory.title, advisory.desc]);

// Frontend polls or uses WebSocket to get real-time updates
```

---

## 📊 Monitoring & Logs

### **View Execution Logs**
```bash
# Real-time logs
tail -f logs/foresightx.log

# Check last cron execution
psql -d foresightx_db -c "
  SELECT * FROM execution_log 
  ORDER BY completed_at DESC 
  LIMIT 5;"
```

### **Sample Log Output**
```
[2024-01-15 10:15:00] 🔄 CRON: Computing advisories for 18,945 users...
[2024-01-15 10:15:02] ✅ Fetched real-time conditions
  - AQI: 12 wards updated
  - Weather: 12 wards updated
  - Traffic: 12 wards updated
[2024-01-15 10:15:15] 📊 Risk assessment complete
  - High risk: 234 users
  - Medium risk: 567 users
  - Low risk: 18,144 users
[2024-01-15 10:15:20] 📧 Email: 450 sent, 2 failed
[2024-01-15 10:15:25] 📱 SMS: 150 sent, 0 failed
[2024-01-15 10:15:26] ✅ Execution complete (26 seconds)
```

### **Monitoring Queries**
```sql
-- Today's advisory count by severity
SELECT severity, COUNT(*) FROM advisories_sent 
WHERE DATE(sent_at) = TODAY() 
GROUP BY severity;

-- Users by group
SELECT group_key, COUNT(DISTINCT user_id) FROM user_risk_groups 
GROUP BY group_key;

-- Email delivery rate
SELECT 
  (SUM(CASE WHEN email_sent THEN 1 ELSE 0 END)::float / COUNT(*)) * 100 as delivery_rate
FROM advisories_sent 
WHERE sent_at >= NOW() - INTERVAL '24 hours';
```

---

## 🐛 Troubleshooting

### **Issue: API Keys Not Working**
```bash
# Verify .env is loaded
node -e "console.log(process.env.WAQI_API_KEY)"

# Test API directly
curl "https://api.waqi.info/feed/beijing/?token=YOUR_KEY"
```

### **Issue: Database Connection Failed**
```bash
# Test connection
psql -U postgres -h localhost -d foresightx_db -c "SELECT 1"

# Check PostgreSQL is running
sudo service postgresql status

# Restart if needed
sudo service postgresql restart
```

### **Issue: Cron Job Not Running**
```bash
# Check if process is alive
ps aux | grep "node advisory-backend"

# Check cron logs
tail -f /var/log/syslog | grep CRON
```

### **Issue: High Memory Usage**
```bash
# Monitor memory
top -p $(pgrep -f "advisory-backend")

# Solution: Batch users in chunks
const BATCH_SIZE = 100;
for (let i = 0; i < users.length; i += BATCH_SIZE) {
  const batch = users.slice(i, i + BATCH_SIZE);
  await processBatch(batch);
}
```

---

## 📈 Next Steps

1. **Scale to Production**
   - Set up load balancing
   - Configure multi-region deployment
   - Set up database replication

2. **Advanced Features**
   - Machine learning for risk prediction
   - User feedback loop optimization
   - A/B testing for advisory copy

3. **Integration**
   - Webhook support for third-party platforms
   - GraphQL API for mobile apps
   - Real-time WebSocket updates

4. **Analytics**
   - Dashboard for advisory metrics
   - User engagement tracking
   - Impact measurement

---

## 📞 Support
For issues, feature requests, or questions:
- GitHub Issues: https://github.com/your-org/foresightx/issues
- Email: support@foresightx.app

---

**Last Updated:** January 2024
**Version:** 2.0.0 (Automated)

---
---

# 🧩 APPENDIX — Extended Setup, Commands & Operations

> This appendix **adds** to everything above. Nothing earlier was changed. It fills in the commands and detail that a first-time operator usually needs but that weren't spelled out in the core guide.

---

## 🔧 A. Pre-Flight Environment Checks

Run these **before** Step 1 to confirm your machine is ready.

```bash
# Confirm Node.js version is v16+
node -v

# Confirm npm version is v8+
npm -v

# Confirm PostgreSQL is installed and its version
psql --version
postgres --version

# Confirm git is installed
git --version

# Confirm Redis is installed (optional caching layer)
redis-server --version
redis-cli ping        # should reply: PONG

# Check which ports are already in use (make sure 5000 & 5432 are free)
# macOS / Linux
sudo lsof -i :5000
sudo lsof -i :5432

# Windows (PowerShell)
netstat -ano | findstr :5000
netstat -ano | findstr :5432
```

If a port is taken, either stop that process or change `PORT` in `.env`.

---

## 🗄️ B. Full Database Setup (Extended)

The core guide creates the DB and loads the schema. These commands cover the parts most people trip on: creating a dedicated user, granting permissions, seeding data, and taking backups.

### B.1 — Install & start PostgreSQL server (not just the client)
```bash
# macOS (Homebrew) — installs server + starts it
brew install postgresql@15
brew services start postgresql@15

# Ubuntu/Debian — full server
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib
sudo systemctl enable postgresql
sudo systemctl start postgresql

# Verify the server is actually running
pg_isready
```

### B.2 — Create a dedicated DB user (safer than using `postgres`)
```bash
# Open the psql shell as the postgres superuser
sudo -u postgres psql
```
```sql
-- Inside psql:
CREATE USER foresightx_user WITH PASSWORD 'change_this_password';
CREATE DATABASE foresightx_db OWNER foresightx_user;
GRANT ALL PRIVILEGES ON DATABASE foresightx_db TO foresightx_user;
\q
```

> Then set `DB_USER=foresightx_user` and the matching password in `.env`.

### B.3 — Load schema + seed data
```bash
# Load the schema (as in Step 2)
psql -U foresightx_user -d foresightx_db -f database-schema.sql

# Load seed/sample data if a seed file exists
psql -U foresightx_user -d foresightx_db -f seed-data.sql

# List all tables to confirm the schema loaded
psql -U foresightx_user -d foresightx_db -c "\dt"

# Inspect the structure of a specific table
psql -U foresightx_user -d foresightx_db -c "\d users"
psql -U foresightx_user -d foresightx_db -c "\d advisory_rules"
```

### B.4 — Backup & restore
```bash
# Backup the whole database to a file
pg_dump -U foresightx_user foresightx_db > backup_$(date +%F).sql

# Restore from a backup
psql -U foresightx_user -d foresightx_db < backup_2024-01-15.sql

# Backup only the schema (no data)
pg_dump -U foresightx_user --schema-only foresightx_db > schema_only.sql
```

### B.5 — Reset the database (danger: wipes everything)
```bash
dropdb -U foresightx_user foresightx_db
createdb -U foresightx_user foresightx_db
psql -U foresightx_user -d foresightx_db -f database-schema.sql
```

---

## 📥 C. Node Dependencies (Explicit Install)

If a fresh `npm install` fails or you're building `package.json` from scratch, these are the packages the code in this guide relies on.

```bash
# Core server + DB + scheduling
npm install express pg dotenv node-cron

# HTTP client for fetching AQI / weather / traffic
npm install axios

# Notification providers
npm install @sendgrid/mail twilio

# Optional caching
npm install redis

# Dev-only: auto-reload during development (powers `npm run dev`)
npm install --save-dev nodemon
```

Add these scripts to `package.json` if they aren't present:
```json
{
  "scripts": {
    "start": "node advisory-backend.js",
    "dev": "nodemon advisory-backend.js"
  }
}
```

---

## ⏰ D. Scheduling the 15-Minute Job

The architecture promises "every 15 minutes." Here are the two common ways to actually make that happen.

### D.1 — In-code with `node-cron` (recommended, runs inside the app)
```javascript
const cron = require('node-cron');

// Runs at minute 0, 15, 30, 45 of every hour
cron.schedule('*/15 * * * *', async () => {
  console.log('🔄 CRON: Computing advisories for all users...');
  await computeAdvisoriesForAllUsers();
});
```

### D.2 — System crontab (runs the script from the OS)
```bash
# Open the crontab editor
crontab -e

# Add this line — runs every 15 minutes and logs output
*/15 * * * * cd /path/to/foresightx && /usr/bin/node advisory-backend.js >> logs/cron.log 2>&1

# List active cron jobs to confirm it was saved
crontab -l
```

Cron field cheat sheet: `minute hour day-of-month month day-of-week`.

---

## 🧪 E. Testing Every API Endpoint (curl)

The core guide only showed `GET /api/users`. Here is every endpoint exercised with curl.

```bash
# 1) Compute advisories for one user
curl "http://localhost:5000/api/compute-advisories?userId=user_001"

# 2) Compute advisories for ALL users (batch)
curl "http://localhost:5000/api/compute-advisories"

# 3) Get all users
curl http://localhost:5000/api/users

# 4) Add a new user (POST with JSON body)
curl -X POST http://localhost:5000/api/users \
  -H "Content-Type: application/json" \
  -d '{
    "name": "New Citizen",
    "email": "new@example.com",
    "phone": "+91-9000000000",
    "ward": "Your Ward — City",
    "age": 35,
    "groups": ["elder", "commuter"],
    "location": { "lat": 12.7139, "lng": 77.6245 },
    "preferences": { "email": true, "sms": true, "inApp": true }
  }'

# 5) Advisory history (last 50 for a user)
curl "http://localhost:5000/api/advisories-history?limit=50&userId=user_001"

# Pretty-print any JSON response (pipe through jq)
curl http://localhost:5000/api/users | jq .

# Show HTTP status + headers while testing
curl -i http://localhost:5000/api/users
```

---

## 📡 F. Testing the External Data APIs Directly

Confirm each third-party key works **before** blaming your own code.

```bash
# WAQI — air quality by city
curl "https://api.waqi.info/feed/bangalore/?token=$WAQI_API_KEY"

# WAQI — air quality by GPS coordinates
curl "https://api.waqi.info/feed/geo:12.7139;77.6245/?token=$WAQI_API_KEY"

# OpenWeatherMap — current weather by coordinates (metric = Celsius)
curl "https://api.openweathermap.org/data/2.5/weather?lat=12.7139&lon=77.6245&units=metric&appid=$OPENWEATHER_API_KEY"

# Google Maps — traffic-aware travel time (Distance Matrix)
curl "https://maps.googleapis.com/maps/api/distancematrix/json?origins=12.7139,77.6245&destinations=12.9716,77.5946&departure_time=now&key=$GOOGLE_MAPS_API_KEY"
```

> Tip: load your `.env` into the current shell first so `$WAQI_API_KEY` etc. are available:
> ```bash
> export $(grep -v '^#' .env | xargs)
> ```

---

## 🐳 G. Docker — Extended (Compose with DB + Redis)

The core guide has a single-container Dockerfile. In practice you'll want the app, Postgres, and Redis together. Save this as `docker-compose.yml`:

```yaml
version: "3.8"
services:
  app:
    build: .
    ports:
      - "5000:5000"
    env_file: .env
    depends_on:
      - db
      - redis

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: foresightx_db
      POSTGRES_USER: foresightx_user
      POSTGRES_PASSWORD: change_this_password
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pgdata:
```

```bash
# Build and start the whole stack in the background
docker compose up -d --build

# Watch logs from all services
docker compose logs -f

# Load the schema into the running DB container
docker compose exec db psql -U foresightx_user -d foresightx_db -f /schema/database-schema.sql

# Stop everything
docker compose down

# Stop AND delete the database volume (fresh start)
docker compose down -v

# List running containers / check health
docker ps
docker compose ps
```

---

## ☁️ H. Cloud Deployment — Extra Commands

### H.1 — Heroku (add-ons the app actually needs)
```bash
# Provision managed Postgres
heroku addons:create heroku-postgresql:mini -a foresightx-backend

# Provision managed Redis
heroku addons:create heroku-redis:mini -a foresightx-backend

# Set ALL required env vars in one go
heroku config:set \
  OPENWEATHER_API_KEY=xxx \
  GOOGLE_MAPS_API_KEY=xxx \
  SENDGRID_API_KEY=SG.xxx \
  TWILIO_ACCOUNT_SID=ACxxx \
  TWILIO_AUTH_TOKEN=xxx \
  TWILIO_PHONE_NUMBER=+1234567890 \
  -a foresightx-backend

# Run schema load on the Heroku DB
heroku pg:psql -a foresightx-backend -f database-schema.sql

# Scale to one always-on web dyno
heroku ps:scale web=1 -a foresightx-backend

# Open the deployed app in a browser
heroku open -a foresightx-backend
```

### H.2 — Docker image push to a registry (AWS ECR example)
```bash
# Log in to ECR
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <acct>.dkr.ecr.ap-south-1.amazonaws.com

# Tag and push
docker tag foresightx-backend:latest <acct>.dkr.ecr.ap-south-1.amazonaws.com/foresightx-backend:latest
docker push <acct>.dkr.ecr.ap-south-1.amazonaws.com/foresightx-backend:latest
```

---

## 🔐 I. Security & Secrets Hygiene

```bash
# Make sure .env is NEVER committed to git
echo ".env" >> .gitignore
echo "node_modules/" >> .gitignore
echo "logs/" >> .gitignore

# Check that .env is actually ignored
git check-ignore -v .env

# Audit installed npm packages for known vulnerabilities
npm audit
npm audit fix

# Rotate a leaked key: update .env, then restart
#   local:  Ctrl+C then `npm start`
#   docker: docker compose restart app
#   heroku: heroku config:set KEY=newvalue -a foresightx-backend
```

Rules of thumb: never hard-code keys in source, use a dedicated DB user (not `postgres`) in production, and keep a separate `.env` per environment (dev / staging / prod).

---

## 📊 J. Extra Monitoring & Maintenance Queries

Adds to the Monitoring section above.

```sql
-- Notifications waiting to be read (in-app)
SELECT user_id, COUNT(*) AS unread
FROM notifications
WHERE is_read = false
GROUP BY user_id
ORDER BY unread DESC;

-- Failed deliveries in the last 24h
SELECT channel, COUNT(*) AS failures
FROM advisories_sent
WHERE sent_at >= NOW() - INTERVAL '24 hours'
  AND delivery_status = 'failed'
GROUP BY channel;

-- Average risk score per group today
SELECT urg.group_key, ROUND(AVG(a.risk_score), 2) AS avg_score
FROM advisories_sent a
JOIN user_risk_groups urg ON a.user_id = urg.user_id
WHERE DATE(a.sent_at) = CURRENT_DATE
GROUP BY urg.group_key;

-- Largest tables (find what's eating disk)
SELECT relname AS table, pg_size_pretty(pg_total_relation_size(relid)) AS size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 10;
```

```bash
# Rotate/trim a growing log file (keep last 1000 lines)
tail -n 1000 logs/foresightx.log > logs/foresightx.tmp && mv logs/foresightx.tmp logs/foresightx.log

# Count how many advisories were logged today
grep "$(date +%F)" logs/foresightx.log | grep -c "Execution complete"
```

---

## 🩺 K. Troubleshooting — Additional Cases

Adds to the Troubleshooting section above.

### K.1 — Port 5000 already in use
```bash
# Find and kill whatever holds the port
lsof -ti :5000 | xargs kill -9
# Or just change PORT in .env
```

### K.2 — `npm install` fails / corrupted modules
```bash
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

### K.3 — Emails/SMS silently not arriving
```bash
# SendGrid: verify the key and sender identity
curl -X POST https://api.sendgrid.com/v3/mail/send \
  -H "Authorization: Bearer $SENDGRID_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"personalizations":[{"to":[{"email":"you@example.com"}]}],"from":{"email":"noreply@foresightx.app"},"subject":"Test","content":[{"type":"text/plain","value":"hi"}]}'

# Twilio: confirm account + list recent messages (check for errors)
curl -X GET "https://api.twilio.com/2010-04-01/Accounts/$TWILIO_ACCOUNT_SID/Messages.json?PageSize=5" \
  -u "$TWILIO_ACCOUNT_SID:$TWILIO_AUTH_TOKEN"
```
> Common causes: unverified SendGrid sender, trial Twilio number can only text verified numbers, or a rate limit.

### K.4 — Timezone / cron firing at the wrong time
```bash
# Check the server's clock and timezone
date
timedatectl        # Linux

# Set a timezone for node-cron explicitly in code:
#   cron.schedule('*/15 * * * *', fn, { timezone: 'Asia/Kolkata' });
```

### K.5 — Redis connection refused
```bash
redis-cli ping                 # expect PONG
sudo systemctl start redis     # start if down (Linux)
brew services start redis      # macOS
```

---

## ✅ L. End-to-End Smoke Test (run after any deploy)

A single ordered checklist to confirm the whole pipeline works.

```bash
# 1. Server is up
curl -i http://localhost:5000/api/users

# 2. External data keys respond
curl "https://api.waqi.info/feed/bangalore/?token=$WAQI_API_KEY"
curl "https://api.openweathermap.org/data/2.5/weather?lat=12.7139&lon=77.6245&units=metric&appid=$OPENWEATHER_API_KEY"

# 3. Add a test user
curl -X POST http://localhost:5000/api/users -H "Content-Type: application/json" \
  -d '{"name":"Smoke Test","email":"you@example.com","phone":"+919000000000","ward":"Test Ward — City","age":70,"groups":["elder"],"location":{"lat":12.7139,"lng":77.6245},"preferences":{"email":true,"sms":false,"inApp":true}}'

# 4. Compute advisories for that user
curl "http://localhost:5000/api/compute-advisories?userId=user_001"

# 5. Confirm it was recorded in history
curl "http://localhost:5000/api/advisories-history?limit=5&userId=user_001"

# 6. Confirm the run appears in logs
tail -n 20 logs/foresightx.log
```

If all six steps return sensible output, the deployment is healthy. ✅

---

**Appendix Added:** Extended commands for environment checks, DB user/backup, dependency install, scheduling, endpoint & external-API testing, Docker Compose, cloud add-ons, security hygiene, monitoring, extra troubleshooting, and an end-to-end smoke test.
