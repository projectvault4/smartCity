FROM node:22-alpine AS deps

WORKDIR /app

COPY backend/package*.json ./
RUN npm ci --omit=dev

FROM node:22-alpine AS runtime

ENV NODE_ENV=production
ENV PROJECT_ROOT=/app
WORKDIR /app

RUN addgroup -S foresightx && adduser -S foresightx -G foresightx

COPY --from=deps /app/node_modules ./node_modules
COPY backend/package*.json ./
COPY backend/src ./src

# Forecast / anomaly artifacts served by the model API. These are copied from
# the repository root build context (docker-compose sets context: .).
RUN mkdir -p /app/outputs
COPY outputs/past_present_future_forecast.csv ./outputs/past_present_future_forecast.csv
COPY outputs/forecast_2026.csv ./outputs/forecast_2026.csv
COPY outputs/urban_events.json ./outputs/urban_events.json
COPY outputs/urban_anomaly_timeline.csv ./outputs/urban_anomaly_timeline.csv
COPY outputs/urban_events_2026.json ./outputs/urban_events_2026.json
COPY outputs/urban_anomaly_timeline_2026.csv ./outputs/urban_anomaly_timeline_2026.csv

RUN chown -R foresightx:foresightx /app

USER foresightx

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD node -e "const http=require('http'); const req=http.get('http://127.0.0.1:'+(process.env.PORT||5000)+'/api/health', (res)=>process.exit(res.statusCode===200?0:1)); req.on('error',()=>process.exit(1)); req.setTimeout(4000,()=>{req.destroy(); process.exit(1);});"

CMD ["npm", "start"]
