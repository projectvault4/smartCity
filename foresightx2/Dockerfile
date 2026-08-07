FROM node:22-alpine AS deps

WORKDIR /app

COPY package*.json ./
RUN npm ci

FROM node:22-alpine AS build

ARG GROQ_API_KEY=""
ARG GEMINI_API_KEY=""
ARG VITE_API_BASE_URL=/api

ENV GROQ_API_KEY=$GROQ_API_KEY
ENV GEMINI_API_KEY=$GEMINI_API_KEY
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

RUN npm run build

FROM nginx:1.27-alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
