FROM node:22-alpine AS deps

WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=dev

FROM node:22-alpine AS runtime

ENV NODE_ENV=production
WORKDIR /app

RUN addgroup -S foresightx && adduser -S foresightx -G foresightx

COPY --from=deps /app/node_modules ./node_modules
COPY package*.json ./
COPY src ./src

USER foresightx

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD node -e "const http=require('http'); const req=http.get('http://127.0.0.1:'+(process.env.PORT||5000)+'/api/health', (res)=>process.exit(res.statusCode===200?0:1)); req.on('error',()=>process.exit(1)); req.setTimeout(4000,()=>{req.destroy(); process.exit(1);});"

CMD ["npm", "start"]
