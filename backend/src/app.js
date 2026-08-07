const express = require('express');
const cors = require('cors');
const helmet = require('helmet');

const config = require('./config/env');
const routes = require('./routes');
const requestId = require('./middleware/requestId');
const requestLogger = require('./middleware/requestLogger');
const notFound = require('./middleware/notFound');
const errorHandler = require('./middleware/errorHandler');

const app = express();

app.use(requestId);
app.use(helmet());
app.use(cors({ origin: config.clientOrigin }));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(requestLogger);

app.get('/', (req, res) => {
  res.status(200).json({
    status: 'ok',
    service: 'ForeSightX backend',
    health: '/api/health'
  });
});

app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'ok',
    service: 'ForeSightX backend',
    health: '/api/health'
  });
});

app.use('/api', routes);

app.use(notFound);
app.use(errorHandler);

module.exports = app;
