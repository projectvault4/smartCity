const config = require('../config/env');
const logger = require('../utils/logger');

const getDatabaseErrorResponse = (error) => {
  if (error.code === '23505') {
    return {
      statusCode: 409,
      message: 'Resource already exists',
      details: {
        constraint: error.constraint
      }
    };
  }

  if (error.code === '22P02') {
    return {
      statusCode: 400,
      message: 'Invalid value for one or more fields'
    };
  }

  if (error.code === '23514') {
    return {
      statusCode: 400,
      message: 'Request violates a database constraint',
      details: {
        constraint: error.constraint
      }
    };
  }

  return null;
};

const getProviderErrorResponse = (error) => {
  if (error.name !== 'ProviderError') {
    return null;
  }

  const status = error.details?.status;

  return {
    statusCode: status && status >= 400 && status < 500 ? 502 : 503,
    message: error.message,
    details: {
      provider: error.provider,
      ...error.details
    }
  };
};

const getNotificationErrorResponse = (error) => {
  if (error.name !== 'NotificationError') {
    return null;
  }

  return {
    statusCode: 502,
    message: error.message,
    details: {
      channel: error.channel,
      ...error.details
    }
  };
};

const errorHandler = (error, req, res, _next) => {
  const databaseError = getDatabaseErrorResponse(error);
  const providerError = getProviderErrorResponse(error);
  const notificationError = getNotificationErrorResponse(error);
  const mappedError = databaseError || providerError || notificationError;
  const statusCode = mappedError?.statusCode || error.statusCode || error.status || 500;
  const message = mappedError?.message || error.message || 'Internal server error';
  const details = mappedError?.details || error.details;

  logger.error('Request failed', {
    requestId: req.id,
    method: req.method,
    path: req.originalUrl,
    statusCode,
    message,
    details,
    stack: error.stack
  });

  res.status(statusCode).json({
    error: {
      statusCode,
      message,
      ...(details && { details }),
      requestId: req.id
    },
    message,
    ...(details && { details }),
    requestId: req.id,
    ...(config.nodeEnv === 'development' && { stack: error.stack })
  });
};

module.exports = errorHandler;
