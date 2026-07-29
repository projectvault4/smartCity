const HttpError = require('../utils/HttpError');

const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const supportedChannels = new Set(['in_app', 'email', 'sms', 'push']);

const validateUser = (user, errors) => {
  if (!user || typeof user !== 'object' || Array.isArray(user)) {
    errors.push({ field: 'user', message: 'user must be an object' });
    return;
  }

  if (!user.id || !uuidPattern.test(user.id)) {
    errors.push({ field: 'user.id', message: 'user.id must be a valid UUID' });
  }
};

const validateAdvisory = (advisory, field, errors) => {
  if (!advisory || typeof advisory !== 'object' || Array.isArray(advisory)) {
    errors.push({ field, message: `${field} must be an object` });
    return;
  }

  ['userId', 'title', 'message', 'severity'].forEach((key) => {
    if (!advisory[key]) {
      errors.push({ field: `${field}.${key}`, message: `${field}.${key} is required` });
    }
  });

  if (advisory.userId && !uuidPattern.test(advisory.userId)) {
    errors.push({ field: `${field}.userId`, message: `${field}.userId must be a valid UUID` });
  }

  if (advisory.deliveryChannels !== undefined) {
    if (!Array.isArray(advisory.deliveryChannels)) {
      errors.push({ field: `${field}.deliveryChannels`, message: `${field}.deliveryChannels must be an array` });
    } else {
      advisory.deliveryChannels.forEach((channel) => {
        if (!supportedChannels.has(channel)) {
          errors.push({ field: `${field}.deliveryChannels`, message: `Unsupported delivery channel: ${channel}` });
        }
      });
    }
  }
};

const validatePagination = (query, errors) => {
  const limit = Number(query.limit || 50);
  const offset = Number(query.offset || 0);

  if (!Number.isInteger(limit) || limit < 1 || limit > 100) {
    errors.push({ field: 'limit', message: 'limit must be an integer between 1 and 100' });
  }

  if (!Number.isInteger(offset) || offset < 0) {
    errors.push({ field: 'offset', message: 'offset must be a non-negative integer' });
  }

  query.limit = limit;
  query.offset = offset;
  query.unreadOnly = query.unreadOnly === true || query.unreadOnly === 'true';
};

const validateDeliverAdvisory = (req, _res, next) => {
  const errors = [];

  validateUser(req.body.user, errors);
  validateAdvisory(req.body.advisory, 'advisory', errors);

  if (req.body.channels !== undefined) {
    if (!Array.isArray(req.body.channels)) {
      errors.push({ field: 'channels', message: 'channels must be an array' });
    } else {
      req.body.channels.forEach((channel) => {
        if (!supportedChannels.has(channel)) {
          errors.push({ field: 'channels', message: `Unsupported delivery channel: ${channel}` });
        }
      });
    }
  }

  if (errors.length) {
    next(new HttpError(400, 'Validation failed', { errors }));
    return;
  }

  next();
};

const validateDeliverAdvisories = (req, _res, next) => {
  const errors = [];

  validateUser(req.body.user, errors);

  if (!Array.isArray(req.body.advisories)) {
    errors.push({ field: 'advisories', message: 'advisories must be an array' });
  } else {
    req.body.advisories.forEach((advisory, index) => validateAdvisory(advisory, `advisories[${index}]`, errors));
  }

  if (errors.length) {
    next(new HttpError(400, 'Validation failed', { errors }));
    return;
  }

  next();
};

const validateListNotifications = (req, _res, next) => {
  const errors = [];

  if (!uuidPattern.test(req.params.userId)) {
    errors.push({ field: 'userId', message: 'userId must be a valid UUID' });
  }

  validatePagination(req.query, errors);

  if (errors.length) {
    next(new HttpError(400, 'Validation failed', { errors }));
    return;
  }

  next();
};

const validateMarkRead = (req, _res, next) => {
  const errors = [];

  if (!uuidPattern.test(req.params.userId)) {
    errors.push({ field: 'userId', message: 'userId must be a valid UUID' });
  }

  if (!uuidPattern.test(req.params.id)) {
    errors.push({ field: 'id', message: 'id must be a valid UUID' });
  }

  if (errors.length) {
    next(new HttpError(400, 'Validation failed', { errors }));
    return;
  }

  next();
};

module.exports = {
  validateDeliverAdvisory,
  validateDeliverAdvisories,
  validateListNotifications,
  validateMarkRead
};
