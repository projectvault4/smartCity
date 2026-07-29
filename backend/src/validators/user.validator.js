const HttpError = require('../utils/HttpError');

const userStatuses = new Set(['active', 'inactive', 'suspended']);

const createFields = new Set([
  'user_id',
  'name',
  'email',
  'phone',
  'age',
  'city',
  'ward',
  'latitude',
  'longitude',
  'preferences',
  'status'
]);

const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const trimString = (value) => (typeof value === 'string' ? value.trim() : value);

const rejectUnknownFields = (body, allowedFields) => {
  const unknownFields = Object.keys(body).filter((field) => !allowedFields.has(field));

  if (unknownFields.length) {
    throw new HttpError(400, 'Request contains unsupported fields', {
      fields: unknownFields
    });
  }
};

const requireString = (body, field, errors, maxLength) => {
  body[field] = trimString(body[field]);

  if (typeof body[field] !== 'string') {
    errors.push({ field, message: `${field} must be a string` });
    return;
  }

  if (!body[field]) {
    errors.push({ field, message: `${field} is required` });
    return;
  }

  if (body[field].length > maxLength) {
    errors.push({ field, message: `${field} must be ${maxLength} characters or fewer` });
  }
};

const validateOptionalString = (body, field, errors, maxLength) => {
  if (!Object.prototype.hasOwnProperty.call(body, field) || body[field] === null) {
    return;
  }

  body[field] = trimString(body[field]);

  if (typeof body[field] !== 'string') {
    errors.push({ field, message: `${field} must be a string` });
    return;
  }

  if (!body[field]) {
    body[field] = null;
    return;
  }

  if (body[field].length > maxLength) {
    errors.push({ field, message: `${field} must be ${maxLength} characters or fewer` });
  }
};

const validateUserBody = (body, { requireFields }) => {
  const errors = [];

  rejectUnknownFields(body, createFields);

  if (requireFields || Object.prototype.hasOwnProperty.call(body, 'user_id')) {
    requireString(body, 'user_id', errors, 80);
  }

  if (requireFields || Object.prototype.hasOwnProperty.call(body, 'name')) {
    requireString(body, 'name', errors, 160);
  }

  validateOptionalString(body, 'email', errors, 255);
  validateOptionalString(body, 'phone', errors, 32);
  validateOptionalString(body, 'city', errors, 120);
  validateOptionalString(body, 'ward', errors, 120);

  if (body.email && !emailPattern.test(body.email)) {
    errors.push({ field: 'email', message: 'email must be valid' });
  }

  if (Object.prototype.hasOwnProperty.call(body, 'age') && body.age !== null) {
    if (!Number.isInteger(body.age) || body.age < 0 || body.age > 130) {
      errors.push({ field: 'age', message: 'age must be an integer between 0 and 130' });
    }
  }

  ['latitude', 'longitude'].forEach((field) => {
    if (!Object.prototype.hasOwnProperty.call(body, field) || body[field] === null) {
      return;
    }

    if (typeof body[field] !== 'number' || Number.isNaN(body[field])) {
      errors.push({ field, message: `${field} must be a number` });
      return;
    }

    const min = field === 'latitude' ? -90 : -180;
    const max = field === 'latitude' ? 90 : 180;

    if (body[field] < min || body[field] > max) {
      errors.push({ field, message: `${field} must be between ${min} and ${max}` });
    }
  });

  if (Object.prototype.hasOwnProperty.call(body, 'preferences')) {
    const isObject = body.preferences !== null && typeof body.preferences === 'object' && !Array.isArray(body.preferences);

    if (!isObject) {
      errors.push({ field: 'preferences', message: 'preferences must be a JSON object' });
    }
  }

  if (Object.prototype.hasOwnProperty.call(body, 'status') && !userStatuses.has(body.status)) {
    errors.push({ field: 'status', message: 'status must be active, inactive, or suspended' });
  }

  if (errors.length) {
    throw new HttpError(400, 'Validation failed', { errors });
  }
};

const validateCreateUser = (req, _res, next) => {
  try {
    validateUserBody(req.body, { requireFields: true });
    next();
  } catch (error) {
    next(error);
  }
};

const validateUpdateUser = ({ requireFields }) => (req, _res, next) => {
  try {
    if (!requireFields && Object.keys(req.body).length === 0) {
      throw new HttpError(400, 'At least one field is required');
    }

    validateUserBody(req.body, { requireFields });
    next();
  } catch (error) {
    next(error);
  }
};

const validateUserId = (req, _res, next) => {
  if (!uuidPattern.test(req.params.id)) {
    next(new HttpError(400, 'User id must be a valid UUID'));
    return;
  }

  next();
};

const validateUserListQuery = (req, _res, next) => {
  const errors = [];
  const limit = Number(req.query.limit || 50);
  const offset = Number(req.query.offset || 0);

  if (!Number.isInteger(limit) || limit < 1 || limit > 100) {
    errors.push({ field: 'limit', message: 'limit must be an integer between 1 and 100' });
  }

  if (!Number.isInteger(offset) || offset < 0) {
    errors.push({ field: 'offset', message: 'offset must be a non-negative integer' });
  }

  if (req.query.status && !userStatuses.has(req.query.status)) {
    errors.push({ field: 'status', message: 'status must be active, inactive, or suspended' });
  }

  if (errors.length) {
    next(new HttpError(400, 'Validation failed', { errors }));
    return;
  }

  req.query.limit = limit;
  req.query.offset = offset;
  req.query.city = trimString(req.query.city);
  req.query.ward = trimString(req.query.ward);
  req.query.search = trimString(req.query.search);

  next();
};

module.exports = {
  validateCreateUser,
  validateUpdateUser,
  validateUserId,
  validateUserListQuery
};
