const HttpError = require('../utils/HttpError');

const validateNumber = (value, field, errors, min, max) => {
  if (value === undefined || value === null) {
    return;
  }

  if (typeof value !== 'number' || Number.isNaN(value) || value < min || value > max) {
    errors.push({ field, message: `${field} must be a number between ${min} and ${max}` });
  }
};

const validateRiskAssessment = (req, _res, next) => {
  const errors = [];
  const { user, riskGroups, aqi, weather, temperature, traffic } = req.body;

  if (!user || typeof user !== 'object' || Array.isArray(user)) {
    errors.push({ field: 'user', message: 'user must be an object' });
  }

  if (riskGroups !== undefined && !Array.isArray(riskGroups)) {
    errors.push({ field: 'riskGroups', message: 'riskGroups must be an array' });
  }

  if (aqi !== undefined && typeof aqi !== 'number' && (typeof aqi !== 'object' || Array.isArray(aqi))) {
    errors.push({ field: 'aqi', message: 'aqi must be a number or normalized AQI object' });
  }

  if (weather !== undefined && (typeof weather !== 'object' || Array.isArray(weather))) {
    errors.push({ field: 'weather', message: 'weather must be a normalized weather object' });
  }

  if (traffic !== undefined && typeof traffic !== 'string' && (typeof traffic !== 'object' || Array.isArray(traffic))) {
    errors.push({ field: 'traffic', message: 'traffic must be a string or normalized traffic object' });
  }

  validateNumber(temperature, 'temperature', errors, -80, 80);

  if (errors.length) {
    next(new HttpError(400, 'Validation failed', { errors }));
    return;
  }

  next();
};

module.exports = {
  validateRiskAssessment
};
