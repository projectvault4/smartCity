const HttpError = require('../utils/HttpError');

const validateUserObject = (user, field, errors) => {
  if (!user || typeof user !== 'object' || Array.isArray(user)) {
    errors.push({ field, message: `${field} must be an object` });
  }
};

const validateMaxAdvisories = (maxAdvisories, errors) => {
  if (maxAdvisories === undefined) {
    return;
  }

  if (!Number.isInteger(maxAdvisories) || maxAdvisories < 1 || maxAdvisories > 5) {
    errors.push({ field: 'maxAdvisories', message: 'maxAdvisories must be an integer between 1 and 5' });
  }
};

const validateRiskAssessment = (riskAssessment, errors) => {
  if (riskAssessment === undefined) {
    return;
  }

  if (!riskAssessment || typeof riskAssessment !== 'object' || Array.isArray(riskAssessment)) {
    errors.push({ field: 'riskAssessment', message: 'riskAssessment must be an object' });
    return;
  }

  if (!Array.isArray(riskAssessment.factors)) {
    errors.push({ field: 'riskAssessment.factors', message: 'riskAssessment.factors must be an array' });
  }
};

const validateGenerateAdvisory = (req, _res, next) => {
  const errors = [];

  validateUserObject(req.body.user, 'user', errors);
  validateMaxAdvisories(req.body.maxAdvisories, errors);
  validateRiskAssessment(req.body.riskAssessment, errors);

  if (errors.length) {
    next(new HttpError(400, 'Validation failed', { errors }));
    return;
  }

  next();
};

const validateGenerateAdvisories = (req, _res, next) => {
  const errors = [];

  if (!Array.isArray(req.body.users)) {
    errors.push({ field: 'users', message: 'users must be an array' });
  } else {
    req.body.users.forEach((user, index) => validateUserObject(user, `users[${index}]`, errors));
  }

  validateMaxAdvisories(req.body.maxAdvisories, errors);

  if (req.body.sharedConditions !== undefined && (!req.body.sharedConditions || typeof req.body.sharedConditions !== 'object' || Array.isArray(req.body.sharedConditions))) {
    errors.push({ field: 'sharedConditions', message: 'sharedConditions must be an object' });
  }

  if (errors.length) {
    next(new HttpError(400, 'Validation failed', { errors }));
    return;
  }

  next();
};

module.exports = {
  validateGenerateAdvisory,
  validateGenerateAdvisories
};
