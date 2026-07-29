const { calculateRisk } = require('./riskEngine.service');

const severityRank = {
  critical: 4,
  warning: 3,
  medium: 3,
  low: 2,
  info: 1
};

const groupLabels = {
  resp: 'respiratory sensitivity',
  elder: 'elderly citizen',
  child: 'child',
  worker: 'outdoor worker',
  commuter: 'commuter',
  general: 'general'
};

const actionTemplates = {
  aqi_very_poor: [
    'Stay indoors where possible.',
    'Keep windows closed and reduce outdoor exertion.',
    'Use a mask if travel is unavoidable.'
  ],
  aqi_poor: [
    'Limit long outdoor activity.',
    'Watch for breathing discomfort.',
    'Prefer cleaner indoor spaces until conditions improve.'
  ],
  respiratory_very_poor_aqi: [
    'Avoid outdoor exposure.',
    'Keep medication or inhalers accessible if prescribed.',
    'Seek medical help if symptoms worsen.'
  ],
  child_very_poor_aqi: [
    'Prefer indoor recess and indoor sports.',
    'Avoid roadside exposure during school commute.',
    'Keep caregivers informed about symptoms.'
  ],
  worker_very_poor_aqi: [
    'Use appropriate PPE outdoors.',
    'Take frequent breaks in cleaner-air areas.',
    'Reduce heavy exertion during peak pollution.'
  ],
  elder_extreme_temperature: [
    'Stay hydrated and rest in a cool or warm indoor space.',
    'Avoid strenuous outdoor activity.',
    'Check in with a caregiver if discomfort starts.'
  ],
  rain_severe_traffic: [
    'Expect delays and leave extra travel time.',
    'Avoid non-essential trips where possible.',
    'Use safer routes and watch for waterlogging.'
  ],
  commuter_rain_severe_traffic: [
    'Delay non-essential travel.',
    'Consider public transit or alternate routes.',
    'Share your ETA if your commute is affected.'
  ],
  default: [
    'Stay alert to changing local conditions.',
    'Follow civic advisories for your area.'
  ]
};

const getDisplayName = (user = {}) => user.name || user.user_id || 'there';

const getLocationText = (user = {}, assessment = {}) => {
  if (user.ward && user.city) {
    return `${user.ward}, ${user.city}`;
  }

  if (user.ward) {
    return user.ward;
  }

  if (user.city) {
    return user.city;
  }

  return assessment.inputs?.aqi?.location?.name || 'your area';
};

const getUserRiskGroups = (assessment = {}) => (
  assessment.inputs?.riskGroups || []
).map((group) => groupLabels[group] || group);

const formatConditions = (assessment = {}) => {
  const parts = [];
  const aqi = assessment.inputs?.aqi;
  const weather = assessment.inputs?.weather;
  const traffic = assessment.inputs?.traffic;

  if (aqi?.value !== null && aqi?.value !== undefined) {
    parts.push(`AQI ${aqi.value} (${String(aqi.category || 'unknown').replace('_', ' ')})`);
  }

  if (weather?.temperatureCelsius !== null && weather?.temperatureCelsius !== undefined) {
    parts.push(`${weather.temperatureCelsius}C`);
  }

  if (weather?.condition) {
    parts.push(String(weather.condition).replace('_', ' '));
  }

  if (traffic?.level && traffic.level !== 'unknown') {
    parts.push(`${traffic.level} traffic`);
  }

  return parts.length ? parts.join(', ') : 'local conditions';
};

const buildSummary = (user, assessment) => {
  const name = getDisplayName(user);
  const location = getLocationText(user, assessment);
  const conditions = formatConditions(assessment);

  return `Dear ${name}, current conditions in ${location} indicate ${assessment.riskLevel.toLowerCase()} risk (${conditions}).`;
};

const dedupeActions = (actions) => Array.from(new Set(actions));

const getActions = (sourceFactors) => {
  const actions = sourceFactors.flatMap((factor) => actionTemplates[factor.code] || actionTemplates.default);
  return dedupeActions(actions).slice(0, 5);
};

const getPrimaryTitle = (sourceFactors, assessment) => {
  const titles = sourceFactors.map((factor) => factor.title);

  if (titles.includes('Commute Alert') && titles.includes('School Indoor Recess Alert')) {
    return 'School Indoor Recess + Commute Delay Alert';
  }

  if (titles.includes('Extreme Weather Risk')) {
    return 'Extreme Heat Alert - Hydration & Rest Reminder';
  }

  if (titles.includes('Field Safety Alert')) {
    return 'Field Safety Alert - Mandatory PPE & Frequent Breaks';
  }

  if (titles.includes('Commute Alert')) {
    return 'Commute Alert - Severe Delays Expected';
  }

  if (titles.includes('School Indoor Recess Alert')) {
    return 'School Indoor Recess Alert';
  }

  return titles[0] || `${assessment.riskLevel} Risk Advisory`;
};

const getDeliveryChannels = (user = {}) => {
  const preferences = user.preferences || {};
  const channels = [];

  if (preferences.inApp !== false) {
    channels.push('in_app');
  }

  if (preferences.email === true && user.email) {
    channels.push('email');
  }

  if (preferences.sms === true && user.phone) {
    channels.push('sms');
  }

  if (preferences.push === true) {
    channels.push('push');
  }

  return channels;
};

const getTopFactors = (assessment, maxAdvisories) => (
  assessment.factors || []
)
  .slice()
  .sort((left, right) => {
    const severityDelta = (severityRank[right.severity] || 0) - (severityRank[left.severity] || 0);
    return severityDelta || right.score - left.score;
  })
  .slice(0, maxAdvisories);

const buildAdvisory = ({ user, assessment, sourceFactors }) => {
  const title = getPrimaryTitle(sourceFactors, assessment);
  const summary = buildSummary(user, assessment);
  const actions = getActions(sourceFactors);
  const riskGroups = getUserRiskGroups(assessment);
  const personalizedContext = riskGroups.length
    ? `This advisory is personalized for your ${riskGroups.join(', ')} risk profile.`
    : 'This advisory is personalized for your current local conditions.';

  return {
    userId: user.id || user.user_id || null,
    personalizedFor: getDisplayName(user),
    title,
    message: `${summary} ${personalizedContext} Recommended actions: ${actions.join(' ')}`,
    severity: assessment.severity,
    riskScore: assessment.score,
    riskLevel: assessment.riskLevel,
    sourceFactors: sourceFactors.map((factor) => factor.code),
    recommendedActions: actions,
    deliveryChannels: getDeliveryChannels(user),
    context: {
      inputs: assessment.inputs,
      factors: sourceFactors
    }
  };
};

const generateForUser = ({ user = {}, riskAssessment, maxAdvisories = 1, ...riskInput } = {}) => {
  const assessment = riskAssessment || calculateRisk({ user, ...riskInput });
  const topFactors = getTopFactors(assessment, maxAdvisories);

  if (!topFactors.length || assessment.score <= 0) {
    return {
      userId: user.id || user.user_id || null,
      riskAssessment: assessment,
      advisories: []
    };
  }

  return {
    userId: user.id || user.user_id || null,
    riskAssessment: assessment,
    advisories: [
      buildAdvisory({
        user,
        assessment,
        sourceFactors: topFactors
      })
    ]
  };
};

const generateForUsers = ({ users = [], sharedConditions = {}, maxAdvisories = 1 } = {}) => ({
  count: users.length,
  results: users.map((user) => generateForUser({
    user,
    maxAdvisories,
    ...sharedConditions,
    riskGroups: user.riskGroups || user.groups || sharedConditions.riskGroups
  }))
});

module.exports = {
  generateForUser,
  generateForUsers
};
