const advisorySentRepository = require('../repositories/advisorySent.repository');
const emailNotificationService = require('./emailNotification.service');
const smsNotificationService = require('./smsNotification.service');
const inAppNotificationService = require('./inAppNotification.service');

const getErrorMessage = (error) => error.details?.response || error.message || 'Delivery failed';

const listSmsHistory = async ({ limit, offset } = {}) => (
  advisorySentRepository.findSmsHistory({ limit, offset })
);

const recordAdvisorySent = async ({ advisory, channel, deliveryStatus, errorMessage, providerResult }) => (
  advisorySentRepository.create({
    userId: advisory.userId,
    title: advisory.title,
    message: advisory.message,
    severity: advisory.severity,
    riskScore: advisory.riskScore,
    riskLevel: advisory.riskLevel,
    context: {
      ...advisory.context,
      sourceFactors: advisory.sourceFactors,
      recommendedActions: advisory.recommendedActions,
      providerResult
    },
    channel,
    deliveryStatus,
    emailSent: channel === 'email' && deliveryStatus === 'sent',
    smsSent: channel === 'sms' && deliveryStatus === 'sent',
    errorMessage,
    deliveredAt: deliveryStatus === 'delivered' || deliveryStatus === 'sent' ? new Date() : null
  })
);

const deliverChannel = async ({ advisory, user, channel }) => {
  try {
    if (channel === 'email') {
      const providerResult = await emailNotificationService.sendEmail({
        to: user.email,
        subject: advisory.title,
        text: advisory.message
      });

      const record = await recordAdvisorySent({
        advisory,
        channel,
        deliveryStatus: 'sent',
        providerResult
      });

      return { channel, status: 'sent', advisorySent: record, providerResult };
    }

    if (channel === 'sms') {
      const providerResult = await smsNotificationService.sendSms({
        to: user.phone,
        message: `${advisory.title}: ${advisory.message}`
      });

      const record = await recordAdvisorySent({
        advisory,
        channel,
        deliveryStatus: 'sent',
        providerResult
      });

      return { channel, status: 'sent', advisorySent: record, providerResult };
    }

    if (channel === 'in_app') {
      const record = await recordAdvisorySent({
        advisory,
        channel,
        deliveryStatus: 'delivered'
      });
      const notification = await inAppNotificationService.createInAppNotification({
        userId: advisory.userId,
        advisorySentId: record.id,
        title: advisory.title,
        message: advisory.message,
        metadata: {
          severity: advisory.severity,
          riskScore: advisory.riskScore,
          riskLevel: advisory.riskLevel,
          sourceFactors: advisory.sourceFactors
        }
      });

      return { channel, status: 'delivered', advisorySent: record, notification };
    }

    return { channel, status: 'skipped', reason: 'Unsupported channel' };
  } catch (error) {
    const record = await recordAdvisorySent({
      advisory,
      channel,
      deliveryStatus: 'failed',
      errorMessage: getErrorMessage(error)
    });

    return {
      channel,
      status: 'failed',
      advisorySent: record,
      error: {
        message: error.message,
        details: error.details
      }
    };
  }
};

const deliverAdvisory = async ({ user, advisory, channels = advisory.deliveryChannels } = {}) => {
  const deliveryChannels = Array.from(new Set(channels || []));
  const results = [];

  for (const channel of deliveryChannels) {
    results.push(await deliverChannel({ advisory, user, channel }));
  }

  return {
    userId: advisory.userId,
    advisoryTitle: advisory.title,
    results
  };
};

const deliverAdvisories = async ({ user, advisories = [], channels }) => {
  const results = [];

  for (const advisory of advisories) {
    results.push(await deliverAdvisory({ user, advisory, channels }));
  }

  return {
    userId: user.id || user.user_id || null,
    count: advisories.length,
    results
  };
};

module.exports = {
  deliverAdvisory,
  deliverAdvisories,
  listSmsHistory
};
