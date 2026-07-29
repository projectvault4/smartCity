const twilio = require('twilio');

const config = require('../config/env');
const NotificationError = require('../utils/NotificationError');

const channel = 'sms';

let client;

const getClient = () => {
  const { accountSid, authToken, phoneNumber, messagingServiceSid } = config.notifications.twilio;

  if (!accountSid || !authToken || (!phoneNumber && !messagingServiceSid)) {
    throw new NotificationError(channel, 'Twilio configuration is incomplete');
  }

  if (!client) {
    client = twilio(accountSid, authToken);
  }

  return client;
};

const truncateSms = (message) => {
  if (message.length <= 1500) {
    return message;
  }

  return `${message.slice(0, 1497)}...`;
};

const normalizeRecipientPhoneNumber = (phoneNumber) => {
  if (typeof phoneNumber !== 'string') {
    return '';
  }

  const compact = phoneNumber.trim().replace(/[\s().-]/g, '');

  if (/^[6-9]\d{9}$/.test(compact)) {
    return `+91${compact}`;
  }

  if (/^91[6-9]\d{9}$/.test(compact)) {
    return `+${compact}`;
  }

  return compact;
};

const sendSms = async ({ to, message }) => {
  const recipient = normalizeRecipientPhoneNumber(to);

  if (!recipient) {
    throw new NotificationError(channel, 'Recipient phone number is required');
  }

  if (!/^\+[1-9]\d{7,14}$/.test(recipient)) {
    throw new NotificationError(channel, 'Recipient phone number must include a valid country code');
  }

  try {
    const sender = config.notifications.twilio.messagingServiceSid
      ? { messagingServiceSid: config.notifications.twilio.messagingServiceSid }
      : { from: config.notifications.twilio.phoneNumber };

    const result = await getClient().messages.create({
      body: truncateSms(message),
      to: recipient,
      ...sender
    });

    return {
      channel,
      status: 'sent',
      providerMessageId: result.sid,
      providerStatus: result.status
    };
  } catch (error) {
    throw new NotificationError(channel, 'Twilio SMS delivery failed', {
      status: error.status,
      code: error.code,
      response: error.message
    });
  }
};

module.exports = {
  normalizeRecipientPhoneNumber,
  sendSms
};
