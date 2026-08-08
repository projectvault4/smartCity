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

const GSM7_REPLACEMENTS = {
  '\u00b0': 'deg',
  '\u2018': "'",
  '\u2019': "'",
  '\u201c': '"',
  '\u201d': '"',
  '\u2013': '-',
  '\u2014': '-',
  '\u2026': '...'
};

const toGsm7Safe = (message) =>
  message
    .replace(/[\u00b0\u2018\u2019\u201c\u201d\u2013\u2014\u2026]/g, (char) => GSM7_REPLACEMENTS[char])
    // Any character outside printable ASCII would switch the whole message to
    // UCS-2 encoding (70 chars/segment), which trial accounts cannot send.
    .replace(/[^\x20-\x7E]/g, ' ')
    .replace(/ {2,}/g, ' ')
    .trim();

// The body is truncated to the configured max chars so the body plus Twilio's
// trial prefix ("Sent from your Twilio trial account - ", 38 chars) stays
// within a single 160-char GSM-7 segment (trial accounts fail with error
// 30044 on multi-segment messages) rather than hard-cutting mid-word.
const truncateSms = (message, maxChars = config.notifications.twilio.maxSmsChars) => {
  if (message.length <= maxChars) {
    return message;
  }

  const suffix = '...';
  const cut = message.slice(0, maxChars - suffix.length).trimEnd();
  const lastSpace = cut.lastIndexOf(' ');
  const truncated = lastSpace > maxChars * 0.5 ? cut.slice(0, lastSpace) : cut;

  return `${truncated.slice(0, maxChars - suffix.length)}${suffix}`;
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
      body: truncateSms(toGsm7Safe(message)),
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
  toGsm7Safe,
  truncateSms,
  sendSms
};
