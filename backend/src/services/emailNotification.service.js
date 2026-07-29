const sgMail = require('@sendgrid/mail');

const config = require('../config/env');
const NotificationError = require('../utils/NotificationError');

const channel = 'email';

const requireConfig = () => {
  const { apiKey, fromEmail } = config.notifications.sendGrid;

  if (!apiKey || !fromEmail) {
    throw new NotificationError(channel, 'SendGrid configuration is incomplete');
  }

  sgMail.setApiKey(apiKey);
};

const sendEmail = async ({ to, subject, text, html }) => {
  if (!to) {
    throw new NotificationError(channel, 'Recipient email is required');
  }

  requireConfig();

  const { fromEmail, fromName } = config.notifications.sendGrid;

  try {
    const [response] = await sgMail.send({
      to,
      from: {
        email: fromEmail,
        name: fromName
      },
      subject,
      text,
      html: html || `<p>${text}</p>`
    });

    return {
      channel,
      status: 'sent',
      providerMessageId: response.headers?.['x-message-id'] || null,
      providerStatusCode: response.statusCode || null
    };
  } catch (error) {
    throw new NotificationError(channel, 'SendGrid email delivery failed', {
      status: error.code,
      response: error.response?.body || error.message
    });
  }
};

module.exports = {
  sendEmail
};
