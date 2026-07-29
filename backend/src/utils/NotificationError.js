class NotificationError extends Error {
  constructor(channel, message, details) {
    super(message);
    this.name = 'NotificationError';
    this.channel = channel;
    this.details = details;
  }
}

module.exports = NotificationError;
