const notificationRepository = require('../repositories/notification.repository');
const NotificationError = require('../utils/NotificationError');

const channel = 'in_app';

const createInAppNotification = async ({ userId, advisorySentId, title, message, metadata = {} }) => {
  if (!userId) {
    throw new NotificationError(channel, 'userId is required for in-app notifications');
  }

  return notificationRepository.create({
    userId,
    advisorySentId,
    title,
    message,
    channel,
    status: 'delivered',
    isRead: false,
    metadata
  });
};

const listUserNotifications = async ({ userId, unreadOnly, limit, offset }) => (
  notificationRepository.findByUserId({ userId, unreadOnly, limit, offset })
);

const markNotificationRead = async ({ id, userId }) => (
  notificationRepository.markRead({ id, userId })
);

module.exports = {
  createInAppNotification,
  listUserNotifications,
  markNotificationRead
};
