const notificationService = require('../services/notification.service');
const inAppNotificationService = require('../services/inAppNotification.service');
const asyncHandler = require('../utils/asyncHandler');
const HttpError = require('../utils/HttpError');

const deliverAdvisory = asyncHandler(async (req, res) => {
  const result = await notificationService.deliverAdvisory(req.body);

  res.status(200).json({ data: result });
});

const deliverAdvisories = asyncHandler(async (req, res) => {
  const result = await notificationService.deliverAdvisories(req.body);

  res.status(200).json({ data: result });
});

const listUserNotifications = asyncHandler(async (req, res) => {
  const notifications = await inAppNotificationService.listUserNotifications({
    userId: req.params.userId,
    unreadOnly: req.query.unreadOnly,
    limit: req.query.limit,
    offset: req.query.offset
  });

  res.status(200).json({ data: notifications });
});

const markNotificationRead = asyncHandler(async (req, res) => {
  const notification = await inAppNotificationService.markNotificationRead({
    id: req.params.id,
    userId: req.params.userId
  });

  if (!notification) {
    throw new HttpError(404, 'Notification not found');
  }

  res.status(200).json({ data: notification });
});

module.exports = {
  deliverAdvisory,
  deliverAdvisories,
  listUserNotifications,
  markNotificationRead
};
