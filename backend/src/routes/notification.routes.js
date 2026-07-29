const express = require('express');

const notificationController = require('../controllers/notification.controller');
const {
  validateDeliverAdvisory,
  validateDeliverAdvisories,
  validateListNotifications,
  validateMarkRead
} = require('../validators/notification.validator');

const router = express.Router();

router.post('/deliver', validateDeliverAdvisory, notificationController.deliverAdvisory);
router.post('/deliver-batch', validateDeliverAdvisories, notificationController.deliverAdvisories);
router.get('/users/:userId', validateListNotifications, notificationController.listUserNotifications);
router.patch('/users/:userId/:id/read', validateMarkRead, notificationController.markNotificationRead);

module.exports = router;
