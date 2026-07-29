const express = require('express');
const advisoryRoutes = require('./advisory.routes');
const healthRoutes = require('./health.routes');
const jobRoutes = require('./job.routes');
const modelRoutes = require('./model.routes');
const notificationRoutes = require('./notification.routes');
const riskRoutes = require('./risk.routes');
const userRoutes = require('./user.routes');

const router = express.Router();

router.use('/health', healthRoutes);
router.use('/advisories', advisoryRoutes);
router.use('/jobs', jobRoutes);
router.use('/model', modelRoutes);
router.use('/notifications', notificationRoutes);
router.use('/risk', riskRoutes);
router.use('/users', userRoutes);

module.exports = router;
