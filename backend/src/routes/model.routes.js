const express = require('express');

const modelController = require('../controllers/model.controller');

const router = express.Router();

router.get('/conditions', modelController.getModelConditions);

module.exports = router;
