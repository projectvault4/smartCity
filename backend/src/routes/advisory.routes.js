const express = require('express');

const advisoryController = require('../controllers/advisory.controller');
const {
  validateGenerateAdvisory,
  validateGenerateAdvisories
} = require('../validators/advisory.validator');

const router = express.Router();

router.post('/generate', validateGenerateAdvisory, advisoryController.generateAdvisory);
router.post('/generate-batch', validateGenerateAdvisories, advisoryController.generateAdvisories);

module.exports = router;
