const express = require('express');

const userController = require('../controllers/user.controller');
const {
  validateCreateUser,
  validateUpdateUser,
  validateUserId,
  validateUserListQuery
} = require('../validators/user.validator');

const router = express.Router();

router.get('/', validateUserListQuery, userController.listUsers);
router.get('/:id', validateUserId, userController.getUser);
router.post('/', validateCreateUser, userController.createUser);
router.put('/:id', validateUserId, validateUpdateUser({ requireFields: true }), userController.replaceUser);
router.patch('/:id', validateUserId, validateUpdateUser({ requireFields: false }), userController.updateUser);
router.delete('/:id', validateUserId, userController.deleteUser);

module.exports = router;
