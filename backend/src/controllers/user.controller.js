const userRepository = require('../repositories/user.repository');
const asyncHandler = require('../utils/asyncHandler');
const HttpError = require('../utils/HttpError');

const listUsers = asyncHandler(async (req, res) => {
  const result = await userRepository.findAll(req.query);

  res.status(200).json(result);
});

const getUser = asyncHandler(async (req, res) => {
  const user = await userRepository.findById(req.params.id);

  if (!user) {
    throw new HttpError(404, 'User not found');
  }

  res.status(200).json({ data: user });
});

const createUser = asyncHandler(async (req, res) => {
  const user = await userRepository.create(req.body);

  res.status(201).json({ data: user });
});

const replaceUser = asyncHandler(async (req, res) => {
  const user = await userRepository.updateById(req.params.id, req.body);

  if (!user) {
    throw new HttpError(404, 'User not found');
  }

  res.status(200).json({ data: user });
});

const updateUser = asyncHandler(async (req, res) => {
  const user = await userRepository.updateById(req.params.id, req.body);

  if (!user) {
    throw new HttpError(404, 'User not found');
  }

  res.status(200).json({ data: user });
});

const deleteUser = asyncHandler(async (req, res) => {
  const deleted = await userRepository.deleteById(req.params.id);

  if (!deleted) {
    throw new HttpError(404, 'User not found');
  }

  res.status(204).send();
});

module.exports = {
  listUsers,
  getUser,
  createUser,
  replaceUser,
  updateUser,
  deleteUser
};
