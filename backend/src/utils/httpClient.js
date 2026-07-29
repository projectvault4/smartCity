const axios = require('axios');
const config = require('../config/env');
const ProviderError = require('./ProviderError');

const sleep = (ms) => new Promise((resolve) => {
  setTimeout(resolve, ms);
});

const isRetryableError = (error) => {
  if (!error.response) {
    return true;
  }

  const status = error.response.status;
  return status === 408 || status === 429 || status >= 500;
};

const getErrorDetails = (error) => ({
  status: error.response?.status,
  data: error.response?.data,
  code: error.code
});

const createHttpClient = ({ provider, baseURL, headers = {}, timeoutMs, retries, retryDelayMs }) => {
  const client = axios.create({
    baseURL,
    timeout: timeoutMs || config.http.timeoutMs,
    headers
  });

  const request = async (options) => {
    const maxRetries = retries ?? config.http.retries;
    const baseDelay = retryDelayMs ?? config.http.retryDelayMs;
    let lastError;

    for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
      try {
        return await client.request(options);
      } catch (error) {
        lastError = error;

        if (attempt === maxRetries || !isRetryableError(error)) {
          break;
        }

        await sleep(baseDelay * 2 ** attempt);
      }
    }

    throw new ProviderError(
      provider,
      `${provider} request failed`,
      getErrorDetails(lastError)
    );
  };

  return {
    request,
    get: (url, options) => request({ ...options, method: 'GET', url }),
    post: (url, data, options) => request({ ...options, method: 'POST', url, data })
  };
};

module.exports = {
  createHttpClient
};
