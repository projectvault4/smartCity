class ProviderError extends Error {
  constructor(provider, message, details) {
    super(message);
    this.name = 'ProviderError';
    this.provider = provider;
    this.details = details;
  }
}

module.exports = ProviderError;
