const { normalizeRecipientPhoneNumber } = require('../src/services/smsNotification.service');

describe('SMS notification service', () => {
  test('normalizes Indian recipient phone numbers for Twilio', () => {
    expect(normalizeRecipientPhoneNumber('9876543210')).toBe('+919876543210');
    expect(normalizeRecipientPhoneNumber('919876543210')).toBe('+919876543210');
    expect(normalizeRecipientPhoneNumber('+91 98765-43210')).toBe('+919876543210');
  });

  test('keeps already international recipient phone numbers', () => {
    expect(normalizeRecipientPhoneNumber('+14155552671')).toBe('+14155552671');
  });

  test('sanitizes non-GSM-7 characters that would force UCS-2 encoding', () => {
    const { toGsm7Safe } = require('../src/services/smsNotification.service');
    expect(toGsm7Safe('45°C heat — limit exposure')).toBe('45degC heat - limit exposure');
  });
});
