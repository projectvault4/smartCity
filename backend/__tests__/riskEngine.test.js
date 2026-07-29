const {
  calculateRisk,
  getAqiCategory,
  getRiskLevel,
  normalizeRiskGroups
} = require('../src/services/riskEngine.service');

describe('Risk Engine', () => {
  test('categorizes AQI values', () => {
    expect(getAqiCategory({ aqi: 45 })).toBe('good');
    expect(getAqiCategory({ aqi: 125 })).toBe('poor');
    expect(getAqiCategory({ aqi: 240 })).toBe('very_poor');
    expect(getAqiCategory({ aqi: 330 })).toBe('severe');
  });

  test('maps scores to risk levels from the implementation guide', () => {
    expect(getRiskLevel(4.99)).toBe('Low');
    expect(getRiskLevel(5)).toBe('Medium');
    expect(getRiskLevel(8)).toBe('High');
  });

  test('normalizes risk group aliases', () => {
    expect(Array.from(normalizeRiskGroups(['respiratory', 'elderly', 'commuters']))).toEqual([
      'resp',
      'elder',
      'commuter'
    ]);
  });

  test('calculates elderly heat risk', () => {
    const result = calculateRisk({
      user: { id: 'user_001', age: 68 },
      riskGroups: ['elder'],
      aqi: { aqi: 40 },
      weather: { temperature: { value: 40 }, weather: { main: 'Clear' } },
      traffic: { congestionLevel: 'light' }
    });

    expect(result.score).toBe(2);
    expect(result.severity).toBe('critical');
    expect(result.factors).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: 'elder_extreme_temperature' })
    ]));
  });

  test('calculates child AQI risk', () => {
    const result = calculateRisk({
      user: { id: 'user_002', age: 12 },
      riskGroups: ['child', 'commuter'],
      aqi: { aqi: 260 },
      weather: { weather: { main: 'Fog' }, temperature: { value: 15 } },
      traffic: { congestionLevel: 'heavy' }
    });

    expect(result.score).toBe(5);
    expect(result.riskLevel).toBe('Medium');
    expect(result.factors).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: 'aqi_very_poor' }),
      expect.objectContaining({ code: 'child_very_poor_aqi' })
    ]));
  });

  test('allows weather hazards to stack for rain, heat, and traffic', () => {
    const result = calculateRisk({
      user: { id: 'user_003', age: 70 },
      riskGroups: ['elder', 'commuter'],
      aqi: { aqi: 240 },
      weather: {
        weather: { main: 'Rain', description: 'heavy rain' },
        temperature: { value: 40 },
        rainLastHourMm: 8
      },
      traffic: { congestionLevel: 'severe' }
    });

    expect(result.score).toBe(10);
    expect(result.riskLevel).toBe('High');
    expect(result.factors).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: 'aqi_very_poor' }),
      expect.objectContaining({ code: 'elder_extreme_temperature' }),
      expect.objectContaining({ code: 'rain_severe_traffic' }),
      expect.objectContaining({ code: 'commuter_rain_severe_traffic' })
    ]));
  });
});
