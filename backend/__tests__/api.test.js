const request = require('supertest');

jest.mock('../src/repositories/user.repository', () => ({
  findAll: jest.fn(),
  findById: jest.fn(),
  create: jest.fn(),
  updateById: jest.fn(),
  deleteById: jest.fn(),
  findActiveWithRiskGroups: jest.fn()
}));

jest.mock('../src/services/notification.service', () => ({
  deliverAdvisory: jest.fn(),
  deliverAdvisories: jest.fn()
}));

const app = require('../src/app');
const userRepository = require('../src/repositories/user.repository');
const notificationService = require('../src/services/notification.service');

const userId = '00000000-0000-4000-8000-000000000001';

describe('API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('GET /api/health returns service health and request id', async () => {
    const response = await request(app)
      .get('/api/health')
      .set('x-request-id', 'test-request-id')
      .expect(200);

    expect(response.headers['x-request-id']).toBe('test-request-id');
    expect(response.body).toEqual(expect.objectContaining({
      status: 'ok',
      service: 'foresightx-backend'
    }));
  });

  test('GET /api/health/metrics returns process metrics', async () => {
    const response = await request(app)
      .get('/api/health/metrics')
      .expect(200);

    expect(response.body).toEqual(expect.objectContaining({
      process: expect.objectContaining({
        uptimeSeconds: expect.any(Number)
      }),
      memory: expect.objectContaining({
        heapUsed: expect.any(Number)
      })
    }));
  });

  test('unknown route returns centralized 404 error shape', async () => {
    const response = await request(app)
      .get('/api/missing')
      .expect(404);

    expect(response.body).toEqual(expect.objectContaining({
      error: expect.objectContaining({
        statusCode: 404,
        message: 'Route not found: /api/missing',
        requestId: expect.any(String)
      })
    }));
  });

  test('POST /api/risk/assess validates payloads', async () => {
    const response = await request(app)
      .post('/api/risk/assess')
      .send({ riskGroups: 'elder' })
      .expect(400);

    expect(response.body.error.details.errors).toEqual(expect.arrayContaining([
      expect.objectContaining({ field: 'user' }),
      expect.objectContaining({ field: 'riskGroups' })
    ]));
  });

  test('POST /api/risk/assess calculates risk', async () => {
    const response = await request(app)
      .post('/api/risk/assess')
      .send({
        user: { id: userId, age: 68 },
        riskGroups: ['elder'],
        aqi: { aqi: 40 },
        weather: { weather: { main: 'Clear' }, temperature: { value: 40 } },
        traffic: { congestionLevel: 'light' }
      })
      .expect(200);

    expect(response.body.data).toEqual(expect.objectContaining({
      score: 2,
      severity: 'critical'
    }));
  });

  test('GET /api/users returns paginated users', async () => {
    userRepository.findAll.mockResolvedValue({
      data: [{ id: userId, user_id: 'user_001', name: 'Ravi Kumar' }],
      meta: { total: 1, limit: 10, offset: 0 }
    });

    const response = await request(app)
      .get('/api/users?limit=10&offset=0')
      .expect(200);

    expect(userRepository.findAll).toHaveBeenCalledWith(expect.objectContaining({
      limit: 10,
      offset: 0
    }));
    expect(response.body.data).toHaveLength(1);
  });

  test('POST /api/users rejects invalid email', async () => {
    const response = await request(app)
      .post('/api/users')
      .send({
        user_id: 'user_001',
        name: 'Ravi Kumar',
        email: 'not-an-email'
      })
      .expect(400);

    expect(response.body.error.details.errors).toEqual(expect.arrayContaining([
      expect.objectContaining({ field: 'email' })
    ]));
  });

  test('POST /api/users creates a valid user', async () => {
    userRepository.create.mockResolvedValue({
      id: userId,
      user_id: 'user_001',
      name: 'Ravi Kumar',
      email: 'ravi@example.com'
    });

    const response = await request(app)
      .post('/api/users')
      .send({
        user_id: 'user_001',
        name: 'Ravi Kumar',
        email: 'ravi@example.com'
      })
      .expect(201);

    expect(userRepository.create).toHaveBeenCalledWith(expect.objectContaining({
      user_id: 'user_001',
      email: 'ravi@example.com'
    }));
    expect(response.body.data.id).toBe(userId);
  });

  test('POST /api/advisories/generate produces a personalized advisory', async () => {
    const response = await request(app)
      .post('/api/advisories/generate')
      .send({
        user: {
          id: userId,
          name: 'Ravi Kumar',
          age: 68,
          ward: 'Anekal Ward',
          city: 'Bangalore',
          preferences: { inApp: true }
        },
        riskGroups: ['elder'],
        aqi: { aqi: 40 },
        weather: { weather: { main: 'Clear' }, temperature: { value: 40 } },
        traffic: { congestionLevel: 'light' }
      })
      .expect(200);

    expect(response.body.data.advisories[0]).toEqual(expect.objectContaining({
      personalizedFor: 'Ravi Kumar',
      severity: 'critical'
    }));
  });

  test('POST /api/notifications/deliver delegates delivery service', async () => {
    notificationService.deliverAdvisory.mockResolvedValue({
      userId,
      advisoryTitle: 'Risk Alert',
      results: [{ channel: 'in_app', status: 'delivered' }]
    });

    const response = await request(app)
      .post('/api/notifications/deliver')
      .send({
        user: { id: userId, email: 'ravi@example.com' },
        advisory: {
          userId,
          title: 'Risk Alert',
          message: 'Stay alert.',
          severity: 'warning',
          deliveryChannels: ['in_app']
        }
      })
      .expect(200);

    expect(notificationService.deliverAdvisory).toHaveBeenCalled();
    expect(response.body.data.results[0].status).toBe('delivered');
  });
});
