
export interface CityMetric {
  value: number;
  label: string;
  range: [number, number];
  delta: string;
  trend: 'up' | 'down' | 'neutral';
  history: number[];
  unit: string;
}

export interface CityData {
  traffic: CityMetric;
  air: CityMetric;
  weather: CityMetric;
  energy: CityMetric;
  timestamp: string;
  lastUpdate: string;
}

export interface DriftData {
  week: number;
  accuracy: number;
  drift: boolean;
  retrained: boolean;
  improved: boolean;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001/api';

export interface BackendHealth {
  status: string;
  service: string;
  uptimeSeconds?: number;
  startedAt?: string;
}

export interface UserPayload {
  user_id: string;
  name: string;
  email?: string;
  phone?: string;
  age?: number;
  city?: string;
  ward?: string;
  latitude?: number;
  longitude?: number;
  preferences?: Record<string, unknown>;
  status?: 'active' | 'inactive' | 'suspended';
}

export interface RiskAssessmentPayload {
  user: Record<string, unknown>;
  riskGroups?: string[];
  aqi?: number | { aqi: number };
  weather?: Record<string, unknown>;
  traffic?: string | Record<string, unknown>;
  temperature?: number;
}

export interface DeliveryPayload {
  user: Record<string, unknown>;
  advisory: Record<string, unknown>;
  channels?: string[];
}

export interface AdvisoryBatchPayload {
  startedBy?: string;
  channels?: string[];
  sharedConditions?: {
    aqi?: number | { aqi: number };
    weather?: Record<string, unknown>;
    traffic?: string | Record<string, unknown>;
  };
}

const apiRequest = async <T>(path: string, options: RequestInit = {}): Promise<T> => {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    }
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload?.error?.message || payload?.message || 'Backend request failed');
  }

  return payload;
};

export const backendApi = {
  health: () => apiRequest<BackendHealth>('/health'),
  metrics: () => apiRequest('/health/metrics'),
  readiness: () => apiRequest('/health/ready'),
  modelConditions: (city = 'bangalore') => apiRequest(`/model/conditions?city=${encodeURIComponent(city)}`),
  listUsers: () => apiRequest('/users'),
  createUser: (user: UserPayload) => apiRequest('/users', {
    method: 'POST',
    body: JSON.stringify(user)
  }),
  updateUser: (id: string, user: Partial<UserPayload>) => apiRequest(`/users/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(user)
  }),
  deleteUser: (id: string) => apiRequest(`/users/${id}`, {
    method: 'DELETE'
  }),
  assessRisk: (payload: RiskAssessmentPayload) => apiRequest('/risk/assess', {
    method: 'POST',
    body: JSON.stringify(payload)
  }),
  generateAdvisory: (payload: RiskAssessmentPayload) => apiRequest('/advisories/generate', {
    method: 'POST',
    body: JSON.stringify(payload)
  }),
  deliverNotification: (payload: DeliveryPayload) => apiRequest('/notifications/deliver', {
    method: 'POST',
    body: JSON.stringify(payload)
  }),
  listNotifications: (userId: string) => apiRequest(`/notifications/users/${userId}`),
  markNotificationRead: (userId: string, notificationId: string) => apiRequest(`/notifications/users/${userId}/${notificationId}/read`, {
    method: 'PATCH'
  }),
  runAdvisoryBatch: (payload?: AdvisoryBatchPayload) => apiRequest('/jobs/advisories/run', {
    method: 'POST',
    body: JSON.stringify(payload || {})
  })
};

export const getDriftData = (): DriftData[] => [
  { week: 1, accuracy: 94.2, drift: false, retrained: false, improved: false },
  { week: 2, accuracy: 91.7, drift: true, retrained: false, improved: false },
  { week: 3, accuracy: 93.1, drift: false, retrained: true, improved: false },
  { week: 4, accuracy: 95.4, drift: false, retrained: false, improved: true },
];

const generateHistory = (base: number, length: number) => {
  return Array.from({ length }, () => base + (Math.random() - 0.5) * (base * 0.1));
};

export const getInitialData = (): CityData => {
  const now = new Date();
  const oneHourAgo = new Date(now.getTime() - 3600000);

  return {
    traffic: {
      value: 5670,
      label: 'Light',
      range: [4884, 6296],
      delta: '↓ Down by 1384',
      trend: 'down',
      history: generateHistory(5670, 10),
      unit: 'vehicles/hr'
    },
    air: {
      value: 156,
      label: 'Poor',
      range: [145, 165],
      delta: '↑ Up by 12',
      trend: 'up',
      history: generateHistory(156, 10),
      unit: 'AQI'
    },
    weather: {
      value: 28,
      label: 'Warm',
      range: [26, 30],
      delta: '→ Stable',
      trend: 'neutral',
      history: generateHistory(28, 10),
      unit: '°C'
    },
    energy: {
      value: 847,
      label: 'High',
      range: [800, 900],
      delta: '→ Stable',
      trend: 'neutral',
      history: generateHistory(847, 10),
      unit: 'MW'
    },
    timestamp: now.toLocaleString(),
    lastUpdate: oneHourAgo.toLocaleString(),
  };
};

export const updateMetric = (metric: CityMetric): CityMetric => {
  const volatility = 0.02; // 2% fluctuation
  const change = (Math.random() - 0.5) * (metric.value * volatility);
  const newValue = Math.max(0, Math.round(metric.value + change));
  
  const history = [...metric.history.slice(1), newValue];
  const deltaValue = newValue - (metric.history[metric.history.length - 1] || newValue);
  const trend = deltaValue > 0 ? 'up' : deltaValue < 0 ? 'down' : 'neutral';
  const delta = trend === 'up' ? `↑ Up by ${Math.abs(deltaValue)}` : trend === 'down' ? `↓ Down by ${Math.abs(deltaValue)}` : '→ Stable';

  // Update label based on value
  let label = metric.label;
  if (metric.unit === 'AQI') {
    if (newValue < 50) label = 'Good';
    else if (newValue < 100) label = 'Moderate';
    else if (newValue < 150) label = 'Unhealthy';
    else if (newValue < 200) label = 'Poor';
    else label = 'Hazardous';
  } else if (metric.unit === 'vehicles/hr') {
    if (newValue < 3000) label = 'Light';
    else if (newValue < 7000) label = 'Moderate';
    else label = 'Heavy';
  }

  return {
    ...metric,
    value: newValue,
    label,
    delta,
    trend,
    history
  };
};
