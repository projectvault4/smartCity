
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
  modelSource?: string;
  forecastFor?: string;
}

export interface ModelConditions {
  city: string;
  source: string;
  forecastFor: string;
  stepAhead: number;
  aqi: {
    provider: string;
    aqi: number | null;
    observedAt: string;
  };
  weather: {
    provider: string;
    weather: {
      main: string;
      description: string;
    };
    temperature: {
      value: number | null;
      units: string;
    };
    humidity: number | null;
    rainLastHourMm: number;
    observedAt: string;
  };
  traffic: {
    provider: string;
    congestionLevel: string;
    flow: number | null;
    observedAt: string;
  };
  raw: {
    electricity_demand?: string;
    [key: string]: string | undefined;
  };
}

export interface DriftData {
  week: number;
  accuracy: number;
  drift: boolean;
  retrained: boolean;
  improved: boolean;
}

export interface ForecastPoint {
  timestamp: string;
  stepAhead: number;
  trafficFlow: number | null;
  aqi: number | null;
  temperature: number | null;
  humidity: number | null;
  electricityDemand: number | null;
  weather: {
    main: string;
    description: string;
  };
  traffic: {
    congestionLevel: string;
  };
}

export interface MultivariatePoint {
  time: string;
  traffic: number;
  aqi: number;
  energy: number;
  trafficRaw: number;
  aqiRaw: number;
  energyRaw: number;
}

export interface MultivariateAnalysis {
  source: string;
  window: {
    hours: number;
    from: string;
    to: string;
  };
  series: MultivariatePoint[];
  stats: {
    phaseLagHours: number;
    phaseLagDirection: string;
    phaseLagCorr: number;
    syncFactor: number;
    coherence: number;
    tempEnergyCorr: number;
    aqiEnergyCorr: number;
  };
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
  modelConditions: (city = 'bangalore', stepAhead = 1) => (
    apiRequest<{ data: ModelConditions }>(
      `/model/conditions?city=${encodeURIComponent(city)}&stepAhead=${encodeURIComponent(stepAhead)}`
    )
  ),
  modelForecast: (city = 'bangalore', steps = 24) => (
    apiRequest<{ data: ForecastPoint[] }>(
      `/model/forecast?city=${encodeURIComponent(city)}&steps=${encodeURIComponent(steps)}`
    )
  ),
  modelMultivariate: (windowHours = 720) => (
    apiRequest<{ data: MultivariateAnalysis }>(
      `/model/multivariate?window=${encodeURIComponent(windowHours)}`
    )
  ),
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

const getAqiLabel = (value: number) => {
  if (value <= 50) return 'Good';
  if (value <= 100) return 'Moderate';
  if (value <= 200) return 'Poor';
  if (value <= 300) return 'Very Poor';
  return 'Severe';
};

const getTrafficLabel = (level?: string) => {
  if (!level) return 'Unknown';
  return level.charAt(0).toUpperCase() + level.slice(1);
};

const getWeatherLabel = (value: number) => {
  if (value >= 38) return 'Heatwave';
  if (value >= 32) return 'Hot';
  if (value >= 24) return 'Warm';
  if (value >= 15) return 'Cool';
  return 'Cold';
};

const getEnergyLabel = (value: number) => {
  if (value >= 900) return 'High';
  if (value >= 550) return 'Moderate';
  return 'Low';
};

const withModelMetric = (
  existing: CityMetric,
  value: number,
  label: string,
  rangePadding: number
): CityMetric => {
  const previous = existing.history[existing.history.length - 1] || value;
  const deltaValue = Math.round(value - previous);
  const trend = deltaValue > 0 ? 'up' : deltaValue < 0 ? 'down' : 'neutral';
  const delta = trend === 'up'
    ? `Up by ${Math.abs(deltaValue)}`
    : trend === 'down'
      ? `Down by ${Math.abs(deltaValue)}`
      : 'Stable';

  return {
    ...existing,
    value,
    label,
    range: [Math.max(0, Math.round(value - rangePadding)), Math.round(value + rangePadding)],
    delta,
    trend,
    history: [...existing.history.slice(1), value]
  };
};

export const cityDataFromModelConditions = (conditions: ModelConditions, previous: CityData): CityData => {
  const trafficFlow = Math.round(Number(conditions.traffic?.flow ?? previous.traffic.value));
  const aqi = Math.round(Number(conditions.aqi?.aqi ?? previous.air.value));
  const temperature = Math.round(Number(conditions.weather?.temperature?.value ?? previous.weather.value));
  const electricity = Math.round(Number(conditions.raw?.electricity_demand ?? previous.energy.value));
  const forecastDate = conditions.forecastFor ? new Date(conditions.forecastFor) : new Date();

  return {
    traffic: withModelMetric(previous.traffic, trafficFlow, getTrafficLabel(conditions.traffic?.congestionLevel), 700),
    air: withModelMetric(previous.air, aqi, getAqiLabel(aqi), 20),
    weather: withModelMetric(previous.weather, temperature, getWeatherLabel(temperature), 3),
    energy: withModelMetric(previous.energy, electricity, getEnergyLabel(electricity), 80),
    timestamp: forecastDate.toLocaleString(),
    lastUpdate: `Trained model forecast T+${conditions.stepAhead}H`,
    modelSource: conditions.source,
    forecastFor: conditions.forecastFor
  };
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
