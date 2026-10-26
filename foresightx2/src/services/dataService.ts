
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

export interface MonthlyForecast {
  month: number;
  label: string;
  days: number;
  trafficFlow: number | null;
  aqi: number | null;
  temperature: number | null;
  humidity: number | null;
  electricityDemand: number | null;
  maxTrafficFlow: number | null;
  maxAqi: number | null;
  maxTemperature: number | null;
}

export interface YearlyForecastPoint {
  timestamp: string;
  stepAhead: number | null;
  trafficFlow: number | null;
  aqi: number | null;
  temperature: number | null;
  humidity: number | null;
  electricityDemand: number | null;
}

export interface YearlyForecast {
  year: number;
  city: string;
  source: string;
  generatedFrom: string;
  through: string;
  totalHours: number;
  annual: {
    trafficFlow: number | null;
    aqi: number | null;
    temperature: number | null;
    humidity: number | null;
    electricityDemand: number | null;
  };
  peaks: {
    peakTrafficMonth: MonthlyForecast;
    peakAqiMonth: MonthlyForecast;
    peakTemperatureMonth: MonthlyForecast;
    hottestMonth: MonthlyForecast;
    coolestMonth: MonthlyForecast;
  };
  monthly: MonthlyForecast[];
  series: YearlyForecastPoint[];
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'https://foresightx-backend.onrender.com/api';

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

// Render's free tier sleeps and restarts on deploy; while it boots, fetch
// fails with a network error ("Load failed" in Safari) or a 502/503. Retry
// those for up to ~90 s so panels recover by themselves instead of staying
// on "unavailable".
const RETRY_DELAYS_MS = [2000, 4000, 8000, 12000, 16000, 20000, 25000];
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const fetchWithRetry = async (url: string, init: RequestInit): Promise<Response> => {
  for (let attempt = 0; ; attempt += 1) {
    try {
      const response = await fetch(url, init);
      const retryable = response.status === 502 || response.status === 503 || response.status === 504;
      if (!retryable || attempt >= RETRY_DELAYS_MS.length) return response;
    } catch (error) {
      if (attempt >= RETRY_DELAYS_MS.length) throw error;
    }
    await sleep(RETRY_DELAYS_MS[attempt]);
  }
};

const apiRequest = async <T>(path: string, options: RequestInit = {}): Promise<T> => {
  const method = (options.method || 'GET').toUpperCase();
  const init: RequestInit = {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    }
  };
  // Only idempotent reads are retried; writes (SMS, user edits) are sent once.
  const response = method === 'GET'
    ? await fetchWithRetry(`${API_BASE_URL}${path}`, init)
    : await fetch(`${API_BASE_URL}${path}`, init);

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
  modelYearlyForecast: (city = 'bangalore', year = 2026, granularity = 'monthly') => (
    apiRequest<{ data: YearlyForecast }>(
      `/model/forecast-yearly?city=${encodeURIComponent(city)}&year=${encodeURIComponent(year)}&granularity=${encodeURIComponent(granularity)}`
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
  listSmsHistory: (limit = 50) => apiRequest(`/notifications/sms-history?limit=${limit}`),
  markNotificationRead: (userId: string, notificationId: string) => apiRequest(`/notifications/users/${userId}/${notificationId}/read`, {
    method: 'PATCH'
  }),
  runAdvisoryBatch: (payload?: AdvisoryBatchPayload) => apiRequest('/jobs/advisories/run', {
    method: 'POST',
    body: JSON.stringify(payload || {})
  }),

  // ── Decision intelligence (validated alert pipeline + evidence) ──────────
  decision: (hours = 12, profiles: string[] = []) => apiRequest<{ data: Decision }>(
    `/intelligence/decision?hours=${hours}${profiles.length ? `&profiles=${encodeURIComponent(profiles.join(','))}` : ''}`
  ),
  briefing: (profiles: string[] = [], lang: 'en' | 'kn' = 'en', hours = 12) => apiRequest<{ data: Briefing }>(
    `/intelligence/briefing?hours=${hours}&lang=${lang}${profiles.length ? `&profiles=${encodeURIComponent(profiles.join(','))}` : ''}`
  ),
  summary: (profiles: string[] = [], lang: 'en' | 'kn' = 'en') => apiRequest<{ data: { source: string; text: string; evidence: string; aiError?: string } }>(
    `/intelligence/summary?lang=${lang}${profiles.length ? `&profiles=${encodeURIComponent(profiles.join(','))}` : ''}`
  ),
  proof: () => apiRequest<{ data: Proof }>('/intelligence/proof'),
  explain: () => apiRequest<{ data: ShapSummary }>('/intelligence/explain'),
  whatIf: () => apiRequest<{ data: WhatIf }>('/intelligence/whatif'),
  replay: (offset = 0, limit = 600) => apiRequest<{ data: Replay }>(`/intelligence/replay?offset=${offset}&limit=${limit}`),
  liveCheck: (refresh = true) => apiRequest<{ data: LiveCheck }>(`/intelligence/live-check?refresh=${refresh}`),
  ai: (body: { system: string; prompt: string; temperature?: number; maxOutputTokens?: number }) => apiRequest<{ data: { text: string } }>(
    '/intelligence/ai',
    { method: 'POST', body: JSON.stringify(body) }
  )
};

// ── Decision-intelligence types ─────────────────────────────────────────────
export type TargetKey = 'traffic_flow' | 'aqi' | 'temperature' | 'electricity_demand';

export interface Driver {
  key: string;
  label: string;
  effect: number;
  direction: 'up' | 'down';
  share: number;
}

export interface HourValue {
  value: number;
  outlook: number;
  liveAdjustment: number;
  low: number;
  high: number;
  typical: number | null;
  vsTypicalPct: number | null;
  drivers: Driver[];
  category?: string;
}

export interface DecisionHour {
  timestamp: string;
  hour: number;
  values: Partial<Record<TargetKey, HourValue>>;
  probabilities: Record<string, number>;
}

export interface DecisionAlert {
  event: string;
  label: string;
  severity: 'critical' | 'warning' | 'advisory';
  cutoff: number;
  startsAt: string;
  endsAt: string;
  activeHours: number;
  peakAt: string;
  probability: number;
  peakProbability: number;
  leadTarget: TargetKey;
  forecast: { value: number; low: number; high: number; typical: number | null; vsTypicalPct: number | null } | null;
  drivers: Driver[];
  audience: string[];
  messages: Record<string, string>;
  sms: Record<string, string>;
  profile?: string;
  message?: string;
  smsMessage?: string;
}

export interface Decision {
  generatedAt: string;
  horizonHours: number;
  profiles: string[] | null;
  uncertainty: { coverage: number; basis: string; note: string };
  explanationBasis: string | null;
  liveAdjustments: Partial<Record<TargetKey, { bias: number; points: number }>>;
  forecastMix: Record<string, { neural_hybrid: number; last_observation: number; sarima: number }>;
  cutoffs: Record<string, number>;
  hours: DecisionHour[];
  alerts: DecisionAlert[];
  quiet: boolean;
}

export interface Briefing {
  generatedAt: string;
  forecastFor: string;
  lang: 'en' | 'kn';
  sentences: string[];
  english: string[];
  kannada: string[];
  kannadaNote: string;
  alerts: { event: string; label: string; severity: string; probability: number; startsAt: string }[];
}

export interface RuleScore {
  hour_precision: number | null;
  hour_recall: number | null;
  hour_f1: number | null;
  notifications_sent: number;
  notifications_correct: number;
  notification_precision: number | null;
  false_alarms_per_week: number;
  event_episodes: number;
  episodes_caught: number;
  episodes_warned_before_onset: number;
}

export interface Proof {
  manifest: {
    generated_at: string;
    model: string;
    splits: Record<string, { start: string; end: string; hours: number }>;
    sarima_blend_weight_on_hybrid: Record<string, number>;
    versions: Record<string, string>;
  };
  benchmark: {
    window: { start: string; end: string; hours: number };
    block_hours: number;
    verdicts: string[];
    targets: Record<string, {
      label: string;
      unit: string;
      share_hours_unchanged: number;
      models: Record<string, { MAE: number; RMSE: number }>;
      comparisons: Record<string, { skill_mae: number | null; block_win_rate: number | null; block_skills: number[]; dm_test: { statistic: number | null; p_value: number | null } }>;
      changed_hours: { hours: number; model_MAE: number; persistence_MAE: number; skill_vs_persistence: number | null } | null;
    }>;
    daily_resolution: Record<string, { days: number; model_daily_MAE: number; persistence_daily_MAE: number; skill_vs_persistence: number | null; yesterday_mean_MAE: number | null }>;
  };
  conformal: { coverage_target: number; targets: Record<string, { adaptive: { coverage: number; mean_width: number }; static_split: { coverage: number; mean_width: number } }> };
  anomaly: {
    threshold: number;
    threshold_rule: string;
    test_hours: number;
    clean_window: { model_flags_per_week: number; zscore_flags_per_week: number; model_flag_rate: number; zscore_flag_rate: number };
    forecast_error_link?: { mean_norm_error_flagged: number | null; mean_norm_error_unflagged: number; flagged_hours: number };
    injection_tests: { id: string; label: string; incidents: number; detected_by_model: number; detected_by_zscore: number; recall_model: number; recall_zscore: number }[];
    overall_recall_model: number;
    overall_recall_zscore: number;
  };
  replay: {
    window: { start: string; end: string; hours: number };
    cooldownHours: number;
    scorecard: Record<string, { notifications_sent: number; notifications_correct: number; event_episodes: number; episodes_caught: number; episodes_warned_before_onset: number; false_alarms: number; notification_precision: number | null; false_alarms_per_week: number; onset_warning_rate: number | null }>;
    events: Record<string, { label: string; severity: string; cutoff: number; cutoff_rule: string; test_event_hours: number; validation_event_hours: number; brier_score: number; brier_climatology: number; rules: Record<string, RuleScore> }>;
  };
}

export interface ShapSummary {
  method: string;
  targets: Record<string, {
    final_mix: { neural_hybrid: number; last_observation: number; sarima: number };
    global_drivers: { key: string; label: string; mean_abs_effect: number; share: number }[];
    time_context: { key: string; label: string; share: number }[];
    typical_drivers_by_hour: Record<string, Driver[]>;
    additivity_check: { mean_abs_gap: number; network_output_std: number };
  }>;
}

export interface WhatIfScenario {
  id: string;
  driver: string;
  kind: 'scale' | 'shift';
  value: number;
  label: string;
  effects: Record<string, { mean_change: number; mean_change_pct: number | null; p05: number; p95: number; network_mean_change: number; by_hour: Record<string, number> }>;
}

export interface WhatIf {
  method: string;
  caveat: string;
  window: { start: string; end: string; hours: number };
  scenarios: WhatIfScenario[];
  daily_associations: { x: string; y: string; per: number; x_unit: string; y_unit: string; slope: number; ci95: [number, number]; correlation: number; days: number; significant: boolean; sentence: string }[];
  findings: string[];
}

export interface ReplayHour {
  timestamp: string;
  alerts: { event: string; label: string; severity: string; probability: number; audience: string[]; drivers: Driver[]; messages: Record<string, string>; outcome: 'correct' | 'false_alarm' }[];
  probabilities: Record<string, number>;
  event_happened: Record<string, boolean>;
  anomaly_score?: number | null;
  anomaly_flag?: boolean;
  traffic_flow: { actual: number; forecast: number; low: number; high: number };
  aqi: { actual: number; forecast: number; low: number; high: number };
  temperature: { actual: number; forecast: number; low: number; high: number };
  electricity_demand: { actual: number; forecast: number; low: number; high: number };
}

export interface Replay {
  window: { start: string; end: string; hours: number };
  cooldown_hours: number;
  totalHours: number;
  offset: number;
  scorecard: Proof['replay']['scorecard'];
  events: Proof['replay']['events'];
  hours: ReplayHour[];
}

export interface LiveCheckRow {
  observedHour: string;
  observedAt: string;
  aqiCpcb: number | null;
  temperature: number | null;
  humidity: number | null;
  forecastAqi: number | null;
  forecastTemperature: number | null;
  forecastHumidity: number | null;
  aqiCategory?: string;
  intervals?: { aqi: { low: number; high: number } | null; temperature: { low: number; high: number } | null };
  metadata?: { air?: { source: string; station?: string }; weather?: { source: string }; errors?: { provider: string; message: string }[] };
}

export interface LiveCheck {
  generatedAt: string;
  latest: LiveCheckRow | null;
  history: LiveCheckRow[];
  stats: Record<string, { points: number; mae?: number; bias?: number; coverage?: number | null; expectedMae?: number | null; errorRatio?: number | null; status: string }>;
  drift: string[];
  status: 'ok' | 'drift' | 'collecting';
  rule: string;
  notes: string[];
  refreshError: { message: string } | null;
}

// CPCB National AQI bands.
export const cpcbCategory = (aqi: number) => {
  if (aqi <= 50) return 'Good';
  if (aqi <= 100) return 'Satisfactory';
  if (aqi <= 200) return 'Moderate';
  if (aqi <= 300) return 'Poor';
  if (aqi <= 400) return 'Very Poor';
  return 'Severe';
};

// Every timestamp on the dashboard is shown on the Bengaluru clock.
export const fmtCityHour = (ts: string) => new Date(ts).toLocaleTimeString('en-IN', { hour: 'numeric', hour12: true, timeZone: 'Asia/Kolkata' });
export const fmtCityDateTime = (ts: string) => new Date(ts).toLocaleString('en-IN', { month: 'short', day: 'numeric', hour: 'numeric', hour12: true, timeZone: 'Asia/Kolkata' });

export const PROFILE_LABELS: Record<string, string> = {
  general: 'Everyone',
  commuter: 'Commuter',
  outdoor_worker: 'Outdoor worker',
  respiratory: 'Respiratory / heart',
  elderly: 'Older adult',
  child: 'Parent of a child'
};

export const TARGET_META: Record<TargetKey, { label: string; unit: string; color: string; digits: number }> = {
  traffic_flow: { label: 'Traffic', unit: 'veh/hr', color: '#ff8a3d', digits: 0 },
  aqi: { label: 'AQI', unit: 'AQI', color: '#46c9b0', digits: 0 },
  temperature: { label: 'Temperature', unit: '°C', color: '#6fb8e8', digits: 1 },
  electricity_demand: { label: 'Electricity', unit: 'MW', color: '#b48eea', digits: 0 }
};

export const fmtNum = (v: number | null | undefined, digits = 0) => (
  v === null || v === undefined || !Number.isFinite(v) ? '—' : v.toLocaleString('en-IN', { maximumFractionDigits: digits, minimumFractionDigits: digits })
);
export const pct = (v: number | null | undefined, digits = 0) => (
  v === null || v === undefined || !Number.isFinite(v) ? '—' : `${(v * 100).toFixed(digits)}%`
);

// CPCB National AQI bands (101-200 is "Moderate", not "Poor").
const getAqiLabel = (value: number) => {
  if (value <= 50) return 'Good';
  if (value <= 100) return 'Satisfactory';
  if (value <= 200) return 'Moderate';
  if (value <= 300) return 'Poor';
  if (value <= 400) return 'Very Poor';
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
