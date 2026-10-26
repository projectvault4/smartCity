import { useEffect, useMemo, useState } from 'react';
import { backendApi, Decision, ForecastPoint, TargetKey } from '../services/dataService';

// ─── Data (mirrors city-pulse-forecast.html) ─────────────────────────────────
const HORIZONS = ['T+1H', 'T+2H', 'T+3H', 'T+4H', 'T+5H', 'T+6H'];
const STEPS = 6;
// Metric card key -> decision-engine target (for the 90% ranges).
const METRIC_TARGET: Record<string, TargetKey> = {
  traffic: 'traffic_flow',
  aqi: 'aqi',
  energy: 'electricity_demand',
  weather: 'temperature'
};

interface Metric {
  key: string;
  label: string;
  unit: string;
  color: string;
  values: number[];
  timestamps: string[];
}

// Fallback used when the model forecast isn't available (backend offline)
const FALLBACK_METRICS: Metric[] = [
  { key: 'traffic', label: 'Traffic', unit: 'veh/hr', color: '#ff8a3d', timestamps: [], values: [26573, 27918, 25642, 24292, 22860, 23980] },
  { key: 'aqi',     label: 'AQI',     unit: 'AQI',    color: '#46c9b0', timestamps: [], values: [85, 73, 70, 82, 78, 74] },
  { key: 'energy',  label: 'Energy',  unit: 'MW',     color: '#b48eea', timestamps: [], values: [8032, 8059, 8535, 8627, 8710, 8550] },
  { key: 'weather', label: 'Weather', unit: '°C',     color: '#6fb8e8', timestamps: [], values: [18, 18, 18, 18, 19, 19] },
];

const fmt = (n: number) => (n >= 1000 ? n.toLocaleString() : String(n));

function deltaTag(curr: number, prev: number): { text: string; cls: 'up' | 'down' | 'flat' } {
  const diff = curr - prev;
  if (Math.abs(diff) < prev * 0.002) return { text: '→ steady', cls: 'flat' };
  const pct = ((diff / prev) * 100).toFixed(1);
  return diff > 0
    ? { text: `↑ +${pct}%`, cls: 'up' }
    : { text: `↓ ${pct}%`, cls: 'down' };
}

// ─── Sparkline ────────────────────────────────────────────────────────────────
// Scaled by value/max (same scale as the per-row value bars) so the line,
// the bars, and the numbers all agree.
function Spark({ values, color }: { values: number[]; color: string }) {
  const w = 280, h = 64, pad = 6;
  const max = Math.max(...values) || 1;
  const pts = values.map((v, i) => ({
    x: pad + (i / (values.length - 1)) * (w - pad * 2),
    y: h - pad - (v / max) * (h - pad * 2),
  }));
  const line = pts.map((p, i) => (i === 0 ? 'M' : 'L') + p.x.toFixed(1) + ',' + p.y.toFixed(1)).join(' ');
  const area = line + ` L${pts[pts.length - 1].x.toFixed(1)},${h} L${pts[0].x.toFixed(1)},${h} Z`;

  return (
    <svg className="w-full h-16 mb-4" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <path d={area} fill={color} opacity={0.16} />
      <path d={line} fill="none" stroke={color} strokeWidth={2} />
      {pts.map((p, i) => (
        <circle key={i} cx={p.x.toFixed(1)} cy={p.y.toFixed(1)} r={3} fill={color} />
      ))}
    </svg>
  );
}

// ─── Time formatting ──────────────────────────────────────────────────────────
function fmtHour(ts: string): string {
  const d = new Date(ts);

  return d.toLocaleTimeString('en-IN', {
    hour: 'numeric',
    hour12: true,
    timeZone: 'Asia/Kolkata'
  });
}

function buildForecastHours(count: number): string[] {
  const base = new Date();
  base.setMinutes(0, 0, 0);
  base.setHours(base.getHours() + 1);

  return Array.from({ length: count }, (_, index) => {
    const d = new Date(base);
    d.setHours(base.getHours() + index);
    return d.toLocaleTimeString([], {
      hour: 'numeric',
      hour12: true
    });
  });
}

function fmtForecastTs(ts: string): string {
  const d = new Date(ts);

  return d.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true
  }).replace(',', ' ·');
}

const DELTA_CLASS: Record<string, string> = {
  up:   'color:#8fe6c9;background:rgba(70,201,176,0.12)',
  down: 'color:#ffb08a;background:rgba(255,138,61,0.12)',
  flat: 'color:#6f8079;background:rgba(255,255,255,0.05)',
};

// ─── Build metrics from model forecast series ─────────────────────────────────
function metricsFromForecast(forecast: ForecastPoint[]): Metric[] {
  const points = forecast.slice(0, STEPS);
  const timestamps = points.map((p) => p.timestamp);

  return [
    {
      key: 'traffic', label: 'Traffic', unit: 'veh/hr', color: '#ff8a3d', timestamps,
      values: points.map((p) => Math.round(p.trafficFlow ?? 0)),
    },
    {
      key: 'aqi', label: 'AQI', unit: 'AQI', color: '#46c9b0', timestamps,
      values: points.map((p) => Math.round(p.aqi ?? 0)),
    },
    {
      key: 'energy', label: 'Energy', unit: 'MW', color: '#b48eea', timestamps,
      values: points.map((p) => Math.round(p.electricityDemand ?? 0)),
    },
    {
      key: 'weather', label: 'Weather', unit: '°C', color: '#6fb8e8', timestamps,
      values: points.map((p) => Math.round(p.temperature ?? 0)),
    },
  ];
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function CityPulseForecast({
  forecast,
  city = 'bangalore',
}: {
  forecast?: ForecastPoint[] | null;
  city?: string;
}) {
  const [fetched, setFetched] = useState<ForecastPoint[] | null>(null);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [status, setStatus] = useState<'model' | 'fallback' | 'loading'>('loading');

  useEffect(() => {
    let mounted = true;
    backendApi.modelForecast(city, STEPS)
      .then((response) => {
        if (!mounted) return;
        setFetched(response.data);
        setStatus('model');
      })
      .catch(() => { if (mounted) setStatus('fallback'); });
    backendApi.decision(STEPS)
      .then((response) => { if (mounted) setDecision(response.data); })
      .catch(() => {});
    return () => { mounted = false; };
  }, [city]);

  // Strongest event probability per hour, and whether any alert fires then.
  const hourRisk = useMemo(() => (decision?.hours || []).map((h) => {
    const entries = Object.entries(h.probabilities);
    const top = entries.reduce((best, cur) => (cur[1] > best[1] ? cur : best), ['', 0] as [string, number]);
    const fires = entries.some(([ev, p]) => p >= (decision?.cutoffs[ev] ?? 1));
    return { p: top[1], fires };
  }), [decision]);
  const rangeFor = (metricKey: string, i: number) => {
    const v = decision?.hours[i]?.values[METRIC_TARGET[metricKey]];
    return v ? [v.low, v.high] : null;
  };

  const source = forecast ?? fetched;
  const metrics = useMemo(
    () => (source && source.length > 0 ? metricsFromForecast(source) : FALLBACK_METRICS),
    [source],
  );
  const hours = metrics[0].timestamps.length > 0
    ? metrics[0].timestamps.map(fmtHour)
    : buildForecastHours(metrics[0].timestamps.length);
  const forecastWindow = source && source.length > 0 ? source[0].timestamp : null;
  const forecastWindowLabel = forecastWindow ? fmtHour(forecastWindow) : '—';

  return (
    <div
      className="-mx-4 -my-6 rounded-xl p-6 md:p-8"
      style={{
        fontFamily: "'Inter', sans-serif",
        color: '#edf3ee',
        background: 'radial-gradient(120% 100% at 50% -10%, #10201a 0%, #080f0c 55%)',
        border: '1px solid #1e2b24',
      }}
    >
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="flex justify-between items-end flex-wrap gap-6 mb-2">
        <div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, letterSpacing: '0.18em', color: '#6f8079', textTransform: 'uppercase', marginBottom: 10 }}>
            Urban operations · predictive layer
          </div>
          <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: 38, letterSpacing: '-0.01em', margin: 0 }}>
            City Pulse forecast
          </h1>
          <div style={{ fontSize: 14, color: '#6f8079', marginTop: 8, maxWidth: 460, lineHeight: 1.5 }}>
            Six-hour outlook across traffic, air quality, grid load, and weather. Each value carries a 90% range measured from past forecast errors at that hour.
          </div>
        </div>
        <div className="text-right">
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, letterSpacing: '0.16em', color: '#6f8079', textTransform: 'uppercase' }}>
            {forecastWindow ? 'Forecast from' : 'Reference'}
          </div>
          <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 32, fontWeight: 500, marginTop: 2 }}>
            {forecastWindowLabel}
          </div>
        </div>
      </header>

      {/* ── Status pill ────────────────────────────────────────────────── */}
      <div className="flex items-center gap-2 mt-4">
        <span
          className="rounded-[20px] px-3 py-1 text-[10px] font-mono uppercase tracking-widest"
          style={{
            background: status === 'model' ? 'rgba(70,201,176,0.12)' : 'rgba(255,138,61,0.12)',
            color: status === 'model' ? '#8fe6c9' : status === 'loading' ? '#6f8079' : '#ffb08a',
            border: '1px solid #1e2b24',
          }}
        >
          {status === 'model' ? '● Model forecast live' : status === 'loading' ? '◌ Loading model…' : '○ Fallback data'}
        </span>
      </div>

      {/* ── Horizon rail ───────────────────────────────────────────────── */}
      <div className="relative overflow-hidden mt-9 rounded-[14px] px-7 py-[22px]"
        style={{ background: '#0e1712', border: '1px solid #1e2b24' }}>
        <div className="relative h-11">
          <div className="absolute top-5 left-0 right-0 h-[2px]" style={{ background: '#1e2b24' }} />
          <div className="absolute top-5 left-0 h-[2px]" style={{ width: '16.6%', background: 'linear-gradient(90deg,#3a4a41,#46c9b0)' }} />
          {hours.map((h, i) => {
            const isFirst = i === 0;
            return (
              <div key={h} className="absolute top-0 w-20 text-center"
                style={{ left: `${(i / (hours.length - 1)) * 100}%`, transform: 'translateX(-50%)' }}>
                <div className="absolute -top-5 left-1/2 whitespace-nowrap -translate-x-1/2"
                  style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: isFirst ? '#edf3ee' : '#6f8079', fontWeight: isFirst ? 500 : 400 }}>
                  {h}
                </div>
                <div className="mx-auto mt-3 rounded-full relative z-[2]"
                  style={{
                    width: isFirst ? 14 : 8,
                    height: isFirst ? 14 : 8,
                    background: isFirst ? '#46c9b0' : '#465049',
                    marginTop: isFirst ? 9 : 12,
                    boxShadow: isFirst ? '0 0 0 5px rgba(70,201,176,0.18), 0 0 18px rgba(70,201,176,0.55)' : undefined,
                    animation: isFirst ? 'pulse 2.4s ease-in-out infinite' : undefined,
                  }} />
                <div className="mt-[26px] text-[9px] tracking-[0.06em] whitespace-nowrap"
                  style={{ color: hourRisk[i]?.fires ? '#ffb08a' : '#465049' }}
                  title="Highest event probability at this hour">
                  {hourRisk[i] ? (hourRisk[i].fires ? `alert · ${Math.round(hourRisk[i].p * 100)}%` : `risk ${Math.round(hourRisk[i].p * 100)}%`) : ''}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Metric grid ────────────────────────────────────────────────── */}
      <div className="grid mt-7 gap-5" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
        {metrics.map((m) => {
          const delta = deltaTag(m.values[m.values.length - 1], m.values[0]);
          return (
            <div key={m.key} className="flex flex-col rounded-[14px] p-5"
              style={{ background: '#0e1712', border: '1px solid #1e2b24' }}>
              <div className="flex items-center gap-2.5 mb-1">
                <span className="w-[9px] h-[9px] rounded-full" style={{ background: m.color }} />
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, letterSpacing: '0.14em', color: '#6f8079', textTransform: 'uppercase' }}>
                  {m.label}
                </span>
              </div>

              <div className="flex items-baseline gap-2 my-2.5">
                <span style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 30, fontWeight: 500 }}>{fmt(m.values[0])}</span>
                <span style={{ fontSize: 12, color: '#6f8079' }}>{m.unit}</span>
              </div>

              <span className="mb-3.5 rounded-[20px] px-[7px] py-[2px] inline-flex items-center gap-1 w-fit"
                style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, ...Object.fromEntries(DELTA_CLASS[delta.cls].split(';').map((s) => s.split(':'))) }}>
                {delta.text}
              </span>

              <Spark values={m.values} color={m.color} />

              <div className="flex flex-col">
                {m.values.map((v, i) => (
                  <div key={i} className="grid items-center gap-2.5 py-2.5"
                    style={{
                      gridTemplateColumns: '52px 1fr auto',
                      borderBottom: i === m.values.length - 1 ? 'none' : '1px dashed #1e2b24',
                    }}>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: '#6f8079' }}>{hours[i]}</span>
                    <span style={{ fontSize: 10, color: '#465049' }}>{HORIZONS[i]}</span>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 15, fontWeight: 500, textAlign: 'right' }}>
                      {fmt(v)}<span style={{ color: '#6f8079', fontWeight: 400, fontSize: 11 }}> {m.unit}</span>
                      {rangeFor(m.key, i) && (
                        <span style={{ display: 'block', color: '#465049', fontWeight: 400, fontSize: 10 }}>
                          {fmt(Math.round(rangeFor(m.key, i)![0]))}–{fmt(Math.round(rangeFor(m.key, i)![1]))}
                        </span>
                      )}
                    </span>
                    <div className="col-span-3 h-[3px] rounded mt-2 overflow-hidden" style={{ gridColumn: '1/-1', background: '#1e2b24' }}>
                      <div className="h-full rounded" style={{ width: `${((v / Math.max(...m.values)) * 100).toFixed(1)}%`, background: m.color }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Footer legend ──────────────────────────────────────────────── */}
      <footer className="flex items-center gap-2 mt-7" style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: '#465049' }}>
        <span>Small numbers under each value: 90% range, backtested on held-out 2025 data at the same hour of day.</span>
        <span className="ml-4">Rail: highest event probability per hour; "alert" means it crosses the validated cutoff.</span>
      </footer>

      <style>{`@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.55;}}`}</style>
    </div>
  );
}
