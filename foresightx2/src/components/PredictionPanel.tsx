import { useEffect, useMemo, useState } from 'react';
import { motion } from 'motion/react';
import { backendApi, CityData, ModelConditions } from '../services/dataService';

// ─── Types ────────────────────────────────────────────────────────────────────
type MetricKey = 'traffic' | 'air' | 'energy' | 'weather';

interface MetricSpec {
  key: MetricKey;
  label: string;
  unit: string;
  color: string;           // accent hex
  colorClass: string;      // tailwind bg class for dot
}

const METRICS: MetricSpec[] = [
  { key: 'traffic', label: 'Traffic',      unit: 'veh/hr', color: '#f39c12', colorClass: 'bg-traf-acc'  },
  { key: 'air',     label: 'Air quality',  unit: 'AQI',    color: '#3498db', colorClass: 'bg-air-acc'   },
  { key: 'energy',  label: 'Grid load',    unit: 'MW',     color: '#2ecc71', colorClass: 'bg-eng-acc'   },
  { key: 'weather', label: 'Weather',      unit: '°C',     color: '#9b59b6', colorClass: 'bg-wth-acc'   },
];

const HOURS = [1, 2, 3, 4, 5, 6];

// opacity steps: full → very faint, mirroring the HTML "fade & dash" confidence cue
const ROW_OPACITY = [1, 0.88, 0.72, 0.56, 0.4, 0.26];

// ─── Helpers ─────────────────────────────────────────────────────────────────
function valueFromCondition(
  condition: ModelConditions,
  key: MetricKey,
  fallback: CityData,
): number {
  if (key === 'traffic') return Math.round(Number(condition.traffic?.flow ?? fallback.traffic.value));
  if (key === 'air')     return Math.round(Number(condition.aqi?.aqi    ?? fallback.air.value));
  if (key === 'energy')  return Math.round(Number(condition.raw?.electricity_demand ?? fallback.energy.value));
  return Math.round(Number(condition.weather?.temperature?.value ?? fallback.weather.value));
}

function formatVal(val: number, key: MetricKey): string {
  if (key === 'traffic') return val.toLocaleString();
  return String(val);
}

// Format current time as "2:00 PM"
function fmtTime(d: Date): string {
  let h = d.getHours() % 12;
  if (h === 0) h = 12;
  const m = String(d.getMinutes()).padStart(2, '0');
  const ampm = d.getHours() >= 12 ? 'PM' : 'AM';
  return `${h}:${m} ${ampm}`;
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function PredictionPanel({
  data,
  city = 'bangalore',
}: {
  data: CityData;
  city?: string;
}) {
  const [forecasts, setForecasts] = useState<Record<number, ModelConditions>>({});
  const [status, setStatus] = useState<'loading' | 'model' | 'fallback'>('loading');
  const [now, setNow] = useState(new Date());

  // tick clock every minute
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 60_000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    let mounted = true;
    Promise.all(HOURS.map((h) => backendApi.modelConditions(city, h)))
      .then((responses) => {
        if (!mounted) return;
        const map: Record<number, ModelConditions> = {};
        responses.forEach((r, i) => { map[HOURS[i]] = r.data; });
        setForecasts(map);
        setStatus('model');
      })
      .catch(() => { if (mounted) setStatus('fallback'); });
    return () => { mounted = false; };
  }, [city]);

  const fallback = useMemo(() => ({
    traffic: data.traffic.value,
    air:     data.air.value,
    energy:  data.energy.value,
    weather: data.weather.value,
  }), [data]);

  const getVal = (key: MetricKey, hour: number): number => {
    const fc = forecasts[hour];
    return fc ? valueFromCondition(fc, key, data) : fallback[key];
  };

  return (
    <div
      className="min-h-0 text-[#e9f3ee]"
      style={{ fontFamily: "'Inter', sans-serif" }}
    >
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="mb-8">
        <div
          className="text-[0.65rem] font-mono uppercase tracking-[0.18em] mb-2"
          style={{ color: '#5c7269' }}
        >
          Urban operations · predictive layer
        </div>
        <h1
          className="font-['Fraunces',serif] font-light italic text-[clamp(1.5rem,3.5vw,2.1rem)] leading-[1.15] mb-3"
          style={{ color: '#e9f3ee' }}
        >
          City Pulse forecast
        </h1>
        <p className="text-[0.875rem] max-w-[60ch] leading-relaxed" style={{ color: '#8fa69b' }}>
          Six-hour outlook across traffic, air quality, grid load, and weather.
          Rows fade and dash further from now — that's the model telling you how sure it is.
        </p>
      </div>

      {/* ── Current time pill ───────────────────────────────────────────── */}
      <div className="flex items-center gap-4 mb-8">
        <div
          className="inline-flex items-center gap-3 px-4 py-2.5 rounded-[20px] border text-[0.75rem] font-mono"
          style={{ background: '#0e1a16', borderColor: '#1f3831', color: '#8fa69b' }}
        >
          <div
            className="w-[7px] h-[7px] rounded-full animate-pulse"
            style={{ background: '#6fe7b7', boxShadow: '0 0 8px #6fe7b7' }}
          />
          <span style={{ color: '#5c7269' }}>Current time</span>
          <span className="font-semibold" style={{ color: '#e9f3ee' }}>{fmtTime(now)}</span>
        </div>

        {/* backend status */}
        <div
          className="text-[0.65rem] font-mono uppercase tracking-widest px-3 py-1.5 rounded-full border"
          style={{
            background: status === 'model' ? 'rgba(111,231,183,0.08)' : 'rgba(242,102,122,0.08)',
            borderColor: status === 'model' ? 'rgba(111,231,183,0.25)' : 'rgba(242,102,122,0.25)',
            color: status === 'model' ? '#6fe7b7' : status === 'loading' ? '#8fa69b' : '#f2667a',
          }}
        >
          {status === 'model' ? '● Model live' : status === 'loading' ? '◌ Loading…' : '○ Fallback'}
        </div>
      </div>

      {/* ── Forecast table ──────────────────────────────────────────────── */}
      <div
        className="rounded-[18px] border overflow-hidden"
        style={{ background: '#10201a', borderColor: '#1f3831' }}
      >
        {/* Column headers */}
        <div
          className="grid border-b"
          style={{
            gridTemplateColumns: '72px repeat(4, 1fr)',
            borderColor: '#1f3831',
            background: '#0d1a10',
          }}
        >
          <div className="px-4 py-3" />
          {METRICS.map((m) => (
            <div key={m.key} className="px-4 py-3 flex items-center gap-2">
              <div
                className="w-[7px] h-[7px] rounded-full shrink-0"
                style={{ background: m.color, boxShadow: `0 0 8px ${m.color}55` }}
              />
              <span
                className="text-[0.65rem] font-mono uppercase tracking-[0.12em]"
                style={{ color: '#8fa69b' }}
              >
                {m.label}
              </span>
            </div>
          ))}
        </div>

        {/* Forecast rows */}
        {HOURS.map((h, idx) => {
          const opacity = ROW_OPACITY[idx];
          const isFirst = idx === 0;
          const isDashed = idx >= 3;           // last 3 rows dashed (lower confidence)

          return (
            <motion.div
              key={h}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.06, duration: 0.35 }}
              className="grid border-b last:border-b-0"
              style={{
                gridTemplateColumns: '72px repeat(4, 1fr)',
                borderColor: isDashed ? 'transparent' : '#1a2e26',
                // dashed border via box shadow trick
                boxShadow: isDashed && idx !== 0
                  ? 'inset 0 1px 0 0 rgba(31,56,49,0.5)'
                  : undefined,
                opacity,
              }}
            >
              {/* Hour label */}
              <div
                className="px-4 py-4 flex items-center"
                style={{ borderRight: '1px solid #1f3831' }}
              >
                <span
                  className="text-[0.65rem] font-mono uppercase tracking-[0.1em]"
                  style={{
                    color: isFirst ? '#6fe7b7' : '#5c7269',
                    fontWeight: isFirst ? 600 : 400,
                  }}
                >
                  +{h}H
                </span>
              </div>

              {/* Metric cells */}
              {METRICS.map((m) => {
                const val = getVal(m.key, h);
                return (
                  <div
                    key={m.key}
                    className="px-4 py-4 flex flex-col gap-0.5 transition-colors"
                    style={{ borderRight: '1px solid #1a2e26' }}
                  >
                    <span
                      className="font-['Fraunces',serif] font-medium leading-none"
                      style={{
                        fontSize: 'clamp(1.1rem, 2.2vw, 1.45rem)',
                        color: isFirst ? m.color : '#e9f3ee',
                      }}
                    >
                      {status === 'loading' ? '—' : formatVal(val, m.key)}
                    </span>
                    <span
                      className="text-[0.6rem] font-mono uppercase tracking-widest"
                      style={{ color: '#5c7269' }}
                    >
                      {m.unit}
                    </span>
                  </div>
                );
              })}
            </motion.div>
          );
        })}
      </div>

      {/* ── Legend ──────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-6 mt-5 px-1">
        {[
          { label: 'near-term, high confidence', opacity: 1, dashed: false },
          { label: 'farther out, lower confidence', opacity: 0.35, dashed: true },
        ].map(({ label, opacity, dashed }) => (
          <div key={label} className="flex items-center gap-2.5">
            <div
              className="w-10 h-[2px] rounded-full"
              style={{
                background: '#6fe7b7',
                opacity,
                borderBottom: dashed ? '1px dashed #6fe7b7' : undefined,
                backgroundImage: dashed
                  ? 'repeating-linear-gradient(90deg,#6fe7b7 0,#6fe7b7 4px,transparent 4px,transparent 8px)'
                  : undefined,
                backgroundSize: dashed ? '8px 2px' : undefined,
                height: 2,
              }}
            />
            <span className="text-[0.65rem] font-mono" style={{ color: '#5c7269' }}>
              {label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
