import { motion } from 'motion/react';
import { CityData, ForecastPoint } from '../services/dataService';

// ─── Types ────────────────────────────────────────────────────────────────────
type Trend = 'up' | 'down' | 'stable';

interface Signal {
  key: keyof CityData;
  label: string;
  color: string;           // accent hex
  barBg: string;           // tailwind / inline style for filled bar
  leftLabel: string;
  rightLabel: string;
  reason: (trend: Trend) => string;
  trend: Trend;
}

// ─── Bar helper ───────────────────────────────────────────────────────────────
/**
 * Returns 0–1 fill fraction for the gradient bar.
 * "up"   → bar fills to the right  (75 %)
 * "down" → bar fills to the left   (25 %)
 * "stable" → bar sits in the middle (50 %)
 */
function barFill(trend: Trend): number {
  if (trend === 'up')   return 0.75;
  if (trend === 'down') return 0.25;
  return 0.5;
}

function trendArrow(trend: Trend) {
  if (trend === 'up')   return '↑';
  if (trend === 'down') return '↓';
  return '→';
}

function trendLabel(trend: Trend) {
  if (trend === 'up')   return 'Trending up';
  if (trend === 'down') return 'Trending down';
  return 'Stable';
}

// ─── Trend derivation from model forecast ────────────────────────────────────
function trendFromSeries(values: number[] | null | undefined): Trend {
  if (!values || values.length < 2) return 'stable';
  const first = values[0];
  const last = values[values.length - 1];
  const diff = last - first;
  const pct = Math.abs(first) > 0 ? diff / first : diff;
  if (pct > 0.02) return 'up';
  if (pct < -0.02) return 'down';
  return 'stable';
}

function seriesFromForecast(forecast: ForecastPoint[], pick: (p: ForecastPoint) => number | null): number[] {
  return forecast.map((p) => pick(p)).filter((v): v is number => v !== null);
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function XaiPanel({ data, forecast }: { data: CityData; forecast?: ForecastPoint[] | null }) {
  // Prefer the model's predicted series for trend direction when available
  const predicted = forecast && forecast.length > 0 ? forecast : null;

  const trafficTrend: Trend  = predicted
    ? trendFromSeries(seriesFromForecast(predicted, (p) => p.trafficFlow))
    : data.traffic.trend === 'down' ? 'down' : data.traffic.trend === 'up' ? 'up' : 'stable';
  const airTrend: Trend      = predicted
    ? trendFromSeries(seriesFromForecast(predicted, (p) => p.aqi))
    : data.air.trend     === 'down' ? 'down' : data.air.trend     === 'up' ? 'up' : 'stable';
  const weatherTrend: Trend  = predicted
    ? trendFromSeries(seriesFromForecast(predicted, (p) => p.temperature))
    : data.weather.trend === 'up'   ? 'up'   : data.weather.trend === 'down' ? 'down' : 'stable';
  const energyTrend: Trend   = predicted
    ? trendFromSeries(seriesFromForecast(predicted, (p) => p.electricityDemand))
    : weatherTrend === 'up' || trafficTrend === 'up' ? 'up'
    : weatherTrend === 'down' && trafficTrend === 'down' ? 'down'
    : 'stable';

  const signals: Signal[] = [
    {
      key: 'traffic',
      label: 'Traffic',
      color: '#f0a857',
      barBg: 'linear-gradient(90deg, #f0a857 0%, rgba(240,168,87,0.15) 100%)',
      leftLabel: 'Less traffic',
      rightLabel: 'More traffic',
      trend: trafficTrend,
      reason: (t) =>
        t === 'down'
          ? 'Volume may decrease, because recent readings have been cooling down across major junctions.'
          : t === 'up'
          ? 'Volume is rising — congestion building at key intersections during peak hours.'
          : 'Traffic is holding steady near the current baseline.',
    },
    {
      key: 'air',
      label: 'Air quality',
      color: '#3498db',
      barBg: 'linear-gradient(90deg, #f2667a 0%, #6fe7b7 100%)',
      leftLabel: 'Poor air',
      rightLabel: 'Good air',
      trend: airTrend,
      reason: (t) =>
        t === 'down'
          ? 'AQI is expected to ease, clearing up slightly — this lines up with easing traffic, which means fewer road emissions.'
          : t === 'up'
          ? 'AQI is rising. Increased traffic or wind-pattern shift may be concentrating particulates.'
          : 'Air quality is holding at current levels.',
    },
    {
      key: 'weather',
      label: 'Temperature',
      color: '#9b59b6',
      barBg: 'linear-gradient(90deg, #3498db 0%, #f2667a 100%)',
      leftLabel: 'Cooler',
      rightLabel: 'Warmer',
      trend: weatherTrend,
      reason: (t) =>
        t === 'up'
          ? 'Readings are trending upward, likely from ordinary diurnal heating or the urban heat island effect.'
          : t === 'down'
          ? 'Temperature is dipping — cloud cover or an evening cool-down may be contributing.'
          : 'Temperature is stable near the current reading.',
    },
    {
      key: 'energy',
      label: 'Energy demand',
      color: '#2ecc71',
      barBg: 'linear-gradient(90deg, rgba(46,204,113,0.15) 0%, #2ecc71 100%)',
      leftLabel: 'Less demand',
      rightLabel: 'More demand',
      trend: energyTrend,
      reason: (t) =>
        t === 'up'
          ? 'Electricity demand is rising, potentially driven by increased cooling needs as temperatures climb.'
          : t === 'down'
          ? 'Grid load is easing — lower temperatures or reduced activity are cutting cooling and industrial draw.'
          : 'Demand is steady relative to the current baseline.',
    },
  ];

  // Causal chain rows (only render links that are non-trivial)
  const trafficToAir = (() => {
    const rising = trafficTrend === 'up';
    const easing = airTrend === 'down';
    if (rising && easing) {
      return 'Even with road volume climbing, the near-term AQI call still eases — wind dispersion and secondary factors outweigh the added tailpipe emissions in the model window.';
    }
    if (rising && airTrend === 'up') {
      return 'Rising road volume concentrates emissions at junctions, and the model carries that directly into the AQI forecast.';
    }
    if (!rising && easing) {
      return 'Easing traffic pressure means fewer road emissions, which is factored into the near-term AQI call. Secondary correlations with temperature are also weighed in.';
    }
    return 'The model weighs how road emissions interact with wind and temperature before committing to the AQI call.';
  })();

  const tempToEnergy = (() => {
    const cooling = weatherTrend === 'up';
    const easing = energyTrend === 'down';
    if (cooling && energyTrend === 'up') {
      return 'Rising temperature shifts the baseline for electricity demand as cooling needs adjust upward.';
    }
    if (!cooling && easing) {
      return 'Dropping temperature trims cooling load, letting the model ease grid demand across the window.';
    }
    if (cooling && easing) {
      return 'Temperature rises, but the model still eases grid demand — evening hours and reduced activity pull more weight than the cooling signal.';
    }
    return 'The model ties grid load to temperature swings, with cooling and industrial draw moving together.';
  })();

  const causalLinks = [
    {
      from: `Traffic ${trendLabel(trafficTrend).toLowerCase()}`,
      to:   `AQI ${trendLabel(airTrend).toLowerCase()}`,
      explanation: trafficToAir,
    },
    {
      from: `Temperature ${trendLabel(weatherTrend).toLowerCase()}`,
      to:   `Energy ${trendLabel(energyTrend).toLowerCase()}`,
      explanation: tempToEnergy,
    },
  ];

  const summaryLines = [
    { icon: trendArrow(trafficTrend),  text: `Traffic flow is ${trendLabel(trafficTrend).toLowerCase()} from current levels.` },
    { icon: trendArrow(airTrend),      text: `AQI is ${trendLabel(airTrend).toLowerCase()} into the next hour.` },
    { icon: trendArrow(weatherTrend),  text: `Temperature is ${trendLabel(weatherTrend).toLowerCase()} relative to the current value.` },
    { icon: trendArrow(energyTrend),   text: `Electricity demand is ${trendLabel(energyTrend).toLowerCase()}.` },
  ];

  return (
    <div style={{ fontFamily: "'Inter', sans-serif", color: '#e9f3ee' }}>

      {/* ── Header ────────────────────────────────────────────────────── */}
      <div className="mb-8">
        <h1
          className="font-['Fraunces',serif] font-light italic leading-[1.15] mb-2"
          style={{ fontSize: 'clamp(1.5rem,3.5vw,2.1rem)' }}
        >
          Explainable AI (XAI)
        </h1>
        <p className="text-[0.9rem] font-semibold mb-1" style={{ color: '#e9f3ee' }}>
          Reasoning engine
        </p>
        <p className="text-[0.875rem] max-w-[58ch] leading-relaxed" style={{ color: '#8fa69b' }}>
          Why the model thinks this — in plain language, with the signals connected instead of listed separately.
        </p>
      </div>

      {/* ── Signal cards ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
        {signals.map((sig, i) => {
          const fill = barFill(sig.trend);
          return (
            <motion.div
              key={sig.key}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.07, duration: 0.35 }}
              className="rounded-[16px] border p-5"
              style={{ background: '#10201a', borderColor: '#1f3831' }}
            >
              {/* Top row */}
              <div className="flex items-start justify-between mb-1">
                <div className="flex items-center gap-2">
                  <div
                    className="w-[7px] h-[7px] rounded-full shrink-0 mt-[3px]"
                    style={{ background: sig.color, boxShadow: `0 0 8px ${sig.color}66` }}
                  />
                  <span className="text-[0.78rem] font-semibold" style={{ color: '#e9f3ee' }}>
                    {sig.label}
                  </span>
                </div>
                <span
                  className="text-[0.68rem] font-mono px-2.5 py-0.5 rounded-full border"
                  style={{
                    background: `${sig.color}18`,
                    borderColor: `${sig.color}40`,
                    color: sig.color,
                  }}
                >
                  {trendArrow(sig.trend)} {trendLabel(sig.trend)}
                </span>
              </div>

              {/* Reason text */}
              <p className="text-[0.82rem] leading-relaxed mt-3 mb-4" style={{ color: '#8fa69b' }}>
                {sig.reason(sig.trend)}
              </p>

              {/* Gradient bar */}
              <div className="space-y-1.5">
                <div
                  className="relative h-[5px] w-full rounded-full overflow-hidden"
                  style={{ background: '#142720' }}
                >
                  {/* filled portion */}
                  <div
                    className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
                    style={{ width: `${fill * 100}%`, background: sig.barBg }}
                  />
                  {/* thumb */}
                  <div
                    className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full border-2 transition-all duration-700"
                    style={{
                      left: `calc(${fill * 100}% - 5px)`,
                      background: sig.color,
                      borderColor: '#0a1210',
                      boxShadow: `0 0 8px ${sig.color}88`,
                    }}
                  />
                </div>
                <div className="flex justify-between">
                  <span className="text-[0.6rem] font-mono" style={{ color: '#5c7269' }}>{sig.leftLabel}</span>
                  <span className="text-[0.6rem] font-mono" style={{ color: '#5c7269' }}>{sig.rightLabel}</span>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* ── Interconnected output logic ──────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35, duration: 0.4 }}
        className="rounded-[16px] border p-6 mb-5"
        style={{ background: '#10201a', borderColor: '#1f3831' }}
      >
        <div className="mb-5">
          <div
            className="text-[0.65rem] font-mono uppercase tracking-[0.16em] mb-1"
            style={{ color: '#5c7269' }}
          >
            Interconnected output logic
          </div>
          <p className="text-[0.82rem]" style={{ color: '#8fa69b' }}>
            These four signals aren't independent — here's how the model links them.
          </p>
        </div>

        <div className="space-y-4 mb-6">
          {causalLinks.map(({ from, to, explanation }) => (
            <div
              key={from + to}
              className="flex gap-4 items-start p-4 rounded-[12px] border"
              style={{ background: '#0e1a16', borderColor: '#1f3831' }}
            >
              <div className="flex items-center gap-2 shrink-0 pt-0.5">
                <span className="text-[0.72rem] font-mono font-semibold" style={{ color: '#e9f3ee' }}>{from}</span>
                <span style={{ color: '#3f8a71', fontSize: '1rem' }}>→</span>
                <span className="text-[0.72rem] font-mono font-semibold" style={{ color: '#6fe7b7' }}>{to}</span>
              </div>
              <p className="text-[0.78rem] leading-relaxed" style={{ color: '#8fa69b' }}>
                {explanation}
              </p>
            </div>
          ))}
        </div>

        {/* Network momentum strip */}
        <div
          className="flex flex-wrap items-center gap-2 px-4 py-3 rounded-[10px] border border-dashed"
          style={{ background: '#0a1210', borderColor: '#1f3831' }}
        >
          <span className="text-[0.65rem] font-mono uppercase tracking-widest" style={{ color: '#5c7269' }}>
            Overall network momentum:
          </span>
          {[
            { label: `Traffic ${trendArrow(trafficTrend)}`, color: '#f0a857' },
            { sep: true },
            { label: `AQI ${trendArrow(airTrend)}`, color: '#3498db' },
            { sep: true },
            { label: `Temp ${trendArrow(weatherTrend)}`, color: '#9b59b6' },
            { sep: false, arrow: true },
            { label: `Energy ${trendArrow(energyTrend)}`, color: '#2ecc71' },
          ].map((item, i) =>
            'sep' in item && item.sep ? (
              <span key={i} style={{ color: '#3f8a71', fontSize: '0.7rem' }}>|</span>
            ) : 'arrow' in item ? (
              <span key={i} style={{ color: '#3f8a71', fontSize: '0.8rem' }}>→</span>
            ) : (
              <span
                key={i}
                className="text-[0.7rem] font-mono font-semibold"
                style={{ color: (item as { label: string; color: string }).color }}
              >
                {(item as { label: string }).label}
              </span>
            )
          )}
        </div>
      </motion.div>

      {/* ── Simple summary ───────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.45, duration: 0.4 }}
        className="rounded-[16px] border p-6"
        style={{ background: '#10201a', borderColor: '#1f3831' }}
      >
        <div
          className="text-[0.65rem] font-mono uppercase tracking-[0.16em] mb-4"
          style={{ color: '#5c7269' }}
        >
          Simple summary
        </div>
        <ul className="space-y-3">
          {summaryLines.map(({ icon, text }, i) => (
            <li key={i} className="flex items-baseline gap-3">
              <span
                className="text-[1rem] shrink-0 font-bold"
                style={{
                  color:
                    i === 0 ? '#f0a857'
                    : i === 1 ? '#3498db'
                    : i === 2 ? '#9b59b6'
                    : '#2ecc71',
                }}
              >
                {icon}
              </span>
              <span className="text-[0.85rem] leading-snug" style={{ color: '#8fa69b' }}>
                {text}
              </span>
            </li>
          ))}
        </ul>
      </motion.div>
    </div>
  );
}
