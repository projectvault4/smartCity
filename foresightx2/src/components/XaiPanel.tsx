import { motion } from 'motion/react';
import { useCallback, useState } from 'react';
import { CityData, ForecastPoint } from '../services/dataService';

const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';
const GROQ_MODEL = 'llama-3.3-70b-versatile';

// ─── Types ────────────────────────────────────────────────────────────────────
type Trend = 'up' | 'down' | 'stable';

interface Signal {
  key: keyof CityData;
  label: string;
  color: string;           // accent hex
  barBg: string;           // tailwind / inline style for filled bar
  leftLabel: string;
  rightLabel: string;
  trend: Trend;
  value: number | null;    // latest model value
  nextValue: number | null; // forecast value 6h ahead
  unit: string;
  reason: (trend: Trend, value: number | null, nextValue: number | null) => string;
}

// ─── Bar helper ───────────────────────────────────────────────────────────────
/**
 * Returns 0–1 fill fraction for the gradient bar.
 * Position is driven by the real first→last change in the model series,
 * clamped to a visible band so the animation reads clearly.
 */
function barFill(trend: Trend, changePct?: number): number {
  if (trend === 'stable') return 0.5;
  if (trend === 'up') {
    const mag = Math.min(0.85, 0.58 + Math.abs(changePct ?? 0) * 0.5);
    return mag;
  }
  const mag = Math.max(0.15, 0.42 - Math.abs(changePct ?? 0) * 0.5);
  return mag;
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

function fmt(v: number | null): string {
  if (v === null || !Number.isFinite(v)) return '—';
  return String(Math.round(v).toLocaleString());
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

  const trafficSeries = predicted ? seriesFromForecast(predicted, (p) => p.trafficFlow) : [];
  const airSeries     = predicted ? seriesFromForecast(predicted, (p) => p.aqi) : [];
  const weatherSeries = predicted ? seriesFromForecast(predicted, (p) => p.temperature) : [];
  const energySeries  = predicted ? seriesFromForecast(predicted, (p) => p.electricityDemand) : [];

  const trafficTrend: Trend  = trafficSeries.length ? trendFromSeries(trafficSeries) : data.traffic.trend === 'down' ? 'down' : data.traffic.trend === 'up' ? 'up' : 'stable';
  const airTrend: Trend      = airSeries.length ? trendFromSeries(airSeries) : data.air.trend === 'down' ? 'down' : data.air.trend === 'up' ? 'up' : 'stable';
  const weatherTrend: Trend  = weatherSeries.length ? trendFromSeries(weatherSeries) : data.weather.trend === 'up' ? 'up' : data.weather.trend === 'down' ? 'down' : 'stable';
  const energyTrend: Trend   = energySeries.length ? trendFromSeries(energySeries) : weatherTrend === 'up' || trafficTrend === 'up' ? 'up' : weatherTrend === 'down' && trafficTrend === 'down' ? 'down' : 'stable';

  const changePct = (series: number[]) =>
    series.length >= 2 ? (series[series.length - 1] - series[0]) / (Math.abs(series[0]) || 1) : 0;

  const signals: Signal[] = [
    {
      key: 'traffic',
      label: 'Traffic',
      color: '#f0a857',
      barBg: 'linear-gradient(90deg, #f0a857 0%, rgba(240,168,87,0.15) 100%)',
      leftLabel: 'Less traffic',
      rightLabel: 'More traffic',
      trend: trafficTrend,
      value: trafficSeries.length ? trafficSeries[trafficSeries.length - 1] : null,
      nextValue: trafficSeries.length ? trafficSeries[0] : null,
      unit: 'vehicles/hr',
      reason: (t, v, nv) =>
        t === 'down'
          ? `Traffic is easing. The model expects about ${fmt(v)} vehicles per hour by the end of the window — down from ${fmt(nv)} now.`
          : t === 'up'
          ? `Traffic is going up. The model expects about ${fmt(v)} vehicles per hour by the end of the window — up from ${fmt(nv)} now.`
          : 'Traffic stays about the same across the forecast window.',
    },
    {
      key: 'air',
      label: 'Air quality',
      color: '#3498db',
      barBg: 'linear-gradient(90deg, #f2667a 0%, #6fe7b7 100%)',
      leftLabel: 'Poor air',
      rightLabel: 'Good air',
      trend: airTrend,
      value: airSeries.length ? airSeries[airSeries.length - 1] : null,
      nextValue: airSeries.length ? airSeries[0] : null,
      unit: 'AQI',
      reason: (t, v, nv) =>
        t === 'down'
          ? `Air quality is getting better. The AQI should ease to ${fmt(v)} from ${fmt(nv)} — the model links this to lighter traffic and fewer emissions.`
          : t === 'up'
          ? `Air quality is getting worse. The AQI rises to ${fmt(v)} from ${fmt(nv)} — more traffic means more pollution in the air.`
          : 'Air quality stays near its current level across the forecast window.',
    },
    {
      key: 'weather',
      label: 'Temperature',
      color: '#9b59b6',
      barBg: 'linear-gradient(90deg, #3498db 0%, #f2667a 100%)',
      leftLabel: 'Cooler',
      rightLabel: 'Warmer',
      trend: weatherTrend,
      value: weatherSeries.length ? weatherSeries[weatherSeries.length - 1] : null,
      nextValue: weatherSeries.length ? weatherSeries[0] : null,
      unit: '°C',
      reason: (t, v, nv) =>
        t === 'up'
          ? `It's getting slightly warmer. Temperature moves up to ${fmt(v)}°C from ${fmt(nv)}°C — a normal daytime warming pattern.`
          : t === 'down'
          ? `It's getting slightly cooler. Temperature dips to ${fmt(v)}°C — cloud cover or an evening cool-down is likely.`
          : 'Temperature stays about the same as the current reading.',
    },
    {
      key: 'energy',
      label: 'Energy demand',
      color: '#2ecc71',
      barBg: 'linear-gradient(90deg, rgba(46,204,113,0.15) 0%, #2ecc71 100%)',
      leftLabel: 'Less demand',
      rightLabel: 'More demand',
      trend: energyTrend,
      value: energySeries.length ? energySeries[energySeries.length - 1] : null,
      nextValue: energySeries.length ? energySeries[0] : null,
      unit: 'MW',
      reason: (t, v, nv) =>
        t === 'up'
          ? `Electricity use is going up. The model expects about ${fmt(v)} MW by the end of the window — up from ${fmt(nv)} MW — as warmer weather means more cooling.`
          : t === 'down'
          ? `Electricity use is easing to about ${fmt(v)} MW — cooler weather or lower activity means less cooling is needed.`
          : 'Electricity use stays about the same across the forecast window.',
    },
  ];

  // Causal chain rows (only render links that are non-trivial)
  const trafficToAir = (() => {
    const rising = trafficTrend === 'up';
    const easing = airTrend === 'down';
    if (rising && easing) {
      return 'Even though traffic is rising, the model still expects air quality to improve. In this forecast, wind and other weather factors outweigh the extra vehicle pollution for the next few hours.';
    }
    if (rising && airTrend === 'up') {
      return 'More cars on the road means more exhaust and pollution. So when the model sees traffic rising, it raises the air-quality reading too.';
    }
    if (!rising && easing) {
      return 'Less traffic means fewer exhaust fumes. The model carries that directly into the air-quality forecast, so AQI comes down as traffic eases.';
    }
    return 'The model checks how traffic pollution mixes with wind and temperature before deciding on air quality.';
  })();

  const tempToEnergy = (() => {
    const cooling = weatherTrend === 'up';
    const easing = energyTrend === 'down';
    if (cooling && energyTrend === 'up') {
      return 'Warmer weather means people use more fans and air conditioning. So the model raises electricity demand as temperature climbs.';
    }
    if (!cooling && easing) {
      return 'Cooler weather means less need for cooling. The model lowers electricity demand as the temperature drops.';
    }
    if (cooling && easing) {
      return 'Even though it is warming up, the model still sees electricity use falling — evening hours and lower activity have a bigger effect than the extra cooling.';
    }
    return 'The model links electricity use to temperature — hotter days push demand up, cooler days pull it down.';
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
    { icon: trendArrow(trafficTrend),  text: `Traffic is ${trendLabel(trafficTrend).toLowerCase()} — around ${fmt(trafficSeries.length ? trafficSeries[trafficSeries.length - 1] : null)} vehicles per hour by the end of the window.` },
    { icon: trendArrow(airTrend),      text: `Air quality is ${trendLabel(airTrend).toLowerCase()} — AQI near ${fmt(airSeries.length ? airSeries[airSeries.length - 1] : null)}.` },
    { icon: trendArrow(weatherTrend),  text: `Temperature is ${trendLabel(weatherTrend).toLowerCase()} — around ${fmt(weatherSeries.length ? weatherSeries[weatherSeries.length - 1] : null)}°C.` },
    { icon: trendArrow(energyTrend),   text: `Electricity demand is ${trendLabel(energyTrend).toLowerCase()} — about ${fmt(energySeries.length ? energySeries[energySeries.length - 1] : null)} MW.` },
  ];

  // ── Groq plain-language explanation ─────────────────────────────────────
  const [plain, setPlain] = useState<string>('');
  const [plainLoading, setPlainLoading] = useState(false);

  const generatePlain = useCallback(async () => {
    setPlainLoading(true);
    setPlain('');
    try {
      const points = predicted
        ? predicted.map((p) => ({
            h: `T+${p.stepAhead}H`,
            traffic: Math.round(p.trafficFlow),
            aqi: Math.round(p.aqi),
            temp: Math.round(p.temperature),
            energy: Math.round(p.electricityDemand),
          }))
        : [];
      const prompt = `
You are explaining a trained city forecast model to a regular citizen — no jargon, no AI terms.

Real model forecast values for the next 6 hours:
${JSON.stringify(points)}

Summarise in 4-5 short, simple sentences (max ~110 words):
1. What is happening overall (are traffic, air quality, temperature and electricity demand going up, down, or steady?).
2. Why it is happening in everyday words.
3. What it means for the person (one practical tip, e.g. about travel or breathing).
Mention the real numbers naturally. Never refuse or claim the data is insufficient.
`;
      const response = await fetch(GROQ_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${process.env.GROQ_API_KEY || ''}`,
        },
        body: JSON.stringify({
          model: GROQ_MODEL,
          messages: [
            { role: 'system', content: 'You explain city forecast data in plain, simple language a general audience understands. 4-5 short, easy sentences, max ~110 words, everyday words, no jargon, mention real numbers, never refuse.' },
            { role: 'user', content: prompt },
          ],
          temperature: 0.5,
          max_tokens: 400,
        }),
      });
      if (!response.ok) throw new Error(`Groq request failed (${response.status})`);
      const data = await response.json();
      const text = data?.choices?.[0]?.message?.content || '';
      const cleaned = text.replace(/<data>(.*?)<\/data>/s, '$1').replace(/```json|```/g, '').trim();
      const dataMatch = cleaned.match(/<data>(.*?)<\/data>/s);
      setPlain(dataMatch ? dataMatch[1].trim() : cleaned);
    } catch (e) {
      console.error('XAI Groq error', e);
      setPlain('Could not reach the language model just now — please try again in a moment.');
    } finally {
      setPlainLoading(false);
    }
  }, [predicted]);

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
        <div
          className="mt-3 inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-[0.68rem] font-mono"
          style={{ background: '#0e1a16', borderColor: '#1f3831', color: '#6fe7b7' }}
        >
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: '#6fe7b7', boxShadow: '0 0 6px #6fe7b7' }} />
          Signal trends computed from live trained-model forecast
          {predicted ? ` · T+1H → T+${predicted.length}H` : ' · fallback to current conditions'}
        </div>
      </div>

      {/* ── Signal cards ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
        {signals.map((sig, i) => {
          const fill = barFill(sig.trend, changePct(sig.key === 'traffic' ? trafficSeries : sig.key === 'air' ? airSeries : sig.key === 'weather' ? weatherSeries : energySeries));
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

              {/* Model readout */}
              <div className="flex items-baseline gap-2 mt-3">
                <span className="font-display text-[1.5rem] font-black leading-none" style={{ color: '#e9f3ee' }}>
                  {fmt(sig.value)}
                </span>
                <span className="text-[0.62rem] font-mono uppercase tracking-widest" style={{ color: '#5c7269' }}>
                  {sig.unit} · model T+6H
                </span>
                {sig.nextValue !== null && sig.value !== null && sig.nextValue !== sig.value && (
                  <span className="font-mono text-[0.7rem]" style={{ color: sig.trend === 'up' ? '#f0a857' : sig.trend === 'down' ? '#3498db' : '#8fa69b' }}>
                    from {fmt(sig.nextValue)}
                  </span>
                )}
              </div>

              {/* Reason text */}
              <p className="text-[0.82rem] leading-relaxed mt-2 mb-4" style={{ color: '#8fa69b' }}>
                {sig.reason(sig.trend, sig.value, sig.nextValue)}
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

      {/* ── Groq plain-language explanation ───────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.55, duration: 0.4 }}
        className="rounded-[16px] border p-6 mt-5"
        style={{ background: '#10201a', borderColor: '#1f3831' }}
      >
        <div className="flex items-center justify-between gap-4 mb-4">
          <div>
            <div
              className="text-[0.65rem] font-mono uppercase tracking-[0.16em] mb-1"
              style={{ color: '#5c7269' }}
            >
              In plain words (Groq)
            </div>
            <p className="text-[0.8rem]" style={{ color: '#8fa69b' }}>
              The same model forecast, explained in everyday language.
            </p>
          </div>
          <button
            onClick={generatePlain}
            disabled={plainLoading}
            className="px-4 py-2 rounded-[10px] text-[0.75rem] font-semibold transition-opacity disabled:opacity-50"
            style={{
              background: 'linear-gradient(90deg, #0f766e, #2ecc71)',
              color: '#e9f3ee',
              boxShadow: '0 0 14px rgba(46,204,113,0.25)',
            }}
          >
            {plainLoading ? 'Explaining…' : plain ? 'Regenerate' : 'Explain in plain words'}
          </button>
        </div>
        {plain ? (
          <p className="text-[0.9rem] leading-relaxed whitespace-pre-line" style={{ color: '#e9f3ee' }}>
            {plain}
          </p>
        ) : (
          <p className="text-[0.82rem]" style={{ color: '#5c7269' }}>
            Press the button and a language model will read the live forecast and describe it simply.
          </p>
        )}
      </motion.div>
    </div>
  );
}
