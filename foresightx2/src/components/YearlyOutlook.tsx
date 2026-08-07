import { useEffect, useMemo, useState } from 'react';
import { backendApi, YearlyForecast } from '../services/dataService';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, Legend,
} from 'recharts';

// ─── Fallback (backend offline) ───────────────────────────────────────────────
const FALLBACK: YearlyForecast = {
  year: 2026,
  city: 'bangalore',
  source: 'fallback_demo',
  generatedFrom: '2026-01-01 00:00:00',
  through: '2026-12-31 23:00:00',
  totalHours: 8760,
  annual: { trafficFlow: 12168, aqi: 119, temperature: 24, humidity: 65, electricityDemand: 261 },
  peaks: {
    peakTrafficMonth: { month: 10, label: 'October', days: 31, trafficFlow: 13486, aqi: 123, temperature: 23, humidity: 76, electricityDemand: 266, maxTrafficFlow: 26131, maxAqi: 197, maxTemperature: 32 },
    peakAqiMonth: { month: 7, label: 'July', days: 31, trafficFlow: 12434, aqi: 129, temperature: 24, humidity: 71, electricityDemand: 268, maxTrafficFlow: 25461, maxAqi: 197, maxTemperature: 27 },
    peakTemperatureMonth: { month: 4, label: 'April', days: 30, trafficFlow: 11832, aqi: 111, temperature: 28, humidity: 50, electricityDemand: 286, maxTrafficFlow: 21975, maxAqi: 169, maxTemperature: 32 },
    hottestMonth: { month: 4, label: 'April', days: 30, trafficFlow: 11832, aqi: 111, temperature: 28, humidity: 50, electricityDemand: 286, maxTrafficFlow: 21975, maxAqi: 169, maxTemperature: 32 },
    coolestMonth: { month: 12, label: 'December', days: 31, trafficFlow: 12390, aqi: 128, temperature: 22, humidity: 68, electricityDemand: 268, maxTrafficFlow: 26131, maxAqi: 190, maxTemperature: 27 },
  },
  monthly: [
    { month: 1, label: 'January', days: 31, trafficFlow: 10531, aqi: 112, temperature: 22, humidity: 62, electricityDemand: 220, maxTrafficFlow: 26131, maxAqi: 190, maxTemperature: 27 },
    { month: 2, label: 'February', days: 28, trafficFlow: 13108, aqi: 120, temperature: 23, humidity: 56, electricityDemand: 284, maxTrafficFlow: 26131, maxAqi: 197, maxTemperature: 30 },
    { month: 3, label: 'March', days: 31, trafficFlow: 11902, aqi: 113, temperature: 25, humidity: 49, electricityDemand: 269, maxTrafficFlow: 26131, maxAqi: 197, maxTemperature: 32 },
    { month: 4, label: 'April', days: 30, trafficFlow: 11832, aqi: 111, temperature: 28, humidity: 50, electricityDemand: 286, maxTrafficFlow: 21975, maxAqi: 169, maxTemperature: 32 },
    { month: 5, label: 'May', days: 31, trafficFlow: 11593, aqi: 112, temperature: 26, humidity: 62, electricityDemand: 280, maxTrafficFlow: 26131, maxAqi: 197, maxTemperature: 32 },
    { month: 6, label: 'June', days: 30, trafficFlow: 9896, aqi: 116, temperature: 24, humidity: 67, electricityDemand: 207, maxTrafficFlow: 26131, maxAqi: 197, maxTemperature: 30 },
    { month: 7, label: 'July', days: 31, trafficFlow: 12434, aqi: 129, temperature: 24, humidity: 71, electricityDemand: 268, maxTrafficFlow: 25461, maxAqi: 197, maxTemperature: 27 },
    { month: 8, label: 'August', days: 31, trafficFlow: 12681, aqi: 120, temperature: 23, humidity: 75, electricityDemand: 258, maxTrafficFlow: 26131, maxAqi: 190, maxTemperature: 27 },
    { month: 9, label: 'September', days: 30, trafficFlow: 13021, aqi: 124, temperature: 23, humidity: 72, electricityDemand: 263, maxTrafficFlow: 26131, maxAqi: 197, maxTemperature: 27 },
    { month: 10, label: 'October', days: 31, trafficFlow: 13486, aqi: 123, temperature: 23, humidity: 76, electricityDemand: 266, maxTrafficFlow: 26131, maxAqi: 197, maxTemperature: 27 },
    { month: 11, label: 'November', days: 30, trafficFlow: 13415, aqi: 124, temperature: 22, humidity: 76, electricityDemand: 266, maxTrafficFlow: 26131, maxAqi: 197, maxTemperature: 27 },
    { month: 12, label: 'December', days: 31, trafficFlow: 12390, aqi: 128, temperature: 22, humidity: 68, electricityDemand: 268, maxTrafficFlow: 26131, maxAqi: 190, maxTemperature: 27 },
  ],
  series: [],
};

const COLORS: Record<string, string> = {
  traffic: '#ff8a3d',
  aqi: '#46c9b0',
  temperature: '#6fb8e8',
  energy: '#b48eea',
  humidity: '#7ac7a0',
};

const fmt = (n: number | null | undefined, decimals = 0) => {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—';
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: decimals });
};

const AQI_BANDS = [
  { max: 50, color: '#4ade80', label: 'Good' },
  { max: 100, color: '#facc15', label: 'Moderate' },
  { max: 150, color: '#fb923c', label: 'Poor' },
  { max: 200, color: '#f87171', label: 'Very Poor' },
  { max: 999, color: '#a855f7', label: 'Severe' },
];

const aqiColor = (value: number) => AQI_BANDS.find((b) => value <= b.max)?.color || '#a855f7';

// Downsample the full hourly series to ~daily points for the chart.
const downsampleDaily = (series: YearlyForecast['series']) => {
  if (!series || series.length === 0) return [];
  const byDay = new Map<string, { ts: string; traffic: number[]; aqi: number[]; temp: number[] }>();
  for (const p of series) {
    const key = String(p.timestamp).slice(0, 10);
    if (!byDay.has(key)) byDay.set(key, { ts: p.timestamp, traffic: [], aqi: [], temp: [] });
    const bucket = byDay.get(key)!;
    if (p.trafficFlow !== null) bucket.traffic.push(p.trafficFlow);
    if (p.aqi !== null) bucket.aqi.push(p.aqi);
    if (p.temperature !== null) bucket.temp.push(p.temperature);
  }
  const avg = (arr: number[]) => arr.reduce((s, v) => s + v, 0) / Math.max(1, arr.length);
  return Array.from(byDay.values()).map((bucket) => ({
    ts: bucket.ts.slice(0, 10),
    traffic: Math.round(avg(bucket.traffic)),
    aqi: Math.round(avg(bucket.aqi) * 10) / 10,
    temperature: Math.round(avg(bucket.temp) * 10) / 10,
  }));
};

const cardStyle: React.CSSProperties = {
  background: '#0e1712',
  border: '1px solid #1e2b24',
  borderRadius: 14,
  padding: '20px 22px',
};

const labelStyle: React.CSSProperties = {
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 11,
  letterSpacing: '0.14em',
  color: '#6f8079',
  textTransform: 'uppercase',
};

const valueStyle: React.CSSProperties = {
  fontFamily: "'Space Grotesk', sans-serif",
  fontSize: 30,
  fontWeight: 500,
  color: '#edf3ee',
  marginTop: 6,
};

const YearlyOutlook = ({ city = 'bangalore' }: { city?: string }) => {
  const [data, setData] = useState<YearlyForecast | null>(null);
  const [status, setStatus] = useState<'loading' | 'model' | 'fallback'>('loading');

  useEffect(() => {
    let mounted = true;
    setStatus('loading');
    backendApi.modelYearlyForecast(city, 2026, 'monthly')
      .then((response) => { if (mounted) { setData(response.data); setStatus('model'); } })
      .catch(() => { if (mounted) { setData(FALLBACK); setStatus('fallback'); } });
    return () => { mounted = false; };
  }, [city]);

  const daily = useMemo(() => downsampleDaily(data?.series || []), [data?.series]);
  const monthLabels = useMemo(() => (data?.monthly || []).map((m) => m.label.slice(0, 3)), [data]);
  const annual = data?.annual || FALLBACK.annual;
  const peaks = data?.peaks || FALLBACK.peaks;

  const aqiBarColors = (data?.monthly || FALLBACK.monthly).map((m) => aqiColor(m.aqi ?? 0));

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
      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="flex justify-between items-end flex-wrap gap-6 mb-2">
        <div>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, letterSpacing: '0.18em', color: '#6f8079', textTransform: 'uppercase', marginBottom: 10 }}>
            Long-range intelligence · 12-month outlook
          </div>
          <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: 38, letterSpacing: '-0.01em', margin: 0 }}>
            {data?.year || 2026} Forecast
          </h1>
          <div style={{ fontSize: 14, color: '#6f8079', marginTop: 8, maxWidth: 560, lineHeight: 1.5 }}>
            Full-year prediction generated by the trained hybrid model from the 2022–2025 urban
            history. Monthly means below, with hourly detail available in the year timeline. Peak
            cards show the single highest hourly value of that month.
          </div>
        </div>
        <div className="text-right">
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
      </header>

      {/* ── Annual summary cards ───────────────────────────────── */}
      <div className="grid mt-8 gap-4" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
        {[
          { label: 'Avg Traffic Flow', value: `${fmt(annual.trafficFlow)}`, unit: 'veh/hr', color: COLORS.traffic },
          { label: 'Avg AQI', value: `${fmt(annual.aqi)}`, unit: 'AQI', color: COLORS.aqi },
          { label: 'Avg Temperature', value: `${fmt(annual.temperature, 1)}°`, unit: 'C', color: COLORS.temperature },
          { label: 'Avg Grid Load', value: `${fmt(annual.electricityDemand)}`, unit: 'MW', color: COLORS.energy },
        ].map((m) => (
          <div key={m.label} style={cardStyle}>
            <div className="flex items-center gap-2.5">
              <span className="w-[9px] h-[9px] rounded-full" style={{ background: m.color }} />
              <span style={labelStyle}>{m.label}</span>
            </div>
            <div style={valueStyle}>{m.value}<span style={{ color: '#6f8079', fontSize: 12, fontWeight: 400 }}> {m.unit}</span></div>
          </div>
        ))}
      </div>

      {/* ── Peak highlights ────────────────────────────────────── */}
      <div className="grid mt-4 gap-4" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
        {[
          { title: 'Peak AQI month', label: peaks.peakAqiMonth?.label, value: peaks.peakAqiMonth?.maxAqi, unit: 'peak AQI', color: COLORS.aqi },
          { title: 'Hottest month', label: peaks.hottestMonth?.label, value: peaks.hottestMonth?.maxTemperature, unit: 'peak °C', color: COLORS.temperature },
          { title: 'Peak traffic month', label: peaks.peakTrafficMonth?.label, value: peaks.peakTrafficMonth?.maxTrafficFlow, unit: 'peak veh/hr', color: COLORS.traffic },
        ].map((m) => (
          <div key={m.title} className="flex items-center justify-between" style={cardStyle}>
            <div>
              <div style={labelStyle}>{m.title}</div>
              <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 20, fontWeight: 600, marginTop: 4 }}>
                {m.label || '—'}
              </div>
            </div>
            <div className="text-right">
              <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 26, fontWeight: 600, color: m.color }}>
                {fmt(m.value)}
              </div>
              <div style={{ fontSize: 11, color: '#6f8079' }}>{m.unit}</div>
            </div>
          </div>
        ))}
      </div>

      {/* ── Monthly AQI bar chart ──────────────────────────────── */}
      <div style={{ ...cardStyle, marginTop: 20, padding: '24px 26px' }}>
        <div style={labelStyle}>Monthly average AQI</div>
        <div className="mt-4" style={{ width: '100%', height: 240 }}>
          <ResponsiveContainer>
            <BarChart data={data?.monthly || FALLBACK.monthly} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2b24" vertical={false} />
              <XAxis dataKey="label" tick={{ fill: '#6f8079', fontSize: 11 }} axisLine={{ stroke: '#1e2b24' }} tickLine={false} />
              <YAxis tick={{ fill: '#6f8079', fontSize: 11 }} axisLine={false} tickLine={false} domain={[0, 200]} />
              <Tooltip
                cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                contentStyle={{ background: '#0e1712', border: '1px solid #1e2b24', borderRadius: 10, color: '#edf3ee' }}
                labelStyle={{ color: '#6f8079' }}
                formatter={(value: any, name: any) => [`${value} AQI`, 'Average AQI']}
              />
              <Bar dataKey="aqi" name="AQI" radius={[6, 6, 0, 0]}>
                {(data?.monthly || FALLBACK.monthly).map((m, i) => (
                  <Cell key={i} fill={aqiColor(m.aqi ?? 0)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="flex flex-wrap gap-3 mt-3" style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: '#6f8079' }}>
          {AQI_BANDS.map((b) => (
            <span key={b.label} className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm" style={{ background: b.color }} />{b.label}
            </span>
          ))}
        </div>
      </div>

      {/* ── Traffic + temperature trend ────────────────────────── */}
      <div className="grid mt-4 gap-4" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
        <div style={cardStyle}>
          <div style={labelStyle}>Daily average traffic flow</div>
          <div className="mt-4" style={{ width: '100%', height: 220 }}>
            <ResponsiveContainer>
              <AreaChart data={daily} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2b24" vertical={false} />
                <XAxis dataKey="ts" tick={{ fill: '#6f8079', fontSize: 10 }} tickFormatter={(v: string) => v.slice(5)} axisLine={{ stroke: '#1e2b24' }} tickLine={false} minTickGap={40} />
                <YAxis tick={{ fill: '#6f8079', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: '#0e1712', border: '1px solid #1e2b24', borderRadius: 10, color: '#edf3ee' }} labelStyle={{ color: '#6f8079' }} />
                <Area type="monotone" dataKey="traffic" stroke={COLORS.traffic} fill={COLORS.traffic} fillOpacity={0.16} strokeWidth={2} name="Traffic flow" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div style={cardStyle}>
          <div style={labelStyle}>Daily average temperature</div>
          <div className="mt-4" style={{ width: '100%', height: 220 }}>
            <ResponsiveContainer>
              <AreaChart data={daily} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2b24" vertical={false} />
                <XAxis dataKey="ts" tick={{ fill: '#6f8079', fontSize: 10 }} tickFormatter={(v: string) => v.slice(5)} axisLine={{ stroke: '#1e2b24' }} tickLine={false} minTickGap={40} />
                <YAxis tick={{ fill: '#6f8079', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: '#0e1712', border: '1px solid #1e2b24', borderRadius: 10, color: '#edf3ee' }} labelStyle={{ color: '#6f8079' }} />
                <Area type="monotone" dataKey="temperature" stroke={COLORS.temperature} fill={COLORS.temperature} fillOpacity={0.16} strokeWidth={2} name="Temperature" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* ── Monthly detail table ───────────────────────────────── */}
      <div style={{ ...cardStyle, marginTop: 20, padding: '24px 26px' }}>
        <div style={labelStyle}>Month-by-month detail</div>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left" style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #1e2b24' }}>
                {['Month', 'Traffic (veh/hr)', 'AQI', 'Temp (°C)', 'Humidity (%)', 'Grid (MW)'].map((h) => (
                  <th key={h} className="pb-3" style={labelStyle}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(data?.monthly || FALLBACK.monthly).map((m) => (
                <tr key={m.month} style={{ borderBottom: '1px solid rgba(30,43,36,0.6)' }}>
                  <td className="py-2.5 font-semibold" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>{m.label}</td>
                  <td className="py-2.5 font-mono text-[13px]" style={{ color: COLORS.traffic }}>{fmt(m.trafficFlow)}</td>
                  <td className="py-2.5 font-mono text-[13px]" style={{ color: aqiColor(m.aqi ?? 0) }}>{fmt(m.aqi)}</td>
                  <td className="py-2.5 font-mono text-[13px]" style={{ color: COLORS.temperature }}>{fmt(m.temperature, 1)}</td>
                  <td className="py-2.5 font-mono text-[13px]" style={{ color: COLORS.humidity }}>{fmt(m.humidity)}</td>
                  <td className="py-2.5 font-mono text-[13px]" style={{ color: COLORS.energy }}>{fmt(m.electricityDemand)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Footer ─────────────────────────────────────────────── */}
      <footer className="mt-7" style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: '#465049', lineHeight: 1.8 }}>
        <div>Model: {data?.source || 'trained hybrid forecaster'} · trained on 2022–2025 history</div>
        <div>Horizon: {data?.generatedFrom || 'Jan 2026'} → {data?.through || 'Dec 2026'} · {fmt(data?.totalHours)} hourly steps</div>
        <div>Damped recursive forecast blends the trained model with the learned seasonal profile to keep the year-ahead outlook stable.</div>
      </footer>
    </div>
  );
};

export default YearlyOutlook;
