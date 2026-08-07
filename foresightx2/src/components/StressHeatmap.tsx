import { useEffect, useMemo, useState } from 'react';
import Card from './Card';
import { backendApi, YearlyForecast } from '../services/dataService';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

const getColor = (val: number) => {
  if (val > 80) return 'bg-[#ef4444]'; // High Stress
  if (val > 60) return 'bg-[#f97316]'; // Moderate
  if (val > 40) return 'bg-[#eab308]'; // Mild
  return 'bg-[#22c55e]'; // Low Stress
};

// Deterministic pseudo-random fallback so the simulation doesn't flicker on re-render.
const fakeValue = (month: number, day: number) => {
  const x = Math.sin(month * 127.1 + day * 311.7) * 43758.5453;
  let val = (x - Math.floor(x)) * 100;
  if (month === 0 && day < 7) val = 95; // New Year stress
  if (month === 2 && day > 20) val = 15; // Simulated lockdown green
  return val;
};

const avg = (arr: number[]) => (arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : null);

const StressHeatmap = ({ city = 'bangalore' }: { city?: string }) => {
  const [yearly, setYearly] = useState<YearlyForecast | null>(null);
  const [status, setStatus] = useState<'loading' | 'model' | 'fallback'>('loading');

  useEffect(() => {
    let mounted = true;
    setStatus('loading');
    backendApi.modelYearlyForecast(city, 2026, 'monthly')
      .then((response) => { if (mounted) { setYearly(response.data); setStatus('model'); } })
      .catch(() => { if (mounted) setStatus('fallback'); });
    return () => { mounted = false; };
  }, [city]);

  // Per-day stress index (0-100) computed from the real 2026 hourly forecast.
  const dailyStress = useMemo(() => {
    const map = new Map<string, number>();
    if (!yearly || yearly.series.length === 0) return map;

    const byDay = new Map<string, { traffic: number[]; aqi: number[]; energy: number[] }>();
    for (const p of yearly.series) {
      const key = String(p.timestamp).slice(0, 10);
      if (!byDay.has(key)) byDay.set(key, { traffic: [], aqi: [], energy: [] });
      const bucket = byDay.get(key)!;
      if (p.trafficFlow !== null) bucket.traffic.push(p.trafficFlow);
      if (p.aqi !== null) bucket.aqi.push(p.aqi);
      if (p.electricityDemand !== null) bucket.energy.push(p.electricityDemand);
    }

    const range = (pick: (b: { traffic: number[]; aqi: number[]; energy: number[] }) => number[]) => {
      const values = [...byDay.values()].map(pick).flat().filter((v) => Number.isFinite(v));
      return values.length
        ? { min: Math.min(...values), max: Math.max(...values) }
        : { min: 0, max: 1 };
    };
    const t = range((b) => b.traffic);
    const a = range((b) => b.aqi);
    const e = range((b) => b.energy);

    const norm = (v: number, lo: number, hi: number) => (hi === lo ? 50 : ((v - lo) / (hi - lo)) * 100);

    for (const [date, bucket] of byDay) {
      const parts: number[] = [];
      const tv = avg(bucket.traffic);
      const av = avg(bucket.aqi);
      const ev = avg(bucket.energy);
      if (tv !== null) parts.push(norm(tv, t.min, t.max));
      if (av !== null) parts.push(norm(av, a.min, a.max));
      if (ev !== null) parts.push(norm(ev, e.min, e.max));
      map.set(date, parts.length ? parts.reduce((s, v) => s + v, 0) / parts.length : 0);
    }
    return map;
  }, [yearly]);

  const hasModelData = status === 'model' && dailyStress.size > 0;

  const dateKey = (month: number, day: number) =>
    `2026-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;

  return (
    <Card title="Temporal Heatmap Calendar" theme="home">
      <div className="p-2 space-y-6">
        <p className="text-sm text-white/50">GitHub-style urban stress calendar (Traffic + AQI + Energy). Color intensity indicates combined environmental pressure.</p>

        <div className="flex flex-col gap-3">
          {MONTHS.map((month, mi) => (
            <div key={month} className="flex items-center gap-3">
              <div className="w-8 text-[10px] font-bold text-white/30 uppercase">{month}</div>
              <div className="flex gap-1 flex-1 overflow-auto pb-1">
                {Array.from({ length: DAYS_IN_MONTH[mi] }).map((_, day) => {
                  const key = dateKey(mi + 1, day + 1);
                  const val = hasModelData && dailyStress.has(key)
                    ? dailyStress.get(key)!
                    : fakeValue(mi, day + 1);
                  return (
                    <div
                      key={key}
                      className={`w-3 h-3 rounded-[2px] shrink-0 ${getColor(val)} opacity-60 hover:opacity-100 transition-opacity cursor-pointer`}
                      title={`${month} ${day + 1}: Stress Index ${Math.round(val)}`}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        <div className="flex items-center gap-4 pt-4 border-t border-white/5">
           <div className="text-[10px] text-white/30 font-medium uppercase tracking-widest">Stress index:</div>
           <div className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 bg-[#22c55e] rounded-[1px]" />
              <div className="w-2.5 h-2.5 bg-[#eab308] rounded-[1px]" />
              <div className="w-2.5 h-2.5 bg-[#f97316] rounded-[1px]" />
              <div className="w-2.5 h-2.5 bg-[#ef4444] rounded-[1px]" />
              <span className="text-[10px] text-white/40 ml-1">Low → High Stress</span>
           </div>
        </div>

        <div className="flex items-center gap-4 pt-2 border-t border-white/5">
          <span className="text-[10px] text-white/30 font-medium uppercase tracking-widest">Source:</span>
          <span className="text-[10px] text-white/40">
            {status === 'loading' ? 'Loading 2026 forecast…' : hasModelData ? `Trained model · 2026 forecast · ${yearly?.totalHours} hourly steps` : 'Fallback simulation (backend offline)'}
          </span>
        </div>
      </div>
    </Card>
  );
};

export default StressHeatmap;
