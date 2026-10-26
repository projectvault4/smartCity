import { useEffect, useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import Card from './Card';
import { backendApi, ForecastPoint, TARGET_META, TargetKey, WhatIf, fmtNum } from '../services/dataService';

// Data-driven what-if: every scenario was re-run through the full deployed
// chain (features, SARIMA expert, network, calibration, blend) offline, so the
// numbers here are the model's real response, not a slider multiplier.

const DRIVERS = [
  { key: 'traffic_flow', label: 'Traffic', unit: '%', steps: [-30, -20, -10, 0, 10, 20, 30] },
  { key: 'temperature', label: 'Temperature', unit: '°C', steps: [-2, -1, 0, 1, 2] },
  { key: 'humidity', label: 'Humidity', unit: '%', steps: [-10, 0, 10] }
];
const TARGETS = Object.keys(TARGET_META) as TargetKey[];

const scenarioId = (driver: string, step: number) => {
  const prefix = driver === 'traffic_flow' ? 'traffic' : driver;
  return `${prefix}_${step > 0 ? '+' : ''}${step}`;
};

const RippleSimulator = (_props: { forecast?: ForecastPoint[] | null }) => {
  const [data, setData] = useState<WhatIf | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [driver, setDriver] = useState('traffic_flow');
  const [step, setStep] = useState(-20);
  const [focus, setFocus] = useState<TargetKey>('aqi');

  useEffect(() => {
    backendApi.whatIf().then((r) => setData(r.data)).catch((e) => setError(e.message));
  }, []);

  const driverMeta = DRIVERS.find((d) => d.key === driver)!;
  const scenario = useMemo(() => data?.scenarios.find((s) => s.id === scenarioId(driver, step)) || null, [data, driver, step]);
  const byHour = useMemo(() => {
    if (!scenario) return [];
    const e = scenario.effects[focus];
    return Array.from({ length: 24 }, (_, h) => ({ hour: `${h % 12 || 12}${h < 12 ? 'a' : 'p'}`, change: e?.by_hour[String(h)] ?? 0 }));
  }, [scenario, focus]);

  if (error) return <Card title="What-if Explorer" theme="traffic"><p className="text-red-300">Unavailable: {error}</p></Card>;
  if (!data) return <Card title="What-if Explorer" theme="traffic"><p>Loading counterfactual runs…</p></Card>;

  return (
    <div className="space-y-6">
      <Card title="What-if Explorer (counterfactual runs through the real model)" theme="traffic">
        <p className="text-sm text-white/55 mb-6">
          Change one input for the whole held-out window ({data.window.start.slice(0, 10)} → {data.window.end.slice(0, 10)}) and see how the
          deployed forecast responds. {data.caveat}
        </p>

        <div className="flex flex-wrap gap-2 mb-4">
          {DRIVERS.map((d) => (
            <button key={d.key} onClick={() => { setDriver(d.key); setStep(d.steps.find((s) => s !== 0) ?? 0); }}
              className={`px-3 py-1.5 rounded-full text-[11px] font-bold border ${driver === d.key ? 'border-traf-acc/50 text-traf-acc bg-traf-acc/10' : 'border-white/10 text-white/50'}`}>
              {d.label}
            </button>
          ))}
        </div>

        <div className="px-4 py-5 bg-white/5 rounded-2xl border border-white/5 mb-6">
          <div className="flex justify-between items-center mb-3">
            <span className="text-xs font-bold text-traf-acc uppercase tracking-widest">{driverMeta.label} change</span>
            <span className="text-2xl font-display font-extrabold text-traf-acc">{step > 0 ? '+' : ''}{step}{driverMeta.unit}</span>
          </div>
          <input type="range" min={0} max={driverMeta.steps.length - 1} step={1}
            value={driverMeta.steps.indexOf(step)} onChange={(e) => setStep(driverMeta.steps[Number(e.target.value)])}
            className="w-full accent-traf-acc" aria-label={`${driverMeta.label} change`} />
          <div className="flex justify-between text-[10px] text-white/30 mt-1">
            {driverMeta.steps.map((s) => <span key={s}>{s > 0 ? '+' : ''}{s}</span>)}
          </div>
        </div>

        {step === 0 || !scenario ? (
          <p className="text-white/50 text-sm">Pick a non-zero change to see the model's response.</p>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-[12px]">
                <thead>
                  <tr className="text-white/40 text-left">
                    <th className="py-2 font-normal">Forecast</th>
                    <th className="py-2 font-normal text-right">Average change</th>
                    <th className="py-2 font-normal text-right">% change</th>
                    <th className="py-2 font-normal text-right">90% of hours between</th>
                  </tr>
                </thead>
                <tbody>
                  {TARGETS.map((t) => {
                    const e = scenario.effects[t];
                    const m = TARGET_META[t];
                    const digits = t === 'aqi' || t === 'temperature' ? 3 : 1;
                    return (
                      <tr key={t} onClick={() => setFocus(t)} className={`border-t border-white/5 cursor-pointer ${focus === t ? 'text-white' : 'text-white/65'}`}>
                        <td className="py-2">{m.label}</td>
                        <td className="py-2 text-right font-mono">{e.mean_change > 0 ? '+' : ''}{fmtNum(e.mean_change, digits)} {m.unit}</td>
                        <td className="py-2 text-right font-mono">{e.mean_change_pct === null ? '—' : `${e.mean_change_pct > 0 ? '+' : ''}${e.mean_change_pct.toFixed(2)}%`}</td>
                        <td className="py-2 text-right font-mono">{fmtNum(e.p05, digits)} to {fmtNum(e.p95, digits)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="mt-6">
              <div className="text-[10px] uppercase tracking-widest text-white/40 mb-2">{TARGET_META[focus].label}: average change by hour of day</div>
              <div className="h-[180px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={byHour} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="#ffffff10" vertical={false} />
                    <XAxis dataKey="hour" tick={{ fill: '#ffffff55', fontSize: 10 }} interval={2} />
                    <YAxis tick={{ fill: '#ffffff55', fontSize: 10 }} width={60} />
                    <Tooltip contentStyle={{ background: '#0d1a10', border: '1px solid #ffffff22', fontSize: 11 }} formatter={(v: unknown) => fmtNum(Number(v), 3)} />
                    <ReferenceLine y={0} stroke="#ffffff33" />
                    <Bar dataKey="change" fill={TARGET_META[focus].color} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}
      </Card>

      <Card title="What the evidence says about cross-domain effects" theme="home">
        <div className="space-y-2">
          {data.findings.map((f) => <p key={f} className="text-[13px] text-white/70">• {f}</p>)}
        </div>
        <p className="text-[12px] text-white/40 mt-4">
          Hour-ahead forecasts are dominated by each signal's own recent history and daily cycle, so changing traffic barely moves the AQI forecast.
          Day-level data shows where real links exist (temperature → electricity demand) and where they do not (traffic → AQI) in Bengaluru, Feb–Jun 2025.
        </p>
      </Card>
    </div>
  );
};

export default RippleSimulator;
