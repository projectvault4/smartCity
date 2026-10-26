import { useEffect, useMemo, useRef, useState } from 'react';
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Pause, Play, SkipForward, RotateCcw } from 'lucide-react';
import { backendApi, PROFILE_LABELS, Replay, TARGET_META, TargetKey, fmtNum, pct } from '../services/dataService';

const WINDOW = 72;
const THRESHOLDS: Partial<Record<TargetKey, { value: number; label: string }>> = {
  traffic_flow: { value: 20000, label: 'Severe traffic 20,000' },
  electricity_demand: { value: 6000, label: 'Grid peak 6,000 MW' },
  aqi: { value: 100, label: 'CPCB Moderate 100' }
};
const RULE_ORDER = ['ForeSightX (probabilistic)', 'Point forecast crosses threshold', 'Last-hour rule', 'Same-hour-yesterday rule'];

// Replay timestamps are naive local (IST) times from the held-out window.
const fmtTs = (ts: string) => {
  const [d, t] = ts.split('T');
  const [, m, day] = d.split('-');
  const h = Number(t.slice(0, 2));
  return `${day}/${m} ${h % 12 || 12}${h < 12 ? 'am' : 'pm'}`;
};

export default function AlertReplay() {
  const [replay, setReplay] = useState<Replay | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [index, setIndex] = useState(WINDOW);
  const [playing, setPlaying] = useState(false);
  const [target, setTarget] = useState<TargetKey>('traffic_flow');
  const [profile, setProfile] = useState('commuter');
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    backendApi.replay(0, 2000).then((r) => setReplay(r.data)).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!playing || !replay) return undefined;
    timer.current = setInterval(() => {
      setIndex((i) => {
        if (i >= replay.hours.length - 1) { setPlaying(false); return i; }
        return i + 1;
      });
    }, 350);
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [playing, replay]);

  const chartData = useMemo(() => {
    if (!replay) return [];
    return replay.hours.slice(Math.max(0, index - WINDOW), index + 1).map((h) => ({
      time: fmtTs(h.timestamp),
      actual: h[target].actual,
      forecast: h[target].forecast,
      band: [h[target].low, h[target].high],
      alert: h.alerts.length ? h[target].forecast : null
    }));
  }, [replay, index, target]);

  const issued = useMemo(() => {
    if (!replay) return [];
    const out: { ts: string; alert: Replay['hours'][number]['alerts'][number] }[] = [];
    replay.hours.slice(0, index + 1).forEach((h) => h.alerts.forEach((a) => out.push({ ts: h.timestamp, alert: a })));
    return out.reverse();
  }, [replay, index]);

  if (error) return <div className="text-red-300 text-sm">Replay unavailable: {error}</div>;
  if (!replay) return <div className="text-white/40 text-sm">Loading replay…</div>;

  const now = replay.hours[index];
  const running = { correct: issued.filter((x) => x.alert.outcome === 'correct').length, total: issued.length };
  const nextAlert = () => {
    const i = replay.hours.findIndex((h, k) => k > index && h.alerts.length > 0);
    if (i >= 0) { setPlaying(false); setIndex(i); }
  };
  const meta = TARGET_META[target];
  const thr = THRESHOLDS[target];

  return (
    <div className="space-y-8">
      <div>
        <div className="text-[11px] font-mono uppercase tracking-[0.2em] text-white/40">Historical replay · held-out data the model never saw</div>
        <h2 className="font-display text-3xl font-black text-white">Would the alerts have been right?</h2>
        <p className="text-white/50 text-sm max-w-3xl mt-2">
          The full pipeline replayed hour by hour over {replay.window.start.slice(0, 10)} → {replay.window.end.slice(0, 10)} ({replay.window.hours} hours).
          Each alert is issued one hour ahead from data up to that moment, then checked against what actually happened.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button onClick={() => setPlaying((p) => !p)} className="flex items-center gap-2 px-4 py-2 rounded-xl bg-home-acc/15 border border-home-acc/30 text-home-acc text-sm font-bold">
          {playing ? <Pause size={14} /> : <Play size={14} />} {playing ? 'Pause' : 'Play'}
        </button>
        <button onClick={nextAlert} className="flex items-center gap-2 px-4 py-2 rounded-xl border border-white/10 text-white/70 text-sm"><SkipForward size={14} /> Next alert</button>
        <button onClick={() => { setPlaying(false); setIndex(WINDOW); }} className="flex items-center gap-2 px-4 py-2 rounded-xl border border-white/10 text-white/70 text-sm"><RotateCcw size={14} /> Restart</button>
        <input type="range" min={WINDOW} max={replay.hours.length - 1} value={index} onChange={(e) => { setPlaying(false); setIndex(Number(e.target.value)); }} className="flex-1 min-w-[200px] accent-[#7df9a0]" aria-label="Replay position" />
        <div className="font-mono text-white text-sm">{fmtTs(now.timestamp)}</div>
      </div>

      <div className="flex flex-wrap gap-2">
        {(Object.keys(TARGET_META) as TargetKey[]).map((t) => (
          <button key={t} onClick={() => setTarget(t)} className={`px-3 py-1.5 rounded-full text-[11px] font-bold border ${target === t ? 'border-white/40 text-white bg-white/10' : 'border-white/10 text-white/50'}`}>
            {TARGET_META[t].label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 rounded-2xl bg-black/35 border border-white/5 p-5">
          <div className="text-[10px] uppercase tracking-widest text-white/40 mb-2">{meta.label}: actual vs forecast with 90% conformal range (last {WINDOW} h)</div>
          <div className="h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#ffffff10" vertical={false} />
                <XAxis dataKey="time" tick={{ fill: '#ffffff55', fontSize: 10 }} interval={11} />
                <YAxis tick={{ fill: '#ffffff55', fontSize: 10 }} domain={['auto', 'auto']} width={60} />
                <Tooltip contentStyle={{ background: '#0d1a10', border: '1px solid #ffffff22', fontSize: 11 }} formatter={(v: unknown) => (Array.isArray(v) ? `${fmtNum(v[0], meta.digits)} – ${fmtNum(v[1], meta.digits)}` : fmtNum(Number(v), meta.digits))} />
                <Area dataKey="band" stroke="none" fill={meta.color} fillOpacity={0.15} name="90% range" isAnimationActive={false} />
                <Line dataKey="forecast" stroke={meta.color} strokeWidth={2} dot={false} name="Forecast" isAnimationActive={false} />
                <Line dataKey="actual" stroke="#ffffff" strokeWidth={1.5} strokeDasharray="4 3" dot={false} name="Actual" isAnimationActive={false} />
                <Line dataKey="alert" stroke="none" dot={{ r: 5, fill: '#f97316', stroke: '#fff' }} name="Alert issued" isAnimationActive={false} />
                {thr && <ReferenceLine y={thr.value} stroke="#ef4444" strokeDasharray="6 4" label={{ value: thr.label, fill: '#ef9a9a', fontSize: 10, position: 'insideTopRight' }} />}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-3 gap-3 mt-4 text-[12px]">
            <div className="bg-white/5 rounded-xl p-3"><div className="text-white/40 text-[10px] uppercase">Actual</div><div className="font-mono text-white">{fmtNum(now[target].actual, meta.digits)} {meta.unit}</div></div>
            <div className="bg-white/5 rounded-xl p-3"><div className="text-white/40 text-[10px] uppercase">Forecast (1 h ahead)</div><div className="font-mono text-white">{fmtNum(now[target].forecast, meta.digits)}</div></div>
            <div className="bg-white/5 rounded-xl p-3"><div className="text-white/40 text-[10px] uppercase">Anomaly score</div><div className={`font-mono ${now.anomaly_flag ? 'text-red-300' : 'text-white'}`}>{now.anomaly_score === null || now.anomaly_score === undefined ? '—' : now.anomaly_score.toFixed(2)}{now.anomaly_flag ? ' · flagged' : ''}</div></div>
          </div>
        </div>

        <div className="rounded-2xl bg-black/35 border border-white/5 p-5 flex flex-col max-h-[520px]">
          <div className="flex items-center justify-between mb-3">
            <div className="text-[10px] uppercase tracking-widest text-white/40">Alerts sent so far</div>
            <div className="font-mono text-sm text-white">{running.correct}/{running.total} correct</div>
          </div>
          <select value={profile} onChange={(e) => setProfile(e.target.value)} className="mb-3 bg-black/50 border border-white/10 rounded-lg text-[12px] text-white/80 p-2" aria-label="Message profile">
            {Object.entries(PROFILE_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
          <div className="overflow-y-auto space-y-2 pr-1">
            {issued.length === 0 && <div className="text-white/40 text-[12px]">No alert issued yet in this replay.</div>}
            {issued.map(({ ts, alert }, i) => (
              <div key={`${ts}-${alert.event}-${i}`} className={`rounded-xl p-3 border text-[12px] ${alert.outcome === 'correct' ? 'border-home-acc/30 bg-home-acc/5' : 'border-red-400/30 bg-red-500/5'}`}>
                <div className="flex justify-between gap-2">
                  <span className="font-bold text-white">{alert.label}</span>
                  <span className={alert.outcome === 'correct' ? 'text-home-acc' : 'text-red-300'}>{alert.outcome === 'correct' ? '✓ happened' : '✗ false alarm'}</span>
                </div>
                <div className="text-white/40 font-mono">{fmtTs(ts)} · {pct(alert.probability)}</div>
                <p className="text-white/70 mt-1">{alert.messages[profile] || alert.messages[alert.audience[0]]}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="rounded-2xl bg-black/35 border border-white/5 p-5 overflow-x-auto">
        <div className="text-[10px] uppercase tracking-widest text-white/40 mb-3">Scorecard over the whole window · cooldown {replay.cooldown_hours} h</div>
        <table className="w-full text-[12px]">
          <thead>
            <tr className="text-white/40 text-left">
              <th className="py-2 pr-4 font-normal">Alert rule</th>
              <th className="py-2 pr-4 font-normal text-right">Alerts sent</th>
              <th className="py-2 pr-4 font-normal text-right">Correct</th>
              <th className="py-2 pr-4 font-normal text-right">False alarms / week</th>
              <th className="py-2 pr-4 font-normal text-right">Events caught</th>
              <th className="py-2 font-normal text-right">Warned before onset</th>
            </tr>
          </thead>
          <tbody>
            {RULE_ORDER.filter((r) => replay.scorecard[r]).map((rule) => {
              const s = replay.scorecard[rule];
              const ours = rule.startsWith('ForeSightX');
              return (
                <tr key={rule} className={`border-t border-white/5 ${ours ? 'text-home-acc font-bold' : 'text-white/70'}`}>
                  <td className="py-2 pr-4">{rule}</td>
                  <td className="py-2 pr-4 text-right font-mono">{s.notifications_sent}</td>
                  <td className="py-2 pr-4 text-right font-mono">{pct(s.notification_precision)}</td>
                  <td className="py-2 pr-4 text-right font-mono">{s.false_alarms_per_week.toFixed(1)}</td>
                  <td className="py-2 pr-4 text-right font-mono">{s.episodes_caught}/{s.event_episodes}</td>
                  <td className="py-2 text-right font-mono">{s.episodes_warned_before_onset}/{s.event_episodes}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="text-[11px] text-white/35 mt-3">
          Events: severe traffic (&gt;20,000 veh/hr), grid peak (&gt;6,000 MW), and both together. AQI and heat events did not occur in this monsoon window,
          and the system raised no false AQI or heat alerts. Cutoffs were tuned on the validation window, never on these hours.
        </p>
      </div>
    </div>
  );
}
