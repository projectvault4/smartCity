import { useEffect, useMemo, useState } from 'react';
import {
  backendApi, Decision, DecisionAlert, PROFILE_LABELS, TARGET_META, TargetKey,
  fmtCityHour, fmtNum, pct
} from '../services/dataService';

const PROFILES = ['commuter', 'outdoor_worker', 'respiratory', 'elderly', 'child'];
const SEVERITY_STYLE: Record<string, { border: string; chip: string; label: string }> = {
  critical: { border: 'border-red-500/50', chip: 'bg-red-500/15 text-red-300', label: 'Critical' },
  warning: { border: 'border-orange-400/50', chip: 'bg-orange-400/15 text-orange-200', label: 'Warning' },
  advisory: { border: 'border-sky-400/40', chip: 'bg-sky-400/15 text-sky-200', label: 'Advisory' }
};
const EVENT_SHORT: Record<string, string> = {
  traffic_congestion: 'Severe traffic',
  grid_peak: 'Grid peak',
  compound_stress: 'Traffic + grid',
  air_quality_sensitive: 'AQI Moderate+',
  air_quality_poor: 'AQI Poor+',
  heat_stress: 'Heat stress'
};

function probColor(p: number) {
  if (p >= 0.75) return 'rgba(239,68,68,0.85)';
  if (p >= 0.5) return 'rgba(249,115,22,0.75)';
  if (p >= 0.25) return 'rgba(234,179,8,0.55)';
  if (p > 0.02) return 'rgba(125,249,160,0.18)';
  return 'rgba(255,255,255,0.03)';
}

function AlertCard({ alert, profile }: { alert: DecisionAlert; profile: string }) {
  const style = SEVERITY_STYLE[alert.severity] || SEVERITY_STYLE.advisory;
  const shownProfile = alert.audience.includes(profile) ? profile : alert.audience[0];
  const message = alert.messages[shownProfile];
  const sms = alert.sms[shownProfile];
  const meta = TARGET_META[alert.leadTarget];
  return (
    <div className={`rounded-2xl border ${style.border} bg-black/40 p-5 space-y-4`}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className={`text-[10px] font-bold uppercase tracking-widest px-2 py-1 rounded ${style.chip}`}>{style.label}</span>
          <h4 className="text-white font-bold text-base">{alert.label}</h4>
        </div>
        <div className="text-right">
          <div className="font-display text-3xl font-black text-white">{pct(alert.probability)}</div>
          <div className="text-[10px] text-white/40 uppercase tracking-widest">chance · alert cutoff {pct(alert.cutoff)}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-[12px]">
        <div className="bg-white/5 rounded-xl p-3">
          <div className="text-white/40 text-[10px] uppercase tracking-widest mb-1">When (IST)</div>
          <div className="text-white font-mono">{fmtCityHour(alert.startsAt)} – {fmtCityHour(alert.endsAt)}</div>
          <div className="text-white/40">{alert.activeHours} h above cutoff · peak {fmtCityHour(alert.peakAt)}</div>
        </div>
        <div className="bg-white/5 rounded-xl p-3">
          <div className="text-white/40 text-[10px] uppercase tracking-widest mb-1">Forecast · 90% range</div>
          <div className="text-white font-mono">{fmtNum(alert.forecast?.value, meta.digits)} {meta.unit}</div>
          <div className="text-white/40 font-mono">{fmtNum(alert.forecast?.low, meta.digits)} – {fmtNum(alert.forecast?.high, meta.digits)}</div>
        </div>
        <div className="bg-white/5 rounded-xl p-3">
          <div className="text-white/40 text-[10px] uppercase tracking-widest mb-1">Vs a normal hour</div>
          <div className="text-white font-mono">
            {alert.forecast?.vsTypicalPct === null || alert.forecast?.vsTypicalPct === undefined
              ? '—'
              : `${alert.forecast.vsTypicalPct > 0 ? '+' : ''}${alert.forecast.vsTypicalPct.toFixed(0)}%`}
          </div>
          <div className="text-white/40">typical {fmtNum(alert.forecast?.typical, meta.digits)} {meta.unit}</div>
        </div>
      </div>

      <div>
        <div className="text-white/40 text-[10px] uppercase tracking-widest mb-2">Why (SHAP drivers on the hybrid, typical for this hour)</div>
        <div className="space-y-1.5">
          {alert.drivers.map((d) => (
            <div key={d.key} className="flex items-center gap-3 text-[12px]">
              <div className="w-56 text-white/70 truncate">{d.label}</div>
              <div className="flex-1 h-1.5 bg-white/5 rounded overflow-hidden">
                <div className="h-full bg-home-acc/70" style={{ width: `${Math.max(2, d.share * 100)}%` }} />
              </div>
              <div className="w-12 text-right font-mono text-white/50">{pct(d.share)}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="bg-home-acc/5 border border-home-acc/15 rounded-xl p-3">
          <div className="text-home-acc/80 text-[10px] uppercase tracking-widest mb-1">In-app / voice · {PROFILE_LABELS[shownProfile]}</div>
          <p className="text-white/80 text-[13px] leading-relaxed">{message}</p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-xl p-3">
          <div className="text-white/40 text-[10px] uppercase tracking-widest mb-1">SMS · {sms.length} chars</div>
          <p className="text-white/70 text-[13px] font-mono leading-relaxed">{sms}</p>
        </div>
      </div>
    </div>
  );
}

export default function DecisionCenter() {
  const [profile, setProfile] = useState('commuter');
  const [decision, setDecision] = useState<Decision | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    const load = () => backendApi.decision(12)
      .then((r) => { if (mounted) { setDecision(r.data); setError(null); } })
      .catch((e) => { if (mounted) setError(e.message); });
    load();
    const id = setInterval(load, 300000);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  const alertsForProfile = useMemo(
    () => (decision?.alerts || []).filter((a) => a.audience.includes(profile) || a.audience.includes('general')),
    [decision, profile]
  );

  if (error) return <div className="text-red-300 text-sm">Decision engine unavailable: {error}</div>;
  if (!decision) return <div className="text-white/40 text-sm">Loading decision engine…</div>;

  const events = Object.keys(decision.cutoffs);
  const next = decision.hours[0];

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="text-[11px] font-mono uppercase tracking-[0.2em] text-white/40">One validated alert pipeline</div>
          <h2 className="font-display text-3xl font-black text-white">Decision Center</h2>
          <p className="text-white/50 text-sm max-w-2xl mt-2">
            Forecast → 90% range → probability of each event → alert only above a cutoff tuned on held-out data →
            one explanation reused by SMS, in-app, voice and the AI summary.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {PROFILES.map((p) => (
            <button key={p} onClick={() => setProfile(p)}
              className={`px-3 py-1.5 rounded-full text-[11px] font-bold border transition-all ${profile === p ? 'bg-home-acc/15 border-home-acc/40 text-home-acc' : 'border-white/10 text-white/50 hover:text-white'}`}>
              {PROFILE_LABELS[p]}
            </button>
          ))}
        </div>
      </div>

      {/* Next hour, with range and vs typical */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {(Object.keys(TARGET_META) as TargetKey[]).map((t) => {
          const v = next?.values[t];
          const meta = TARGET_META[t];
          if (!v) return null;
          return (
            <div key={t} className="rounded-2xl bg-black/35 border border-white/5 p-4">
              <div className="flex justify-between text-[10px] uppercase tracking-widest text-white/40">
                <span>{meta.label} · {fmtCityHour(next.timestamp)}</span>
                {t === 'aqi' && v.category && <span className="text-home-acc">CPCB {v.category}</span>}
              </div>
              <div className="font-display text-3xl font-black text-white mt-2">{fmtNum(v.value, meta.digits)} <span className="text-sm text-white/40">{meta.unit}</span></div>
              <div className="text-[11px] font-mono text-white/50">90% range {fmtNum(v.low, meta.digits)} – {fmtNum(v.high, meta.digits)}</div>
              <div className="text-[11px] text-white/40">
                {v.vsTypicalPct === null ? '' : `${v.vsTypicalPct > 0 ? '+' : ''}${v.vsTypicalPct.toFixed(0)}% vs a normal ${fmtCityHour(next.timestamp)}`}
              </div>
              {v.liveAdjustment !== 0 && (
                <div className="mt-1 text-[10px] text-sky-300">Adjusted {v.liveAdjustment > 0 ? '+' : ''}{fmtNum(v.liveAdjustment, meta.digits)} from live readings</div>
              )}
            </div>
          );
        })}
      </div>

      {/* Probability strip */}
      <div className="rounded-2xl bg-black/35 border border-white/5 p-5 overflow-x-auto">
        <div className="text-[10px] uppercase tracking-widest text-white/40 mb-3">Event probability, next {decision.horizonHours} h (outlined = alert fires)</div>
        <table className="text-[11px] font-mono">
          <thead>
            <tr>
              <th className="text-left pr-4 text-white/40 font-normal">Event · cutoff</th>
              {decision.hours.map((h) => <th key={h.timestamp} className="px-1 text-white/40 font-normal">{fmtCityHour(h.timestamp)}</th>)}
            </tr>
          </thead>
          <tbody>
            {events.map((ev) => (
              <tr key={ev}>
                <td className="pr-4 py-1 text-white/70 whitespace-nowrap">{EVENT_SHORT[ev] || ev} · {pct(decision.cutoffs[ev])}</td>
                {decision.hours.map((h) => {
                  const p = h.probabilities[ev] || 0;
                  const fires = p >= decision.cutoffs[ev];
                  return (
                    <td key={h.timestamp} className="px-1 py-1">
                      <div title={`${pct(p)}`} className="w-11 h-7 rounded flex items-center justify-center text-white/80"
                        style={{ background: probColor(p), outline: fires ? '2px solid #fff' : 'none' }}>
                        {p >= 0.01 ? Math.round(p * 100) : ''}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Alerts */}
      <div className="space-y-4">
        <div className="text-[10px] uppercase tracking-widest text-white/40">Alerts for {PROFILE_LABELS[profile]}</div>
        {alertsForProfile.length === 0 ? (
          <div className="rounded-2xl border border-home-acc/15 bg-home-acc/5 p-5 text-sm text-white/70">
            No event crosses its alert cutoff for this profile in the next {decision.horizonHours} hours, so nothing is sent.
            Staying quiet is part of the design: the replay shows about 1 false alarm per week.
          </div>
        ) : alertsForProfile.map((a) => <AlertCard key={a.event} alert={a} profile={profile} />)}
      </div>

      <div className="text-[11px] text-white/35 leading-relaxed space-y-1">
        <p>Uncertainty: {decision.uncertainty.note}</p>
        {decision.explanationBasis && <p>Explanations: {decision.explanationBasis}.</p>}
        <p>Thresholds follow CPCB AQI bands and the traffic/grid levels in policy.json; the same policy is scored in the historical replay.</p>
      </div>
    </div>
  );
}
