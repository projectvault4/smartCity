import { useCallback, useEffect, useState } from 'react';
import { motion } from 'motion/react';
import {
  backendApi, CityData, Decision, ForecastPoint, ShapSummary, TARGET_META, TargetKey, WhatIf,
  fmtCityHour, fmtNum, pct
} from '../services/dataService';

// Explainable AI, grounded in measurements:
//   * SHAP (expected gradients) computed on the deployed TFT+GRU hybrid network
//   * the real composition of the deployed forecast (neural part vs SARIMA blend)
//   * cross-domain links only where the observed data supports them (with CIs)
// No hand-written causal stories.

const TARGETS = Object.keys(TARGET_META) as TargetKey[];

function Bar({ share, color }: { share: number; color: string }) {
  return (
    <div className="flex-1 h-1.5 rounded overflow-hidden" style={{ background: '#142720' }}>
      <div className="h-full rounded" style={{ width: `${Math.max(1.5, share * 100)}%`, background: color }} />
    </div>
  );
}

export default function XaiPanel(_props: { data?: CityData; forecast?: ForecastPoint[] | null }) {
  const [shap, setShap] = useState<ShapSummary | null>(null);
  const [whatIf, setWhatIf] = useState<WhatIf | null>(null);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [plain, setPlain] = useState<{ text: string; source: string } | null>(null);
  const [plainLoading, setPlainLoading] = useState(false);

  useEffect(() => {
    Promise.all([backendApi.explain(), backendApi.whatIf(), backendApi.decision(6)])
      .then(([s, w, d]) => { setShap(s.data); setWhatIf(w.data); setDecision(d.data); })
      .catch((e) => setError(e.message));
  }, []);

  const generatePlain = useCallback(async () => {
    setPlainLoading(true);
    try {
      const r = await backendApi.summary(['general']);
      setPlain({ text: r.data.text, source: r.data.source });
    } catch (e) {
      setPlain({ text: `Could not build the summary: ${(e as Error).message}`, source: 'error' });
    } finally {
      setPlainLoading(false);
    }
  }, []);

  if (error) return <div className="text-red-300 text-sm">Explanations unavailable: {error}</div>;
  if (!shap || !whatIf || !decision) return <div className="text-white/40 text-sm">Loading SHAP explanations…</div>;

  const next = decision.hours[0];

  return (
    <div style={{ fontFamily: "'Inter', sans-serif", color: '#e9f3ee' }} className="space-y-6">
      <div>
        <h1 className="font-['Fraunces',serif] font-light italic leading-[1.15] mb-2" style={{ fontSize: 'clamp(1.5rem,3.5vw,2.1rem)' }}>
          Explainable AI (XAI)
        </h1>
        <p className="text-[0.875rem] max-w-[70ch] leading-relaxed" style={{ color: '#8fa69b' }}>
          SHAP values (GradientExplainer, expected gradients) computed on the real TFT + GRU hybrid network, plus what the deployed forecast is actually made of.
          Cross-domain claims are shown only where the observed data supports them.
        </p>
      </div>

      {/* 1. Composition of the deployed forecast */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="rounded-[16px] border p-6" style={{ background: '#10201a', borderColor: '#1f3831' }}>
        <div className="text-[0.65rem] font-mono uppercase tracking-[0.16em] mb-1" style={{ color: '#5c7269' }}>What the deployed forecast is made of</div>
        <p className="text-[0.8rem] mb-4" style={{ color: '#8fa69b' }}>Validation-selected blend weights. For AQI and temperature the statistical SARIMA part dominates, because those signals change once a day.</p>
        <div className="space-y-3">
          {TARGETS.map((t) => {
            const mix = shap.targets[t]?.final_mix;
            if (!mix) return null;
            const parts = [
              { label: 'Neural hybrid', v: mix.neural_hybrid, c: '#6fe7b7' },
              { label: 'SARIMA', v: mix.sarima, c: '#b48eea' },
              { label: 'Last reading', v: mix.last_observation, c: '#f0a857' }
            ];
            const total = parts.reduce((s, p) => s + Math.max(0, p.v), 0) || 1;
            return (
              <div key={t} className="flex items-center gap-4 text-[0.78rem]">
                <div className="w-28 text-white/80">{TARGET_META[t].label}</div>
                <div className="flex-1 flex h-4 rounded overflow-hidden" style={{ background: '#142720' }}>
                  {parts.map((p) => (
                    <div key={p.label} title={`${p.label} ${pct(p.v / total)}`} style={{ width: `${(Math.max(0, p.v) / total) * 100}%`, background: p.c }} />
                  ))}
                </div>
                <div className="w-56 font-mono text-[0.7rem] text-white/50">
                  {parts.filter((p) => p.v > 0.001).map((p) => `${p.label.split(' ')[0]} ${pct(p.v / total)}`).join(' · ')}
                </div>
              </div>
            );
          })}
        </div>
      </motion.div>

      {/* 2. Global SHAP drivers per target */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {TARGETS.map((t) => {
          const info = shap.targets[t];
          if (!info) return null;
          const meta = TARGET_META[t];
          return (
            <motion.div key={t} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="rounded-[16px] border p-5" style={{ background: '#10201a', borderColor: '#1f3831' }}>
              <div className="flex items-center gap-2 mb-3">
                <span className="w-[7px] h-[7px] rounded-full" style={{ background: meta.color }} />
                <span className="text-[0.8rem] font-semibold">{meta.label}: what the network relies on</span>
              </div>
              <div className="space-y-2">
                {info.global_drivers.slice(0, 5).map((d) => (
                  <div key={d.key} className="flex items-center gap-3 text-[0.75rem]">
                    <div className="w-60 truncate" style={{ color: '#8fa69b' }}>{d.label}</div>
                    <Bar share={d.share} color={meta.color} />
                    <div className="w-12 text-right font-mono text-white/60">{pct(d.share)}</div>
                  </div>
                ))}
              </div>
              <div className="mt-3 text-[0.68rem] font-mono" style={{ color: '#5c7269' }}>
                Time context: {info.time_context.slice(0, 3).map((c) => `${c.label} ${pct(c.share)}`).join(' · ')}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* 3. Right now */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="rounded-[16px] border p-6" style={{ background: '#10201a', borderColor: '#1f3831' }}>
        <div className="text-[0.65rem] font-mono uppercase tracking-[0.16em] mb-1" style={{ color: '#5c7269' }}>Forecast for {fmtCityHour(next.timestamp)}, and why</div>
        <p className="text-[0.8rem] mb-4" style={{ color: '#8fa69b' }}>Drivers are the average SHAP effect at this hour of day on held-out data, in forecast units, relative to a typical training hour.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {TARGETS.map((t) => {
            const v = next.values[t];
            if (!v) return null;
            const meta = TARGET_META[t];
            return (
              <div key={t} className="rounded-[12px] border p-4" style={{ background: '#0e1a16', borderColor: '#1f3831' }}>
                <div className="flex justify-between items-baseline">
                  <span className="text-[0.78rem] font-semibold">{meta.label}</span>
                  <span className="font-mono text-[0.9rem]">{fmtNum(v.value, meta.digits)} {meta.unit}</span>
                </div>
                <div className="text-[0.68rem] font-mono mb-2" style={{ color: '#5c7269' }}>
                  90% range {fmtNum(v.low, meta.digits)}–{fmtNum(v.high, meta.digits)}
                  {v.vsTypicalPct !== null ? ` · ${v.vsTypicalPct > 0 ? '+' : ''}${v.vsTypicalPct.toFixed(0)}% vs normal` : ''}
                </div>
                <ul className="space-y-1">
                  {v.drivers.map((d) => (
                    <li key={d.key} className="text-[0.75rem] flex justify-between gap-3" style={{ color: '#8fa69b' }}>
                      <span>{d.direction === 'up' ? '↑' : '↓'} {d.label}</span>
                      <span className="font-mono">{d.effect > 0 ? '+' : ''}{fmtNum(d.effect, meta.digits)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </div>
      </motion.div>

      {/* 4. Cross-domain links supported by data */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="rounded-[16px] border p-6" style={{ background: '#10201a', borderColor: '#1f3831' }}>
        <div className="text-[0.65rem] font-mono uppercase tracking-[0.16em] mb-1" style={{ color: '#5c7269' }}>How the domains connect (evidence, not assumptions)</div>
        <div className="space-y-3 mt-3">
          {whatIf.daily_associations.map((a) => (
            <div key={a.x + a.y} className="flex gap-3 items-start p-3 rounded-[12px] border" style={{ background: '#0e1a16', borderColor: '#1f3831' }}>
              <span className={`text-[0.62rem] font-mono px-2 py-0.5 rounded-full border shrink-0 ${a.significant ? 'text-[#6fe7b7] border-[#6fe7b7]/40' : 'text-white/40 border-white/15'}`}>
                {a.significant ? 'supported' : 'not supported'}
              </span>
              <p className="text-[0.78rem] leading-relaxed" style={{ color: '#8fa69b' }}>{a.sentence}</p>
            </div>
          ))}
          {whatIf.findings.slice(0, 2).map((f) => (
            <p key={f} className="text-[0.75rem] leading-relaxed" style={{ color: '#5c7269' }}>• {f}</p>
          ))}
        </div>
      </motion.div>

      {/* 5. Plain words, grounded in the same evidence */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="rounded-[16px] border p-6" style={{ background: '#10201a', borderColor: '#1f3831' }}>
        <div className="flex items-center justify-between gap-4 mb-4">
          <div>
            <div className="text-[0.65rem] font-mono uppercase tracking-[0.16em] mb-1" style={{ color: '#5c7269' }}>In plain words</div>
            <p className="text-[0.8rem]" style={{ color: '#8fa69b' }}>Built from the same numbers, ranges and drivers as the alerts; the language model may only rephrase them.</p>
          </div>
          <button onClick={generatePlain} disabled={plainLoading} className="px-4 py-2 rounded-[10px] text-[0.75rem] font-semibold disabled:opacity-50" style={{ background: 'linear-gradient(90deg, #0f766e, #2ecc71)', color: '#e9f3ee' }}>
            {plainLoading ? 'Explaining…' : plain ? 'Regenerate' : 'Explain in plain words'}
          </button>
        </div>
        {plain ? (
          <>
            <p className="text-[0.9rem] leading-relaxed whitespace-pre-line">{plain.text}</p>
            <p className="text-[0.65rem] font-mono mt-2" style={{ color: '#5c7269' }}>source: {plain.source === 'gemini_grounded' ? 'Gemini, rephrasing the evidence' : 'evidence template'}</p>
          </>
        ) : (
          <p className="text-[0.82rem]" style={{ color: '#5c7269' }}>Press the button for a short citizen-friendly summary.</p>
        )}
      </motion.div>

      <p className="text-[0.68rem] font-mono" style={{ color: '#5c7269' }}>
        SHAP check: the attributions add up to the network output within a mean gap of {fmtNum(shap.targets.traffic_flow?.additivity_check.mean_abs_gap, 0)} veh/hr
        (network output spread {fmtNum(shap.targets.traffic_flow?.additivity_check.network_output_std, 0)}); expected-gradients is sampled, so small gaps are normal.
      </p>
    </div>
  );
}
