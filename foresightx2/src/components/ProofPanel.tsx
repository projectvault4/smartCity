import { useEffect, useState } from 'react';
import { backendApi, Proof, fmtNum, pct } from '../services/dataService';

const BASELINES: { key: string; label: string }[] = [
  { key: 'persistence', label: 'Last hour' },
  { key: 'seasonal_naive_24h', label: 'Same hour yesterday' },
  { key: 'sarima', label: 'SARIMA' }
];

function SkillCell({ skill, p }: { skill: number | null; p: number | null }) {
  if (skill === null) return <td className="py-2 px-2 text-right text-white/30">—</td>;
  const good = skill > 0;
  const sig = p !== null && p < 0.05;
  return (
    <td className="py-2 px-2 text-right font-mono">
      <span className={good ? 'text-home-acc' : 'text-red-300'}>{good ? '+' : ''}{(skill * 100).toFixed(0)}%</span>
      <span className="text-white/30 text-[10px] ml-1">{sig ? 'sig.' : 'n.s.'}</span>
    </td>
  );
}

export default function ProofPanel() {
  const [proof, setProof] = useState<Proof | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    backendApi.proof().then((r) => setProof(r.data)).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="text-red-300 text-sm">Evidence unavailable: {error}</div>;
  if (!proof) return <div className="text-white/40 text-sm">Loading evidence…</div>;

  const { benchmark, conformal, anomaly, manifest } = proof;
  const targets = Object.entries(benchmark.targets);

  return (
    <div className="space-y-8">
      <div>
        <div className="text-[11px] font-mono uppercase tracking-[0.2em] text-white/40">Honest evaluation · held-out window {benchmark.window.start.slice(0, 10)} → {benchmark.window.end.slice(0, 10)}</div>
        <h2 className="font-display text-3xl font-black text-white">Model quality, measured</h2>
        <p className="text-white/50 text-sm max-w-3xl mt-2">
          Skill = how much lower the error is than a simple baseline (positive is better). Diebold–Mariano tests with hourly autocorrelation
          say whether the gap is real. "Win rate" is the share of {benchmark.block_hours}-hour blocks where the model beat the baseline.
        </p>
      </div>

      <div className="rounded-2xl bg-black/35 border border-white/5 p-5 overflow-x-auto">
        <table className="w-full text-[12px]">
          <thead>
            <tr className="text-white/40 text-left">
              <th className="py-2 pr-4 font-normal">Target</th>
              <th className="py-2 px-2 font-normal text-right">Hybrid MAE</th>
              {BASELINES.map((b) => <th key={b.key} className="py-2 px-2 font-normal text-right">vs {b.label}</th>)}
              <th className="py-2 px-2 font-normal text-right">Win rate vs SARIMA</th>
              <th className="py-2 pl-2 font-normal text-right">Hours unchanged</th>
            </tr>
          </thead>
          <tbody>
            {targets.map(([key, t]) => (
              <tr key={key} className="border-t border-white/5 text-white/80">
                <td className="py-2 pr-4 font-bold">{t.label}</td>
                <td className="py-2 px-2 text-right font-mono">{fmtNum(t.models['ForeSightX hybrid'].MAE, key === 'temperature' || key === 'aqi' ? 3 : 0)} {t.unit}</td>
                {BASELINES.map((b) => (
                  <SkillCell key={b.key} skill={t.comparisons[b.key]?.skill_mae ?? null} p={t.comparisons[b.key]?.dm_test.p_value ?? null} />
                ))}
                <td className="py-2 px-2 text-right font-mono">{pct(t.comparisons.sarima?.block_win_rate)}</td>
                <td className="py-2 pl-2 text-right font-mono text-white/50">{pct(t.share_hours_unchanged)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-4 space-y-1 text-[12px] text-white/60">
          {benchmark.verdicts.map((v) => <p key={v}>• {v}</p>)}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-2xl bg-home-acc/5 border border-home-acc/15 p-5 text-[13px] text-white/75 space-y-2">
          <div className="text-[10px] uppercase tracking-widest text-home-acc/80">What this means</div>
          <p>
            Traffic and electricity: the hybrid has {pct(benchmark.targets.traffic_flow?.comparisons.persistence.skill_mae)} and{' '}
            {pct(benchmark.targets.electricity_demand?.comparisons.persistence.skill_mae)} lower error than "same as last hour", and is within a few
            percent of SARIMA either way.
          </p>
          <p>
            AQI and temperature were recorded once a day and copied across hours ({pct(benchmark.targets.aqi?.share_hours_unchanged)} and{' '}
            {pct(benchmark.targets.temperature?.share_hours_unchanged)} of hours unchanged). Repeating the last value is nearly perfect there, so we
            do not claim hourly skill for them. At daily resolution (n={benchmark.daily_resolution.aqi?.days} days) the honest comparison is reported below.
          </p>
          <p>
            SHAP shows the network leans on its SARIMA expert input, and the deployed blend gives SARIMA {pct(1 - (manifest.sarima_blend_weight_on_hybrid.aqi ?? 1))} weight
            for AQI/temperature. The value we add is the decision layer: calibrated probabilities, earlier warnings and fewer false alarms.
          </p>
        </div>
        <div className="rounded-2xl bg-black/35 border border-white/5 p-5 text-[12px]">
          <div className="text-[10px] uppercase tracking-widest text-white/40 mb-3">Daily resolution (AQI, temperature)</div>
          <table className="w-full">
            <thead><tr className="text-white/40 text-left"><th className="font-normal py-1">Target</th><th className="font-normal text-right">Model daily MAE</th><th className="font-normal text-right">Last-value daily MAE</th><th className="font-normal text-right">Yesterday's mean MAE</th></tr></thead>
            <tbody>
              {Object.entries(benchmark.daily_resolution).map(([k, d]) => (
                <tr key={k} className="border-t border-white/5 text-white/80">
                  <td className="py-2">{benchmark.targets[k]?.label}</td>
                  <td className="text-right font-mono">{fmtNum(d.model_daily_MAE, 3)}</td>
                  <td className="text-right font-mono">{fmtNum(d.persistence_daily_MAE, 3)}</td>
                  <td className="text-right font-mono">{fmtNum(d.yesterday_mean_MAE, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-2xl bg-black/35 border border-white/5 p-5 text-[12px]">
          <div className="text-[10px] uppercase tracking-widest text-white/40 mb-1">Prediction intervals · target coverage {pct(conformal.coverage_target)}</div>
          <p className="text-white/45 mb-3">Adaptive conformal intervals (only past errors used) vs a fixed-width interval.</p>
          <table className="w-full">
            <thead><tr className="text-white/40 text-left"><th className="font-normal py-1">Target</th><th className="font-normal text-right">Adaptive coverage</th><th className="font-normal text-right">Fixed coverage</th><th className="font-normal text-right">Adaptive width</th></tr></thead>
            <tbody>
              {Object.entries(conformal.targets).map(([k, c]) => (
                <tr key={k} className="border-t border-white/5 text-white/80">
                  <td className="py-2">{benchmark.targets[k]?.label}</td>
                  <td className="text-right font-mono text-home-acc">{pct(c.adaptive.coverage, 1)}</td>
                  <td className="text-right font-mono">{pct(c.static_split.coverage, 1)}</td>
                  <td className="text-right font-mono">{fmtNum(c.adaptive.mean_width, k === 'temperature' || k === 'aqi' ? 2 : 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="rounded-2xl bg-black/35 border border-white/5 p-5 text-[12px]">
          <div className="text-[10px] uppercase tracking-widest text-white/40 mb-1">Anomaly detector · fit on training only</div>
          <p className="text-white/45 mb-3">
            Synthetic incidents injected into the held-out window; how many each detector catches. Threshold: {anomaly.threshold_rule}.
          </p>
          <table className="w-full">
            <thead><tr className="text-white/40 text-left"><th className="font-normal py-1">Injected incident</th><th className="font-normal text-right">Isolation Forest + AE</th><th className="font-normal text-right">z-score baseline</th></tr></thead>
            <tbody>
              {anomaly.injection_tests.map((t) => (
                <tr key={t.id} className="border-t border-white/5 text-white/80">
                  <td className="py-2">{t.label}</td>
                  <td className="text-right font-mono text-home-acc">{t.detected_by_model}/{t.incidents}</td>
                  <td className="text-right font-mono">{t.detected_by_zscore}/{t.incidents}</td>
                </tr>
              ))}
              <tr className="border-t border-white/10 font-bold text-white">
                <td className="py-2">Overall recall</td>
                <td className="text-right font-mono text-home-acc">{pct(anomaly.overall_recall_model)}</td>
                <td className="text-right font-mono">{pct(anomaly.overall_recall_zscore)}</td>
              </tr>
              <tr className="text-white/60">
                <td className="py-1">Flags per week, no incidents</td>
                <td className="text-right font-mono">{anomaly.clean_window.model_flags_per_week.toFixed(1)}</td>
                <td className="text-right font-mono">{anomaly.clean_window.zscore_flags_per_week.toFixed(1)}</td>
              </tr>
            </tbody>
          </table>
          {anomaly.forecast_error_link && anomaly.forecast_error_link.mean_norm_error_flagged !== null && (
            <p className="text-white/55 mt-3">
              Flagged hours carry {(anomaly.forecast_error_link.mean_norm_error_flagged / anomaly.forecast_error_link.mean_norm_error_unflagged).toFixed(1)}×
              the usual forecast error, so a flag is a real "trust the forecast less" signal.
            </p>
          )}
        </div>
      </div>

      <div className="text-[11px] text-white/35">
        Model: {manifest.model}. Built {manifest.generated_at.slice(0, 16).replace('T', ' ')} UTC with torch {manifest.versions.torch}, shap {manifest.versions.shap}.
        Splits: train {manifest.splits.train.start.slice(0, 10)}–{manifest.splits.train.end.slice(0, 10)}, validation {manifest.splits.validation.start.slice(0, 10)}–{manifest.splits.validation.end.slice(0, 10)}, test {manifest.splits.test.start.slice(0, 10)}–{manifest.splits.test.end.slice(0, 10)}.
      </div>
    </div>
  );
}
