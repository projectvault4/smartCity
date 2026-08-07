import { useEffect, useState } from 'react';
import Card from './Card';
import { backendApi, CityData, MultivariateAnalysis } from '../services/dataService';
import { ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const fallbackSeries = Array.from({ length: 24 }).map((_, i) => ({
  time: `${i}:00`,
  traffic: Math.sin(i / 3) * 30 + 60 + Math.random() * 5,
  aqi: Math.sin((i - 1) / 3) * 20 + 50 + Math.random() * 10,
  energy: Math.sin((i - 3) / 3) * 40 + 70 + Math.random() * 5,
}));

const fallbackStats = {
  phaseLagHours: 1.2,
  phaseLagDirection: 'traffic_leads_aqi',
  phaseLagCorr: 0.84,
  syncFactor: 0.84,
  coherence: 0.92,
  tempEnergyCorr: 0.7,
  aqiEnergyCorr: 0.5,
};

const impactLabel = (from: string, to: string, s: MultivariateAnalysis['stats']) => {
  if (from === 'Traffic' && to === 'AQI') return `Direct · ρ=${s.phaseLagCorr}`;
  if (from === 'Weather' && to === 'Energy') return `Lagged · ρ=${s.tempEnergyCorr}`;
  if (from === 'Energy' && to === 'Economic') return `Strong · ρ=${s.aqiEnergyCorr}`;
  if (from === 'AQI' && to === 'Health') return `Terminal · ρ=${s.coherence}`;
  return 'Observed';
};

const strengthWord = (r: number) =>
  r >= 0.9 ? 'very strong' : r >= 0.7 ? 'strong' : r >= 0.5 ? 'moderate' : r >= 0.3 ? 'weak' : 'very weak';

const causalDescs = (from: string, to: string, s: MultivariateAnalysis['stats']): string => {
  const r = (from === 'Traffic' && to === 'AQI')
    ? s.phaseLagCorr
    : (from === 'Weather' && to === 'Energy')
      ? s.tempEnergyCorr
      : (from === 'Energy' && to === 'Economic')
        ? s.aqiEnergyCorr
        : s.coherence;
  const strength = strengthWord(r);

  switch (from + '→' + to) {
    case 'Traffic→AQI':
      return `A ${strength} correlation (ρ=${r}) between vehicle flow and air quality, measured from the trained-model time series — the model links more traffic to more pollution in the air.`;
    case 'Weather→Energy':
      return `A ${strength} correlation (ρ=${r}) between temperature and electricity use, from the trained-model series — the model ties warmer weather to more cooling demand.`;
    case 'Energy→Economic':
      return `A ${strength} correlation (ρ=${r}) in the trained-model series — used as a proxy for industrial output and commercial activity, since direct economic data is not part of the model.`;
    case 'AQI→Health':
      return `A ${strength} correlation (ρ=${r}) in the trained-model AQI series — a proxy for respiratory and cardiovascular stress, since direct health records are not part of the model.`;
    default:
      return 'Observed relationship in the trained-model series.';
  }
};

const MultivariatePanel = ({ data }: { data: CityData }) => {
  const [analysis, setAnalysis] = useState<MultivariateAnalysis | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let mounted = true;
    backendApi.modelMultivariate(720)
      .then((response) => { if (mounted) setAnalysis(response.data); })
      .catch(() => {})
      .finally(() => { if (mounted) setLoaded(true); });
    return () => { mounted = false; };
  }, []);

  const stats = analysis?.stats || fallbackStats;
  const syncData = analysis?.series?.map((p: any) => ({
    time: p.time,
    traffic: p.traffic,
    aqi: p.aqi,
    energy: p.energy,
    trafficRaw: p.trafficRaw ?? p.traffic,
    aqiRaw: p.aqiRaw ?? p.aqi,
    energyRaw: p.energyRaw ?? p.energy,
  })) || fallbackSeries;

  const phaseLagText = analysis
    ? stats.phaseLagHours === 0
      ? `Traffic and AQI cycles move together (0h lag), with a correlation of ${stats.phaseLagCorr} measured across the trained-model window.`
      : stats.phaseLagDirection === 'traffic_leads_aqi'
        ? `Traffic congestion cycles consistently precede AQI shifts by ~${Math.round(stats.phaseLagHours * 60)} minutes.`
        : `AQI shifts lead traffic cycles by ~${Math.round(stats.phaseLagHours * 60)} minutes in the trained-model series.`
    : 'Traffic congestion cycles consistently precede AQI shifts by ~72 minutes.';

  const syncFactorText = analysis
    ? `Correlation of ${stats.syncFactor} between energy demand and traffic flow, computed from the trained-model multivariate series.`
    : 'High correlation between industrial energy surges and heavy logistics vehicle movement.';

  const coherenceText = analysis
    ? `Environmental metrics show ${Math.round(stats.coherence * 100)}% coherence with urban mobility patterns in the trained-model window.`
    : 'Environmental metrics show 92% seasonal coherence with urban mobility patterns.';

  return (
    <div className="space-y-6">
      <Card title="Domain Synchronization & Causal Ripple" theme="energy">
        <div className="p-2 space-y-6">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <p className="text-[11px] text-white/50 mb-2 font-medium uppercase tracking-widest">Temporal Lag Analysis (24H Observation)</p>
            {analysis ? (
              <p className="text-[10px] font-mono text-white/40">MODEL · {analysis.window.from} → {analysis.window.to}</p>
            ) : loaded ? (
              <p className="text-[10px] font-mono text-white/40">FALLBACK (model offline)</p>
            ) : null}
          </div>

          <div className="h-[320px] w-full bg-black/20 rounded-2xl p-4 border border-white/5">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={syncData} margin={{ top: 5, right: 40, bottom: 0, left: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                {[
                  { key: 'traffic', label: 'Traffic Flow', color: '#f39c12', band: 0 },
                  { key: 'aqi', label: 'Atmospheric AQI', color: '#3498db', band: 100 },
                  { key: 'energy', label: 'Grid Load (MW)', color: '#2ecc71', band: 200 },
                ].map((s) => (
                  <YAxis key={s.key} yAxisId={s.key} hide domain={[s.band, s.band + 100]} />
                ))}
                <XAxis dataKey="time" hide />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (!active || !payload || !payload.length) return null;
                    const d = payload[0].payload;
                    return (
                      <div className="bg-[#0c0c0c] border border-white/10 rounded-xl p-3 shadow-xl space-y-1">
                        <div className="text-[11px] font-bold text-white/50 mb-1">Time: {label}</div>
                        {[
                          { key: 'traffic', label: 'Traffic Flow', color: '#f39c12', raw: d.trafficRaw != null ? `${d.trafficRaw} veh/hr` : d.traffic },
                          { key: 'aqi', label: 'Atmospheric AQI', color: '#3498db', raw: d.aqiRaw != null ? `${d.aqiRaw} AQI` : d.aqi },
                          { key: 'energy', label: 'Grid Load', color: '#2ecc71', raw: d.energyRaw != null ? `${d.energyRaw} MW` : d.energy },
                        ].map((s) => (
                          <div key={s.key} className="text-[11px] font-bold" style={{ color: s.color }}>
                            {s.label}: {s.raw} <span className="text-[10px] text-white/40">({d[s.key]}/100 norm)</span>
                          </div>
                        ))}
                      </div>
                    );
                  }}
                />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px', paddingTop: '20px' }} />
                <Line type="monotone" yAxisId="traffic" dataKey="traffic" name="Traffic Flow" stroke="#f39c12" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                <Line type="monotone" yAxisId="aqi" dataKey="aqi" name="Atmospheric AQI" stroke="#3498db" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                <Line type="monotone" yAxisId="energy" dataKey="energy" name="Grid Load (MW)" stroke="#2ecc71" strokeWidth={2.5} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="text-[10px] text-white/40 px-2 italic">
            * Note: All 3 series are normalized to a 0–100 scale to compare phase synchronization and temporal lag across disparate units (veh/hr, AQI, MW). Hover over the chart to view exact raw values.
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4">
             <div className="space-y-1">
                <div className="text-[10px] font-black text-traf-acc uppercase tracking-widest">Phase Lag: {stats.phaseLagHours === 0 ? '0H (SYNC)' : `${stats.phaseLagHours}H`}</div>
                <p className="text-[11px] text-white/40 leading-relaxed">{phaseLagText}</p>
             </div>
             <div className="space-y-1">
                <div className="text-[10px] font-black text-eng-acc uppercase tracking-widest">Sync Factor: {stats.syncFactor.toFixed(2)}</div>
                <p className="text-[11px] text-white/40 leading-relaxed">{syncFactorText}</p>
             </div>
             <div className="space-y-1">
                <div className="text-[10px] font-black text-air-acc uppercase tracking-widest">Coherence: {stats.coherence >= 0.7 ? 'Strong' : stats.coherence >= 0.4 ? 'Moderate' : 'Weak'}</div>
                <p className="text-[11px] text-white/40 leading-relaxed">{coherenceText}</p>
             </div>
          </div>
        </div>
      </Card>

      <Card title="Cross-Domain Causal Mapping" theme="home">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
           {[
             { from: 'Traffic', to: 'AQI', impact: 'Direct' },
             { from: 'Weather', to: 'Energy', impact: 'Lagged' },
             { from: 'Energy', to: 'Economic', impact: 'Strong' },
             { from: 'AQI', to: 'Health', impact: 'Terminal' }
           ].map((rel, i) => (
             <div key={i} className="flex items-center gap-4 p-4 bg-white/5 rounded-2xl border border-white/5 hover:border-white/15 transition-all group">
                <div className="flex flex-col items-center gap-1 min-w-[60px]">
                   <div className="text-[10px] font-black text-white/30 uppercase">{rel.from}</div>
                   <div className="h-4 w-[1px] bg-white/10" />
                   <div className="text-[10px] font-black text-home-acc uppercase">{rel.to}</div>
                </div>
                <div className="flex-1">
                   <div className="text-[10px] font-black text-white/20 uppercase tracking-widest mb-1">Impact: {impactLabel(rel.from, rel.to, stats)}</div>
                   <p className="text-xs text-white/60 font-medium leading-normal">{causalDescs(rel.from, rel.to, stats)}</p>
                </div>
             </div>
           ))}
        </div>
      </Card>
    </div>
  );
};

export default MultivariatePanel;
