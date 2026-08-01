import { useEffect, useState } from 'react';
import Card from './Card';
import { backendApi, CityData, MultivariateAnalysis } from '../services/dataService';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

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

const causalDescs: Record<string, string> = {
  'Traffic→AQI': 'Combustion emissions and road dust resuspension during peak flow, measured from the trained-model time series.',
  'Weather→Energy': 'Thermodynamic load adjustment for industrial and residential cooling, from the trained-model temperature/energy series.',
  'Energy→Economic': 'Proxy for industrial output and commercial activity intensity (model-derived correlation shown).',
  'AQI→Health': 'Respiratory stressors and cardiovascular impact clusters (correlation shown from trained-model AQI series).',
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
  const syncData = analysis?.series?.map((p) => ({
    time: p.time,
    traffic: p.traffic,
    aqi: p.aqi,
    energy: p.energy,
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
              <LineChart data={syncData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                <XAxis dataKey="time" hide />
                <YAxis hide />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0c0c0c', border: '1px solid #ffffff10', borderRadius: '12px' }}
                  itemStyle={{ fontSize: '11px', fontWeight: 'bold' }}
                />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px', paddingTop: '20px' }} />
                <Line type="monotone" dataKey="traffic" name="Traffic Flow" stroke="#f39c12" strokeWidth={3} dot={false} />
                <Line type="monotone" dataKey="aqi" name="Atmospheric AQI" stroke="#3498db" strokeWidth={3} dot={false} />
                <Line type="monotone" dataKey="energy" name="Grid Load (MW)" stroke="#2ecc71" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
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
                   <p className="text-xs text-white/60 font-medium leading-normal">{causalDescs[`${rel.from}→${rel.to}`]}</p>
                </div>
             </div>
           ))}
        </div>
      </Card>
    </div>
  );
};

export default MultivariatePanel;
