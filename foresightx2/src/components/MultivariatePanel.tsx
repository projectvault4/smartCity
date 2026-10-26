import { useEffect, useState } from 'react';
import Card from './Card';
import { backendApi, CityData, MultivariateAnalysis, WhatIf } from '../services/dataService';
import { ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

// No invented fallback series: when the backend is offline the chart is empty.
const fallbackSeries: never[] = [];

const fallbackStats = {
  phaseLagHours: 0,
  phaseLagDirection: 'traffic_leads_aqi',
  phaseLagCorr: 0,
  syncFactor: 0,
  coherence: 0,
  tempEnergyCorr: 0,
  aqiEnergyCorr: 0,
};

const MultivariatePanel = ({ data }: { data: CityData }) => {
  const [analysis, setAnalysis] = useState<MultivariateAnalysis | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [associations, setAssociations] = useState<WhatIf['daily_associations'] | null>(null);

  useEffect(() => {
    let mounted = true;
    backendApi.modelMultivariate(720)
      .then((response) => { if (mounted) setAnalysis(response.data); })
      .catch(() => {})
      .finally(() => { if (mounted) setLoaded(true); });
    backendApi.whatIf()
      .then((response) => { if (mounted) setAssociations(response.data.daily_associations); })
      .catch(() => {});
    return () => { mounted = false; };
  }, []);

  const stats = analysis?.stats || fallbackStats;
  const syncData = analysis?.series?.map((p: any) => ({
    time: p.time,
    traffic: p.traffic,
    aqi: p.aqi + 100,
    energy: p.energy + 200,
    trafficNorm: p.traffic,
    aqiNorm: p.aqi,
    energyNorm: p.energy,
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
    : 'Model offline.';

  const syncFactorText = analysis
    ? `Correlation of ${stats.syncFactor} between energy demand and traffic flow, computed from the trained-model multivariate series.`
    : 'Model offline.';

  const coherenceText = analysis
    ? `Environmental metrics show ${Math.round(stats.coherence * 100)}% coherence with urban mobility patterns in the trained-model window.`
    : 'Model offline.';

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
                          { key: 'traffic', label: 'Traffic Flow', color: '#f39c12', raw: d.trafficRaw != null ? `${d.trafficRaw} veh/hr` : d.traffic, norm: d.trafficNorm },
                          { key: 'aqi', label: 'Atmospheric AQI', color: '#3498db', raw: d.aqiRaw != null ? `${d.aqiRaw} AQI` : d.aqi, norm: d.aqiNorm },
                          { key: 'energy', label: 'Grid Load', color: '#2ecc71', raw: d.energyRaw != null ? `${d.energyRaw} MW` : d.energy, norm: d.energyNorm },
                        ].map((s) => (
                          <div key={s.key} className="text-[11px] font-bold" style={{ color: s.color }}>
                            {s.label}: {s.raw} <span className="text-[10px] text-white/40">({s.norm}/100 norm)</span>
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

      <Card title="Cross-domain links measured in observed data" theme="home">
        <p className="text-xs text-white/45 mb-4">
          The curves above come from the 2026 outlook, whose hourly AQI and energy shapes are partly derived from traffic and temperature during
          post-processing, so their correlations overstate coupling. The links below are measured on observed daily Bengaluru data (Feb–Jun 2025)
          with 95% bootstrap intervals.
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {(associations || []).map((a) => (
            <div key={a.x + a.y} className="p-4 bg-white/5 rounded-2xl border border-white/5">
              <div className="flex items-center justify-between mb-1">
                <div className="text-[10px] font-black uppercase tracking-widest text-white/50">{a.x.replace('_flow', '').replace('_', ' ')} → {a.y.replace('_', ' ')}</div>
                <span className={`text-[10px] px-2 py-0.5 rounded border ${a.significant ? 'text-home-acc border-home-acc/40' : 'text-white/40 border-white/15'}`}>{a.significant ? 'supported' : 'not supported'}</span>
              </div>
              <p className="text-xs text-white/65 leading-normal">{a.sentence}</p>
            </div>
          ))}
          {!associations && <p className="text-xs text-white/40">Loading measured links…</p>}
        </div>
      </Card>
    </div>
  );
};

export default MultivariatePanel;
