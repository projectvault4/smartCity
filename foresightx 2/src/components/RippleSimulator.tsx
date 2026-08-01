import { useState, useMemo } from 'react';
import Card from './Card';
import { AreaChart, Area, ResponsiveContainer, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { ForecastPoint } from '../services/dataService';

// Traffic congestion drives AQI and grid load up — that's the ripple effect.
// Baseline comes from the model's predicted series; the slider adds the
// congestion shock on top.

function buildSeries(forecast?: ForecastPoint[] | null) {
  if (forecast && forecast.length > 0) {
    const sorted = [...forecast].sort((a, b) => a.stepAhead - b.stepAhead);
    return Array.from({ length: 12 }, (_, i) => {
      const p = sorted[i % sorted.length];
      return {
        aqi: p.aqi ?? 55,
        energy: p.electricityDemand ?? 4000,
        traffic: p.trafficFlow ?? 6000,
      };
    });
  }
  // Fallback when the model forecast isn't available
  return Array.from({ length: 12 }, (_, i) => {
    const wave = Math.sin(i / 2) * 8;
    return { aqi: 55 + wave, energy: 4000 + wave * 20, traffic: 6000 + wave * 100 };
  });
}

const RippleSimulator = ({ forecast }: { forecast?: ForecastPoint[] | null }) => {
  const [congestion, setCongestion] = useState(50);

  const base = useMemo(() => buildSeries(forecast), [forecast]);

  // AQI and energy rise with congestion. congFactor goes 0.6x (0%) -> 1.6x (100%).
  const congFactor = 0.6 + (congestion / 100) * 1.0;

  const chartData = useMemo(() => {
    return base.map((p, i) => ({
      time: i + ':00',
      aqi: Math.round(p.aqi * congFactor),
      energy: Math.round(p.energy * congFactor),
      traffic: Math.round(p.traffic),
    }));
  }, [base, congFactor]);

  // Fixed Y domains so the curve visibly rises as congestion increases
  const aqiMax = Math.ceil(Math.max(...base.map((p) => p.aqi)) * 1.7);
  const energyMax = Math.ceil(Math.max(...base.map((p) => p.energy)) * 1.7);

  return (
    <Card title="Cross-Domain Ripple Simulator" theme="traffic">
      <div className="p-2 space-y-8">
        <p className="text-sm text-white/50">
          Drag the slider to see how traffic congestion ripples through air quality and energy demand.
        </p>

        <div className="space-y-4 px-4 py-6 bg-white/5 rounded-2xl border border-white/5">
          <div className="flex justify-between items-center mb-4">
            <span className="text-xs font-bold text-traf-acc uppercase tracking-widest">Traffic Congestion %</span>
            <span className="text-2xl font-display font-extrabold text-traf-acc">{congestion}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            value={congestion}
            onChange={(e) => setCongestion(parseInt(e.target.value))}
            className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer accent-traf-acc"
          />
          <div className="text-[10px] text-white/30">
            {forecast && forecast.length > 0
              ? 'Baseline: model predicted values · AQI scales with congestion'
              : 'Model offline — showing synthetic baseline'}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-3">
            <div className="text-[10px] font-bold text-air-acc uppercase tracking-widest">
              Predicted AQI Curve <span className="text-white/30 normal-case">(peak {chartData[chartData.length - 1]?.aqi ?? 0})</span>
            </div>
            <div className="h-[150px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="aqiFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#3498db" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#3498db" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#ffffff10" vertical={false} />
                  <XAxis dataKey="time" tick={{ fill: '#ffffff55', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, aqiMax]} tick={{ fill: '#ffffff55', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: '#0d1a10', border: '1px solid #ffffff22', fontSize: 11 }} />
                  <Area type="monotone" dataKey="aqi" stroke="#3498db" fill="url(#aqiFill)" strokeWidth={3} isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div className="text-[10px] text-white/30 text-center">
              Simulated impact: {congestion > 70 ? 'CRITICAL POLLUTION' : congestion > 40 ? 'MODERATE IMPACT' : 'CLEAN AIR FLOW'}
            </div>
          </div>

          <div className="space-y-3">
            <div className="text-[10px] font-bold text-eng-acc uppercase tracking-widest">
              Electricity Demand Shift <span className="text-white/30 normal-case">(peak {chartData[chartData.length - 1]?.energy ?? 0} MW)</span>
            </div>
            <div className="h-[150px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="energyFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#2ecc71" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#2ecc71" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#ffffff10" vertical={false} />
                  <XAxis dataKey="time" tick={{ fill: '#ffffff55', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, energyMax]} tick={{ fill: '#ffffff55', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: '#0d1a10', border: '1px solid #ffffff22', fontSize: 11 }} />
                  <Area type="monotone" dataKey="energy" stroke="#2ecc71" fill="url(#energyFill)" strokeWidth={3} isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div className="text-[10px] text-white/30 text-center">
              Load correlation: {congestion}% congestion → {Math.round((chartData[0]?.energy ?? 0) * congFactor)}MW baseline
            </div>
          </div>
        </div>

        <div className="p-4 bg-white/5 rounded-xl border border-white/5 flex items-center justify-center gap-4">
          <div className="text-[11px] font-medium text-white/40">Real-time Cross-Domain Link:</div>
          <div className="flex gap-2">
            <div className={`px-2 py-1 rounded text-[10px] font-bold ${congestion > 80 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>CO2 SYNC</div>
            <div className={`px-2 py-1 rounded text-[10px] font-bold ${congestion > 60 ? 'bg-orange-500/20 text-orange-400' : 'bg-blue-500/20 text-blue-400'}`}>GRID LOAD</div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default RippleSimulator;
