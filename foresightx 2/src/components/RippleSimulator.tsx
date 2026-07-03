import { useState, useMemo } from 'react';
import Card from './Card';
import { AreaChart, Area, ResponsiveContainer, XAxis, YAxis } from 'recharts';

const RippleSimulator = () => {
  const [congestion, setCongestion] = useState(50);

  // Derive correlations: Higher traffic -> Higher AQI, Higher Electricity (signal controls, ventilation)
  const chartData = useMemo(() => {
    return Array.from({ length: 12 }, (_, i) => {
      const baseTraffic = 40 + Math.sin(i / 2) * 20;
      const t = Math.max(0, baseTraffic + (congestion - 50));
      return {
        time: i + ':00',
        traffic: t,
        // AQI heavily linked to traffic
        aqi: 20 + t * 1.5 + (congestion / 10),
        // Energy slightly linked to traffic (urban management)
        energy: 100 + t * 0.5 + (congestion / 5)
      };
    });
  }, [congestion]);

  return (
    <Card title="Cross-Domain Ripple Simulator" theme="traffic">
      <div className="p-2 space-y-8">
        <p className="text-sm text-white/50">Drag the slider to see how traffic congestion ripples through air quality and energy demand.</p>
        
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
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
           <div className="space-y-3">
              <div className="text-[10px] font-bold text-air-acc uppercase tracking-widest">Predicted AQI Curve</div>
              <div className="h-[150px] w-full">
                 <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                       <Area type="monotone" dataKey="aqi" stroke="#3498db" fill="#3498db20" strokeWidth={3} isAnimationActive={false} />
                    </AreaChart>
                 </ResponsiveContainer>
              </div>
              <div className="text-[10px] text-white/30 text-center">Simulated impact: {congestion > 70 ? 'CRITICAL POLLUTION' : congestion > 40 ? 'MODERATE IMPACT' : 'CLEAN AIR FLOW'}</div>
           </div>

           <div className="space-y-3">
              <div className="text-[10px] font-bold text-eng-acc uppercase tracking-widest">Electricity Demand Shift</div>
              <div className="h-[150px] w-full">
                 <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                       <Area type="monotone" dataKey="energy" stroke="#2ecc71" fill="#2ecc7120" strokeWidth={3} isAnimationActive={false} />
                    </AreaChart>
                 </ResponsiveContainer>
              </div>
              <div className="text-[10px] text-white/30 text-center">Load correlation: {congestion}% congestion → {Math.round(100 + congestion / 2)}MW baseline</div>
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
