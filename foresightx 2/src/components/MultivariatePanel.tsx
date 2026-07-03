import Card from './Card';
import { CityData } from '../services/dataService';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const MultivariatePanel = ({ data }: { data: CityData }) => {
  // Simulate time-synced data for the 4 domains to show correlation/latency
  const syncData = Array.from({ length: 24 }).map((_, i) => ({
    time: `${i}:00`,
    traffic: Math.sin(i / 3) * 30 + 60 + Math.random() * 5,
    aqi: Math.sin((i - 1) / 3) * 20 + 50 + Math.random() * 10, // 1h lag
    energy: Math.sin((i - 3) / 3) * 40 + 70 + Math.random() * 5, // 3h lag
  }));

  return (
    <div className="space-y-6">
      <Card title="Domain Synchronization & Causal Ripple" theme="energy">
        <div className="p-2 space-y-6">
          <p className="text-[11px] text-white/50 mb-2 font-medium uppercase tracking-widest">Temporal Lag Analysis (24H Observation)</p>
          
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
                <div className="text-[10px] font-black text-traf-acc uppercase tracking-widest">Phase Lag: 1.2H</div>
                <p className="text-[11px] text-white/40 leading-relaxed">Traffic congestion cycles consistently precede AQI shifts by ~72 minutes.</p>
             </div>
             <div className="space-y-1">
                <div className="text-[10px] font-black text-eng-acc uppercase tracking-widest">Sync Factor: 0.84</div>
                <p className="text-[11px] text-white/40 leading-relaxed">High correlation between industrial energy surges and heavy logistics vehicle movement.</p>
             </div>
             <div className="space-y-1">
                <div className="text-[10px] font-black text-air-acc uppercase tracking-widest">Coherence: Strong</div>
                <p className="text-[11px] text-white/40 leading-relaxed">Environmental metrics show 92% seasonal coherence with urban mobility patterns.</p>
             </div>
          </div>
        </div>
      </Card>

      <Card title="Cross-Domain Causal Mapping" theme="home">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
           {[
             { from: 'Traffic', to: 'AQI', desc: 'Combustion emissions and road dust resuspension during peak flow.', impact: 'Direct' },
             { from: 'Weather', to: 'Energy', desc: 'Thermodynamic load adjustment for industrial and residential cooling.', impact: 'Lagged' },
             { from: 'Energy', to: 'Economic', desc: 'Proxy for industrial output and commercial activity intensity.', impact: 'Strong' },
             { from: 'AQI', to: 'Health', desc: 'Respiratory stressors and cardiovascular impact clusters.', impact: 'Terminal' }
           ].map((rel, i) => (
             <div key={i} className="flex items-center gap-4 p-4 bg-white/5 rounded-2xl border border-white/5 hover:border-white/15 transition-all group">
                <div className="flex flex-col items-center gap-1 min-w-[60px]">
                   <div className="text-[10px] font-black text-white/30 uppercase">{rel.from}</div>
                   <div className="h-4 w-[1px] bg-white/10" />
                   <div className="text-[10px] font-black text-home-acc uppercase">{rel.to}</div>
                </div>
                <div className="flex-1">
                   <div className="text-[10px] font-black text-white/20 uppercase tracking-widest mb-1">Impact: {rel.impact}</div>
                   <p className="text-xs text-white/60 font-medium leading-normal">{rel.desc}</p>
                </div>
             </div>
           ))}
        </div>
      </Card>
    </div>
  );
};

export default MultivariatePanel;
