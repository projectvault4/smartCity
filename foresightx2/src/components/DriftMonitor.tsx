import Card from './Card';
import { getDriftData } from '../services/dataService';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceDot } from 'recharts';

const DriftMonitor = () => {
  const data = getDriftData();

  return (
    <Card title="Model Drift Monitor" theme="model">
      <div className="p-2 space-y-6">
        <p className="text-sm text-white/50">Most ML projects train once and freeze. ForeSightX monitors accuracy in real-time and retrains automatically when drift is detected.</p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
           <div className="space-y-4">
              {data.map((d) => (
                <div key={d.week} className="flex justify-between items-center p-3 bg-white/5 rounded-xl border border-white/5">
                   <div className="text-xs font-bold text-white/70">Week {d.week} Accuracy</div>
                   <div className="flex items-center gap-3">
                      <span className={`text-sm font-mono font-bold ${d.drift ? 'text-red-400' : 'text-green-400'}`}>
                        {d.accuracy}%
                      </span>
                      {d.drift && <span className="text-[10px] text-red-500 font-bold uppercase tracking-tighter">← drift detected</span>}
                      {d.retrained && <span className="text-[10px] text-blue-400 font-bold uppercase tracking-tighter">← auto-retrained</span>}
                      {d.improved && <span className="text-[10px] text-home-acc font-bold uppercase tracking-tighter">← improved</span>}
                   </div>
                </div>
              ))}
           </div>

           <div className="h-[200px] w-full bg-black/10 rounded-2xl border border-white/5 p-4">
              <ResponsiveContainer width="100%" height="100%">
                 <AreaChart data={data}>
                    <XAxis dataKey="week" hide />
                    <YAxis domain={[90, 100]} hide />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Area 
                      type="monotone" 
                      dataKey="accuracy" 
                      stroke="#f39c12" 
                      fill="#f39c1215" 
                      strokeWidth={3} 
                    />
                    {data.filter(d => d.drift).map(d => (
                       <ReferenceDot key={d.week} x={d.week} y={d.accuracy} r={4} fill="#ef4444" stroke="none" />
                    ))}
                    {data.filter(d => d.retrained).map(d => (
                       <ReferenceDot key={d.week} x={d.week} y={d.accuracy} r={4} fill="#3b82f6" stroke="none" />
                    ))}
                 </AreaChart>
              </ResponsiveContainer>
              <div className="flex justify-center gap-4 mt-2">
                 <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-red-500" /><span className="text-[9px] text-white/40 uppercase">Drift</span></div>
                 <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-blue-500" /><span className="text-[9px] text-white/40 uppercase">Retrain</span></div>
              </div>
           </div>
        </div>

        <div className="bg-home-acc/5 border border-home-acc/10 rounded-xl p-4">
           <p className="text-[11px] text-white/70 leading-relaxed">
             "On March 14, a new metro line opened. Traffic patterns shifted. ForeSightX detected prediction error increase of 4.2% and triggered automatic retraining on new data. Accuracy restored within 6 hours."
           </p>
        </div>
      </div>
    </Card>
  );
};

export default DriftMonitor;
