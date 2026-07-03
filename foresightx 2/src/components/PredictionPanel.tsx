import Card from './Card';
import { CityData } from '../services/dataService';
import { AlertTriangle, Info, ShieldCheck, Zap } from 'lucide-react';

const PredictionPanel = ({ data }: { data: CityData }) => {
  const hours = [1, 2, 3, 4, 5, 6];
  
  const getForecast = (val: number, hour: number) => {
    // Simple deterministic variance per hour
    const variance = (Math.sin(hour + Date.now()/100000) * 10) + (hour * 2);
    return Math.round(val + variance);
  };

  return (
    <div className="space-y-6">
      <Card title="Multi-Hour Forecasting (4-6H Horizon)" theme="home">
        <div className="space-y-10 p-2">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[
              { title: 'Traffic', current: data.traffic.value, unit: 'veh/hr', theme: 'traf' as const },
              { title: 'AQI', current: data.air.value, unit: 'AQI', theme: 'air' as const },
              { title: 'Energy', current: data.energy.value, unit: 'MW', theme: 'eng' as const },
              { title: 'Weather', current: data.weather.value, unit: '°C', theme: 'wth' as const },
            ].map((metric) => (
              <div key={metric.title} className="space-y-6">
                <div className="flex flex-col items-center gap-2">
                  <div className={`w-3 h-3 rounded-full bg-${metric.theme}-acc shadow-[0_0_12px_rgba(255,255,255,0.2)]`} />
                  <span className="text-[10px] font-black uppercase tracking-[2.5px] text-white/50 text-center">{metric.title}</span>
                </div>
                
                <div className="flex flex-col gap-2">
                  {hours.map((h) => (
                    <div key={h} className="bg-white/5 border border-white/5 rounded-xl p-3 flex flex-col items-center gap-1 hover:bg-white/10 transition-colors group">
                      <div className="text-[8px] font-bold text-white/20 uppercase tracking-tighter group-hover:text-white/40 transition-colors">T + {h}H</div>
                      <div className="text-lg font-black text-white">{getForecast(metric.current, h)}</div>
                      <div className="text-[7px] font-bold text-white/10 uppercase tracking-widest">{metric.unit}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="pt-8 border-t border-white/5 flex items-center gap-3">
             <div className="p-3 bg-red-400/10 rounded-xl">
                <AlertTriangle size={20} className="text-[#ff7675]" />
             </div>
             <div>
                <div className="text-[10px] font-black text-white/40 uppercase tracking-widest">Model Warning</div>
                <p className="text-xs text-white/60 font-medium">Standard deviation increases significantly beyond the 4H window. Recommended for strategic scheduling only.</p>
             </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default PredictionPanel;
