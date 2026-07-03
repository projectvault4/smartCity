import Card from './Card';
import { CityData } from '../services/dataService';
import { BrainCircuit, Link, MessageSquare } from 'lucide-react';

const XaiPanel = ({ data }: { data: CityData }) => {
  return (
    <div className="space-y-6">
      <Card title="Explainable AI (XAI) — Reasoning Engine" theme="home">
        <div className="space-y-8 p-2">
          
          {/* WHY THE MODEL THINKS THIS */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-4">
              <BrainCircuit size={16} className="text-home-acc" />
              <span className="text-[11px] font-bold text-white/40 uppercase tracking-widest">Why The Model Thinks This</span>
            </div>
            <ul className="space-y-3">
              {[
                `Traffic may ${data.traffic.trend === 'down' ? 'decrease' : 'increase'} because the recent traffic pattern has been cooling down.`,
                `AQI is expected to ${data.air.trend === 'down' ? 'ease' : 'stabilize'} because traffic pressure is easing, reducing pollution.`,
                `Temperature looks stable because recent patterns are consistent with seasonal history.`,
                `Electricity demand may decrease because traffic and temperature are expected to stay lower.`
              ].map((text, i) => (
                <li key={i} className="flex items-start gap-4 p-4 bg-white/5 rounded-xl border border-white/5">
                   <div className="w-1.5 h-1.5 rounded-full bg-home-acc mt-1.5 shrink-0" />
                   <p className="text-xs text-white/70 leading-relaxed font-medium">{text}</p>
                </li>
              ))}
            </ul>
          </div>

          {/* INTERCONNECTED OUTPUT */}
          <div className="space-y-4 pt-6 border-t border-white/5">
            <div className="flex items-center gap-2 mb-4">
              <Link size={16} className="text-home-acc" />
              <span className="text-[11px] font-bold text-white/40 uppercase tracking-widest">Interconnected Output Logic</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
               {[
                 `Traffic Flow is expected to decrease, which can make AQI decrease through lower road emissions.`,
                 `AQI is expected to decrease, which can make Temperature stay nearly stable.`,
                 `Temperature is expected to stay nearly stable, which can make Electricity Demand decrease.`,
                 `Overall chain: Traffic Flow -> AQI -> Temperature -> Electricity Demand.`
               ].map((text, i) => (
                 <div key={i} className="p-4 bg-home-acc/5 border border-home-acc/10 rounded-xl">
                   <p className="text-[11px] text-white/60 leading-relaxed italic">
                     {text}
                   </p>
                 </div>
               ))}
            </div>
          </div>

          {/* SIMPLE SUMMARY */}
          <div className="space-y-4 pt-6 border-t border-white/5">
            <div className="flex items-center gap-2 mb-4">
              <MessageSquare size={16} className="text-home-acc" />
              <span className="text-[11px] font-bold text-white/40 uppercase tracking-widest">Simple Summary</span>
            </div>
            <div className="bg-white/5 rounded-xl p-4 border border-white/5">
               <ul className="space-y-2">
                 <li className="text-xs text-white/70 tracking-wide">— Traffic Flow may go down from current levels.</li>
                 <li className="text-xs text-white/70 tracking-wide">— AQI may go down slightly next hour.</li>
                 <li className="text-xs text-white/70 tracking-wide">— Temperature should stay close to the current value.</li>
                 <li className="text-xs text-white/70 tracking-wide">— Electricity Demand may stay stable or decrease slightly.</li>
               </ul>
            </div>
          </div>

        </div>
      </Card>
    </div>
  );
};

export default XaiPanel;
