import { motion } from 'motion/react';
import CityCanvas from './CityCanvas';
import { CityData } from '../services/dataService';
import { MapPin, Activity } from 'lucide-react';

interface HeroProps {
  mode: 'home' | 'traffic' | 'air' | 'energy' | 'weather';
  setMode: (mode: 'home' | 'traffic' | 'air' | 'energy' | 'weather') => void;
  data: CityData;
}

const Hero = ({ mode, setMode, data }: HeroProps) => {
  const infos = {
    home: 'Green sustainable city overview — solar panels, parks, wind turbines & clean streets',
    traffic: 'Traffic mode — amber road heatmap, accelerated vehicle flow active',
    air: 'Air quality mode — blue particle pollution streams and wind current bands',
    energy: 'Energy mode — purple power grid pulses and lightning arcs',
    weather: 'Weather mode — slate rain fronts and cloud systems'
  };

  const kpis = [
    { id: 'traffic', val: data.traffic.value, label: 'TRAFFIC FLOW', delta: data.traffic.delta, range: `Range ${data.traffic.range[0]}–${data.traffic.range[1]} · ${data.traffic.label}`, color: 'traf' },
    { id: 'air', val: `AQI ${data.air.value}`, label: 'AIR QUALITY', delta: data.air.delta, range: `Range ${data.air.range[0]}–${data.air.range[1]} · ${data.air.label}`, color: 'air' },
    { id: 'weather', val: `${data.weather.value}°C`, label: 'TEMPERATURE', delta: data.weather.delta, range: `Range ${data.weather.range[0]}–${data.weather.range[1]} · ${data.weather.label}`, color: 'wth' },
    { id: 'energy', val: `${data.energy.value} MW`, label: 'ELECTRICITY', delta: data.energy.delta, range: `Range ${data.energy.range[0]}–${data.energy.range[1]} · ${data.energy.label}`, color: 'eng' },
  ];

  return (
    <div className="relative w-full h-[600px] overflow-hidden flex flex-col">
      {/* LIVE DATA PULSE BAR */}
      <div className="bg-black/60 backdrop-blur-xl border-b border-white/5 py-2.5 px-8 flex flex-wrap items-center justify-between gap-6 overflow-x-auto whitespace-nowrap z-20">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            <span className="text-[10px] font-black text-white uppercase tracking-widest">LIVE</span>
          </div>
          <div className="flex items-center gap-2">
            <MapPin size={12} className="text-home-acc" />
            <span className="text-[10px] font-bold text-white/70 uppercase">Bengaluru</span>
          </div>
          <div className="text-[10px] text-white/30 font-medium">Last updated: {data.forecastFor ? new Date(data.forecastFor).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : data.timestamp}</div>
        </div>
        
        <div className="flex items-center gap-6 text-[10px] font-bold text-white/50 uppercase tracking-tight">
          <div className="flex items-center gap-2">
             <span>TRAFFIC FLOW: <span className="text-traf-acc">{data.traffic.value.toLocaleString()}</span></span>
          </div>
          <div className="h-3 w-[1px] bg-white/10" />
          <div className="flex items-center gap-2">
             <span>AQI: <span className={data.air.value > 150 ? 'text-red-400' : 'text-air-acc'}>{data.air.value}</span></span>
          </div>
          <div className="h-3 w-[1px] bg-white/10" />
          <div className="flex items-center gap-2">
             <span>GRID LOAD: <span className="text-eng-acc">{data.energy.value.toLocaleString()} MW</span></span>
          </div>
          <div className="h-3 w-[1px] bg-white/10" />
          <div className="flex items-center gap-2">
             <span>TEMP: <span className="text-wth-acc">{data.weather.value}°C</span></span>
          </div>
        </div>
      </div>

      <div className="flex-1 relative">
        <CityCanvas mode={mode} />
        
        <div className="absolute inset-0 flex flex-col justify-between p-8 md:p-12">
          <div className="flex justify-between items-start">
            <div className="flex flex-col gap-1">
              <div className="font-display text-[32px] md:text-[42px] font-black tracking-tighter leading-none">
                Fore<span className="text-home-acc not-italic">Sight</span>X
              </div>
              <div className="flex items-center gap-2 text-[10px] text-white/40 font-bold tracking-[2px] uppercase">
                <div className="w-1.5 h-1.5 bg-home-acc rounded-full" />
                Autonomous Urban Pulse Engine
              </div>
            </div>
            <div className="flex flex-col items-end gap-2">
               <div className="flex items-center gap-2 bg-black/40 border border-white/10 px-4 py-2 rounded-xl backdrop-blur-md">
                  <Activity size={12} className="text-home-acc" />
                  <span className="text-[10px] font-bold text-white tracking-widest uppercase">System Operational</span>
               </div>
               <div className="text-[9px] text-white/30 font-medium">B-NODE: BENGALURU-C7</div>
            </div>
          </div>

          <div className="flex flex-col items-center gap-6">
            <div className="flex gap-3 flex-wrap justify-center">
              {kpis.map((kpi) => (
                <motion.div
                  key={kpi.id}
                  whileHover={{ y: -4, backgroundColor: 'rgba(255,255,255,0.08)' }}
                  onClick={() => setMode(kpi.id as any)}
                  className={`bg-black/35 border border-white/15 rounded-[20px] p-4 px-6 text-center backdrop-blur-xl cursor-pointer transition-all shadow-2xl
                    ${mode === kpi.id ? `border-${kpi.color}-acc bg-${kpi.color}-acc/5 ring-1 ring-${kpi.color}-acc/30` : 'hover:border-white/30'}`}
                >
                  <div className="text-[10px] text-white/50 tracking-[2px] font-bold uppercase">{kpi.label}</div>
                  <div className={`text-[11px] mt-2 font-black ${kpi.delta.includes('Down') ? 'text-[#ff7675]' : kpi.delta.includes('Stable') ? 'text-white/40' : 'text-home-acc'}`}>
                    {kpi.delta}
                  </div>
                  <div className="text-[8px] mt-1 text-white/20 font-bold uppercase truncate max-w-[120px]">{kpi.range.split(' · ')[1]}</div>
                </motion.div>
              ))}
            </div>

            <div className="flex gap-2.5 justify-center flex-wrap">
              {['home', 'traffic', 'air', 'energy', 'weather'].map((t) => (
                <button
                  key={t}
                  onClick={() => setMode(t as any)}
                  className={`px-6 py-2.5 rounded-xl border transition-all text-[10px] font-bold uppercase tracking-widest
                    ${mode === t 
                      ? `bg-${t === 'home' ? 'home' : t === 'traffic' ? 'traf' : t === 'air' ? 'air' : t === 'energy' ? 'eng' : 'wth'}-acc/20 border-${t === 'home' ? 'home' : t === 'traffic' ? 'traf' : t === 'air' ? 'air' : t === 'energy' ? 'eng' : 'wth'}-acc text-${t === 'home' ? 'home' : t === 'traffic' ? 'traf' : t === 'air' ? 'air' : t === 'energy' ? 'eng' : 'wth'}-acc shadow-lg` 
                      : 'bg-black/40 border-white/10 text-white/40 hover:bg-black/60 hover:text-white/80'}`}
                >
                  {t === 'home' ? 'City Pulse' : t}
                </button>
              ))}
            </div>
          </div>

          <div className="flex justify-between items-center w-full">
             <div className="text-[11px] text-white/45 font-medium italic">
                "{infos[mode]}"
             </div>
             <div className="text-[10px] font-bold text-white/30 uppercase tracking-widest">
                Forecast Horizon: 60M
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
