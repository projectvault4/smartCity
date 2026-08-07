import { useEffect, useMemo, useState } from 'react';
import Card from './Card';
import { Calendar, Users, Clock, MapPin } from 'lucide-react';
import { backendApi, YearlyForecast, YearlyForecastPoint } from '../services/dataService';

const EVENT_DEFS = [
  { id: 'ipl', name: 'IPL Match at Chinnaswamy', date: 'May 15, 2026', iso: '2026-05-15', time: '7:30 PM', hour: '19', crowd: '45,000', venue: 'Bengaluru Central' },
  { id: 'festival', name: 'Kalasipalya Festival', date: 'June 02, 2026', iso: '2026-06-02', time: '10:00 AM', hour: '10', crowd: '120,000', venue: 'Kalasipalya Market' },
  { id: 'rally', name: 'Political Rally (Freedom Park)', date: 'May 20, 2026', iso: '2026-05-20', time: '11:00 AM', hour: '11', crowd: '85,000', venue: 'Freedom Park' },
  { id: 'marathon', name: 'Bengaluru Midnight Run', date: 'July 12, 2026', iso: '2026-07-12', time: '12:00 AM', hour: '00', crowd: '25,000', venue: 'Bengaluru Central' },
] as const;

type ImpactItem = { label: string; val: string; sub: string };

interface ImpactBlock {
  pre: ImpactItem[];
  during: ImpactItem[];
  source: 'model' | 'fallback';
}

const FALLBACK: Record<string, ImpactBlock> = {
  ipl: {
    pre: [
      { label: 'Traffic on MG Road', val: '+78%', sub: 'above baseline' },
      { label: 'Parking zone energy', val: '+340%', sub: 'demand spike' },
      { label: 'Air quality near venue', val: 'AQI 210', sub: '(Very Poor)' }
    ],
    during: [
      { label: 'Surrounding traffic', val: '-23%', sub: '(people inside)' },
      { label: 'Stadium power demand', val: '1.4 MW', sub: 'sustained' }
    ],
    source: 'fallback'
  },
  festival: {
    pre: [
      { label: 'Market traffic', val: '+120%', sub: 'congestion red' },
      { label: 'Waste management load', val: '+45%', sub: 'operational surge' }
    ],
    during: [
      { label: 'Local Grid Load', val: '2.1 MW', sub: 'localized peak' }
    ],
    source: 'fallback'
  },
  rally: {
    pre: [
      { label: 'Traffic Diversions', val: '+135%', sub: 'gridlock predicted' },
      { label: 'Public Transport load', val: '+60%', sub: 'over capacity' }
    ],
    during: [
      { label: 'Mobile Network Congestion', val: 'High', sub: 'limited bandwidth' },
      { label: 'Temporary structure load', val: '0.8 MW', sub: 'connected' }
    ],
    source: 'fallback'
  },
  marathon: {
    pre: [
      { label: 'Road Closures', val: '-40%', sub: 'vehicular drop' },
      { label: 'Ambient AQI', val: '-25%', sub: 'clearer air' }
    ],
    during: [
      { label: 'Street Light energy', val: '+15%', sub: 'safety override' },
      { label: 'Hydration hub activity', val: 'Active', sub: '24 nodes' }
    ],
    source: 'fallback'
  }
};

const AQI_BANDS = [
  { max: 50, label: 'Good' },
  { max: 100, label: 'Moderate' },
  { max: 150, label: 'Poor' },
  { max: 200, label: 'Very Poor' },
  { max: 999, label: 'Severe' },
];

const aqiLabel = (v: number | null) => {
  if (v === null || !Number.isFinite(v)) return '—';
  return AQI_BANDS.find((b) => v <= b.max)?.label || 'Severe';
};

const round1 = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? null : Math.round(v * 10) / 10);

const pct = (dayVal: number | null, annual: number | null) => {
  if (dayVal === null || annual === null || annual === 0) return null;
  return Math.round(((dayVal - annual) / annual) * 100);
};

const EventImpact = ({ city = 'bangalore' }: { city?: string }) => {
  const [selectedEvent, setSelectedEvent] = useState('ipl');
  const [yearly, setYearly] = useState<YearlyForecast | null>(null);
  const [status, setStatus] = useState<'loading' | 'model' | 'fallback'>('loading');

  useEffect(() => {
    let mounted = true;
    setStatus('loading');
    backendApi.modelYearlyForecast(city, 2026, 'monthly')
      .then((response) => { if (mounted) { setYearly(response.data); setStatus('model'); } })
      .catch(() => { if (mounted) setStatus('fallback'); });
    return () => { mounted = false; };
  }, [city]);

  const byDay = useMemo(() => {
    const map = new Map<string, YearlyForecastPoint[]>();
    if (!yearly) return map;
    for (const p of yearly.series) {
      const key = String(p.timestamp).slice(0, 10);
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(p);
    }
    return map;
  }, [yearly]);

  const impact = useMemo<Record<string, ImpactBlock>>(() => {
    const base = JSON.parse(JSON.stringify(FALLBACK)) as Record<string, ImpactBlock>;
    if (status !== 'model' || !yearly || byDay.size === 0) return base;

    const annual = yearly.annual;
    const mean = (pts: YearlyForecastPoint[], col: 'trafficFlow' | 'aqi' | 'electricityDemand' | 'temperature') => {
      const vals = pts.map((p) => round1(p[col])).filter((v): v is number => v !== null);
      return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
    };

    for (const ev of EVENT_DEFS) {
      const dayPts = byDay.get(ev.iso) || [];
      if (dayPts.length === 0) continue;

      const dayTraffic = mean(dayPts, 'trafficFlow');
      const dayAqi = mean(dayPts, 'aqi');
      const dayEnergy = mean(dayPts, 'electricityDemand');

      const eventHourPts = dayPts.filter((p) => String(p.timestamp).slice(11, 13) === ev.hour);
      const evPts = eventHourPts.length ? eventHourPts : dayPts;
      const evTraffic = mean(evPts, 'trafficFlow');
      const evAqi = mean(evPts, 'aqi');
      const evEnergy = mean(evPts, 'electricityDemand');
      const evTemp = mean(evPts, 'temperature');

      const tp = pct(dayTraffic, annual.trafficFlow);
      const ep = pct(dayEnergy, annual.electricityDemand);

      const pre: ImpactItem[] = [];
      pre.push({
        label: 'Traffic Flow (day avg)',
        val: tp !== null ? `${tp >= 0 ? '+' : ''}${tp}%` : '—',
        sub: `vs annual mean · ${Math.round(dayTraffic ?? 0).toLocaleString()} veh/hr`
      });
      pre.push({
        label: 'Energy Demand (day avg)',
        val: ep !== null ? `${ep >= 0 ? '+' : ''}${ep}%` : '—',
        sub: `vs annual mean · ${Math.round(dayEnergy ?? 0)} MW`
      });
      pre.push({
        label: 'Air Quality near venue',
        val: dayAqi !== null ? `AQI ${Math.round(dayAqi)}` : '—',
        sub: `(${aqiLabel(dayAqi)}) · day average`
      });

      const during: ImpactItem[] = [];
      if (evTraffic !== null) during.push({ label: 'Traffic at event hour', val: Math.round(evTraffic).toLocaleString(), sub: `veh/hr · ${ev.time}` });
      if (evAqi !== null) during.push({ label: 'Air Quality at event', val: `AQI ${Math.round(evAqi)}`, sub: `(${aqiLabel(evAqi)})` });
      if (evEnergy !== null) during.push({ label: 'Grid load at event', val: `${Math.round(evEnergy)} MW`, sub: 'forecasted demand' });
      if (evTemp !== null) during.push({ label: 'Temperature at event', val: `${Math.round(evTemp)}°C`, sub: 'forecasted ambient' });

      base[ev.id] = { pre, during, source: 'model' };
    }
    return base;
  }, [status, yearly, byDay]);

  const def = EVENT_DEFS.find((e) => e.id === selectedEvent)!;
  const block = impact[selectedEvent] || FALLBACK[selectedEvent];

  return (
    <Card title="Urban Event Impact Predictor" theme="air">
      <div className="p-2 space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
           <div className="space-y-4">
              <p className="text-sm text-white/50">Select a city event to see its projected impact on the 2026 forecasted urban parameters.</p>
              
              <div className="space-y-3">
                 <div className="flex flex-col gap-1.5">
                    <label className="text-[10px] font-bold text-white/30 uppercase">Event Type</label>
                    <select 
                      value={selectedEvent}
                      onChange={(e) => setSelectedEvent(e.target.value)}
                      className="bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none"
                    >
                       {EVENT_DEFS.map((e) => (
                         <option key={e.id} value={e.id}>{e.name}</option>
                       ))}
                    </select>
                 </div>
                 
                 <div className="grid grid-cols-2 gap-3">
                    <div className="bg-white/5 p-3 rounded-xl border border-white/5 flex items-center gap-3">
                       <Calendar size={14} className="text-air-acc" />
                       <div className="text-[11px] font-medium text-white/60">{def.date}</div>
                    </div>
                    <div className="bg-white/5 p-3 rounded-xl border border-white/5 flex items-center gap-3">
                       <Clock size={14} className="text-air-acc" />
                       <div className="text-[11px] font-medium text-white/60">{def.time}</div>
                    </div>
                    <div className="bg-white/5 p-3 rounded-xl border border-white/5 flex items-center gap-3">
                       <Users size={14} className="text-air-acc" />
                       <div className="text-[11px] font-medium text-white/60">{def.crowd}</div>
                    </div>
                    <div className="bg-white/5 p-3 rounded-xl border border-white/5 flex items-center gap-3">
                       <MapPin size={14} className="text-air-acc" />
                       <div className="text-[11px] font-medium text-white/60">{def.venue}</div>
                    </div>
                 </div>

                 <div className="text-[10px] text-white/30 font-medium uppercase tracking-widest">
                   Source: {status === 'loading' ? 'Loading 2026 forecast…' : block.source === 'model' ? 'Trained model · 2026 forecast' : 'Fallback demo (backend offline)'}
                 </div>
              </div>
           </div>

           <div className="space-y-6">
              <div className="space-y-4">
                 <div className="text-[10px] font-bold text-white/30 uppercase tracking-[2px]">Pre-event Impact Prediction</div>
                 <div className="space-y-3">
                    {block.pre.map((item, i) => (
                       <div key={i} className="flex justify-between items-end border-b border-white/5 pb-2">
                          <div className="text-xs text-white/70">{item.label}</div>
                          <div className="text-right">
                             <div className="text-sm font-bold text-air-acc">{item.val}</div>
                             <div className="text-[9px] text-white/30">{item.sub}</div>
                          </div>
                       </div>
                    ))}
                 </div>
              </div>

              <div className="space-y-4 pt-4">
                 <div className="text-[10px] font-bold text-white/30 uppercase tracking-[2px]">During Event (Dynamic state)</div>
                 <div className="space-y-3">
                    {block.during.map((item, i) => (
                       <div key={i} className="flex justify-between items-end border-b border-white/5 pb-2">
                          <div className="text-xs text-white/70">{item.label}</div>
                          <div className="text-right">
                             <div className="text-sm font-bold text-home-acc">{item.val}</div>
                             <div className="text-[9px] text-white/30">{item.sub}</div>
                          </div>
                       </div>
                    ))}
                 </div>
              </div>
           </div>
        </div>
      </div>
    </Card>
  );
};

export default EventImpact;
