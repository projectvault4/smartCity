import { useState } from 'react';
import Card from './Card';
import { Calendar, Users, Clock, MapPin } from 'lucide-react';

const EventImpact = () => {
  const [selectedEvent, setSelectedEvent] = useState('ipl');

const events = {
    ipl: {
      name: 'IPL Match at Chinnaswamy',
      date: 'May 15, 2026',
      time: '7:30 PM',
      crowd: '45,000',
      pre: [
        { label: 'Traffic on MG Road', val: '+78%', sub: 'above baseline' },
        { label: 'Parking zone energy', val: '+340%', sub: 'demand spike' },
        { label: 'Air quality near venue', val: 'AQI 210', sub: '(Very Poor)' }
      ],
      during: [
        { label: 'Surrounding traffic', val: '-23%', sub: '(people inside)' },
        { label: 'Stadium power demand', val: '1.4 MW', sub: 'sustained' }
      ]
    },
    festival: {
      name: 'Kalasipalya Festival',
      date: 'June 02, 2026',
      time: '10:00 AM',
      crowd: '120,000',
      pre: [
        { label: 'Market traffic', val: '+120%', sub: 'congestion red' },
        { label: 'Waste management load', val: '+45%', sub: 'operational surge' }
      ],
      during: [
        { label: 'Local Grid Load', val: '2.1 MW', sub: 'localized peak' }
      ]
    },
    rally: {
      name: 'Political Rally (Freedom Park)',
      date: 'May 20, 2026',
      time: '11:00 AM',
      crowd: '85,000',
      pre: [
        { label: 'Traffic Diversions', val: '+135%', sub: 'gridlock predicted' },
        { label: 'Public Transport load', val: '+60%', sub: 'over capacity' }
      ],
      during: [
        { label: 'Mobile Network Congestion', val: 'High', sub: 'limited bandwidth' },
        { label: 'Temporary structure load', val: '0.8 MW', sub: 'connected' }
      ]
    },
    marathon: {
      name: 'Bengaluru Midnight Run',
      date: 'July 12, 2026',
      time: '12:00 AM',
      crowd: '25,000',
      pre: [
        { label: 'Road Closures', val: '-40%', sub: 'vehicular drop' },
        { label: 'Ambient AQI', val: '-25%', sub: 'clearer air' }
      ],
      during: [
        { label: 'Street Light energy', val: '+15%', sub: 'safety override' },
        { label: 'Hydration hub activity', val: 'Active', sub: '24 nodes' }
      ]
    }
  };

  const curr = events[selectedEvent as keyof typeof events];

  return (
    <Card title="Urban Event Impact Predictor" theme="air">
      <div className="p-2 space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
           <div className="space-y-4">
              <p className="text-sm text-white/50">Select a city event to see how it reshapes urban parameters across multi-domains.</p>
              
              <div className="space-y-3">
                 <div className="flex flex-col gap-1.5">
                    <label className="text-[10px] font-bold text-white/30 uppercase">Event Type</label>
                    <select 
                      value={selectedEvent}
                      onChange={(e) => setSelectedEvent(e.target.value)}
                      className="bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none"
                    >
                       <option value="ipl">IPL Match at Chinnaswamy</option>
                       <option value="festival">Kalasipalya Festival</option>
                       <option value="rally">Political Rally (Freedom Park)</option>
                       <option value="marathon">Bengaluru Midnight Run</option>
                    </select>
                 </div>
                 
                 <div className="grid grid-cols-2 gap-3">
                    <div className="bg-white/5 p-3 rounded-xl border border-white/5 flex items-center gap-3">
                       <Calendar size={14} className="text-air-acc" />
                       <div className="text-[11px] font-medium text-white/60">{curr.date}</div>
                    </div>
                    <div className="bg-white/5 p-3 rounded-xl border border-white/5 flex items-center gap-3">
                       <Clock size={14} className="text-air-acc" />
                       <div className="text-[11px] font-medium text-white/60">{curr.time}</div>
                    </div>
                    <div className="bg-white/5 p-3 rounded-xl border border-white/5 flex items-center gap-3">
                       <Users size={14} className="text-air-acc" />
                       <div className="text-[11px] font-medium text-white/60">{curr.crowd}</div>
                    </div>
                    <div className="bg-white/5 p-3 rounded-xl border border-white/5 flex items-center gap-3">
                       <MapPin size={14} className="text-air-acc" />
                       <div className="text-[11px] font-medium text-white/60">Bengaluru Central</div>
                    </div>
                 </div>
              </div>
           </div>

           <div className="space-y-6">
              <div className="space-y-4">
                 <div className="text-[10px] font-bold text-white/30 uppercase tracking-[2px]">Pre-event Impact Prediction</div>
                 <div className="space-y-3">
                    {curr.pre.map((item, i) => (
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
                    {curr.during.map((item, i) => (
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
