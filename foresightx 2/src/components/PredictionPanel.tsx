import { useEffect, useMemo, useState } from 'react';
import Card from './Card';
import { backendApi, CityData, ModelConditions } from '../services/dataService';
import { AlertTriangle, Info, ShieldCheck, Zap } from 'lucide-react';

const metricSpecs = [
  { key: 'traffic' as const, title: 'Traffic', unit: 'veh/hr', theme: 'traf' as const },
  { key: 'air' as const, title: 'AQI', unit: 'AQI', theme: 'air' as const },
  { key: 'energy' as const, title: 'Energy', unit: 'MW', theme: 'eng' as const },
  { key: 'weather' as const, title: 'Weather', unit: 'C', theme: 'wth' as const },
];

const valueFromCondition = (condition: ModelConditions, key: typeof metricSpecs[number]['key'], fallback: CityData) => {
  if (key === 'traffic') return Math.round(Number(condition.traffic?.flow ?? fallback.traffic.value));
  if (key === 'air') return Math.round(Number(condition.aqi?.aqi ?? fallback.air.value));
  if (key === 'energy') return Math.round(Number(condition.raw?.electricity_demand ?? fallback.energy.value));
  return Math.round(Number(condition.weather?.temperature?.value ?? fallback.weather.value));
};

const PredictionPanel = ({ data, city = 'bangalore' }: { data: CityData; city?: string }) => {
  const hours = [1, 2, 3, 4, 5, 6];
  const [forecasts, setForecasts] = useState<Record<number, ModelConditions>>({});
  const [status, setStatus] = useState<'loading' | 'model' | 'fallback'>('loading');

  useEffect(() => {
    let mounted = true;

    Promise.all(hours.map((hour) => backendApi.modelConditions(city, hour)))
      .then((responses) => {
        if (!mounted) return;
        setForecasts(
          responses.reduce<Record<number, ModelConditions>>((next, response, index) => {
            next[hours[index]] = response.data;
            return next;
          }, {})
        );
        setStatus('model');
      })
      .catch(() => {
        if (mounted) setStatus('fallback');
      });

    return () => {
      mounted = false;
    };
  }, [city]);

  const fallbackValues = useMemo(() => ({
    traffic: data.traffic.value,
    air: data.air.value,
    energy: data.energy.value,
    weather: data.weather.value
  }), [data]);

  const getForecast = (key: typeof metricSpecs[number]['key'], hour: number) => {
    const modelForecast = forecasts[hour];
    if (modelForecast) {
      return valueFromCondition(modelForecast, key, data);
    }

    return fallbackValues[key];
  };

  return (
    <div className="space-y-6">
      <Card title="Multi-Hour Forecasting (4-6H Horizon)" theme="home">
        <div className="space-y-10 p-2">
          <div className="flex items-center gap-2 rounded-xl border border-home-acc/20 bg-home-acc/5 px-4 py-3 text-xs text-white/55">
            <ShieldCheck size={16} className="text-home-acc" />
            {status === 'model'
              ? `Connected to trained dataset model for ${city}.`
              : status === 'loading'
                ? 'Loading trained dataset model forecasts.'
                : 'Backend unavailable; showing last loaded dashboard values.'}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {metricSpecs.map((metric) => (
              <div key={metric.title} className="space-y-6">
                <div className="flex flex-col items-center gap-2">
                  <div className={`w-3 h-3 rounded-full bg-${metric.theme}-acc shadow-[0_0_12px_rgba(255,255,255,0.2)]`} />
                  <span className="text-[10px] font-black uppercase tracking-[2.5px] text-white/50 text-center">{metric.title}</span>
                </div>
                
                <div className="flex flex-col gap-2">
                  {hours.map((h) => (
                    <div key={h} className="bg-white/5 border border-white/5 rounded-xl p-3 flex flex-col items-center gap-1 hover:bg-white/10 transition-colors group">
                      <div className="text-[8px] font-bold text-white/20 uppercase tracking-tighter group-hover:text-white/40 transition-colors">T + {h}H</div>
                      <div className="text-lg font-black text-white">{getForecast(metric.key, h)}</div>
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
                <p className="text-xs text-white/60 font-medium">Forecasts are read from the trained dataset model via the backend model API. Uncertainty increases beyond the 4H window.</p>
             </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default PredictionPanel;
