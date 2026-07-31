import Card from './Card';
import { ModelConditions } from '../services/dataService';

const ANOMALY_APP_URL = import.meta.env.VITE_ANOMALY_APP_URL || '/anomaly-detection.html';

const formatNumber = (value: unknown) => (
  Number.isFinite(Number(value)) ? Math.round(Number(value)).toLocaleString() : 'loading'
);

const AnomalyDetectionEmbed = ({ modelConditions }: { modelConditions?: ModelConditions | null }) => {
  const forecastFields = [
    ['Source', modelConditions?.source || 'trained dataset artifacts'],
    ['Forecast For', modelConditions?.forecastFor || 'loading'],
    ['Traffic', `${formatNumber(modelConditions?.traffic?.flow)} / ${modelConditions?.traffic?.congestionLevel || 'loading'}`],
    ['AQI', formatNumber(modelConditions?.aqi?.aqi)],
    ['Temperature', `${formatNumber(modelConditions?.weather?.temperature?.value)} C`],
    ['Humidity', `${formatNumber(modelConditions?.weather?.humidity)}%`],
    ['Energy', `${formatNumber(modelConditions?.raw?.electricity_demand)} MW`]
  ];

  return (
    <div className="space-y-6">
      <Card title="Anomaly Detection" theme="air">
        <div className="mb-4 rounded-xl border border-air-acc/20 bg-air-acc/5 px-4 py-3 text-xs text-white/55">
          Dataset-connected anomaly dashboard. Alert generation and voice briefing use the same trained-model forecast values shown here.
          <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-7">
            {forecastFields.map(([label, value]) => (
              <div key={label} className="rounded-lg border border-white/10 bg-black/30 p-2">
                <div className="text-[8px] font-black uppercase tracking-widest text-white/30">{label}</div>
                <div className="mt-1 truncate text-[11px] font-bold text-white/75">{value}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="overflow-hidden rounded-xl border border-white/10 bg-black/40">
          <iframe
            title="ForeSightX Anomaly Detection"
            src={ANOMALY_APP_URL}
            className="h-[calc(100vh-180px)] min-h-[720px] w-full border-0"
          />
        </div>
      </Card>
    </div>
  );
};

export default AnomalyDetectionEmbed;
