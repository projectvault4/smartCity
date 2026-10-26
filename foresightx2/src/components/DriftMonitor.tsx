import { useCallback, useEffect, useState } from 'react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import Card from './Card';
import { backendApi, LiveCheck, fmtCityDateTime, fmtCityHour, fmtNum, pct } from '../services/dataService';

// Live reality check: the outlook vs what WAQI / Open-Meteo / OpenWeather
// observe right now, stored hourly, with a drift rule based on the error
// measured on held-out data. Replaces the earlier hardcoded drift demo.

const STATUS_STYLE: Record<string, string> = {
  ok: 'bg-home-acc/15 text-home-acc border-home-acc/30',
  drift: 'bg-red-500/15 text-red-300 border-red-500/30',
  collecting: 'bg-white/5 text-white/60 border-white/15',
  insufficient_data: 'bg-white/5 text-white/60 border-white/15',
  no_data: 'bg-white/5 text-white/40 border-white/10'
};

const DriftMonitor = () => {
  const [data, setData] = useState<LiveCheck | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback((refresh: boolean) => {
    setLoading(true);
    backendApi.liveCheck(refresh)
      .then((r) => { setData(r.data); setError(null); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(true); }, [load]);

  const latest = data?.latest;
  const chart = (data?.history || []).map((r) => ({
    time: fmtCityHour(r.observedHour),
    observedAqi: r.aqiCpcb,
    forecastAqi: r.forecastAqi,
    observedTemp: r.temperature,
    forecastTemp: r.forecastTemperature
  }));

  return (
    <Card title="Live reality check & drift monitor" theme="model">
      <div className="space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-white/55 max-w-2xl">
            Every hour the outlook is compared with what is observed in Bengaluru right now. Drift is flagged when the last 24 h error grows well beyond
            the error we measured on held-out data, which is the signal to retrain.
          </p>
          <div className="flex items-center gap-3">
            {data && <span className={`px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-widest border ${STATUS_STYLE[data.status]}`}>{data.status}</span>}
            <button onClick={() => load(true)} disabled={loading} className="px-3 py-1.5 rounded-lg border border-white/15 text-[11px] text-white/70 disabled:opacity-50">
              {loading ? 'Checking…' : 'Check now'}
            </button>
          </div>
        </div>

        {error && <p className="text-red-300 text-sm">Live check unavailable: {error}</p>}

        {latest && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white/5 rounded-xl p-4">
              <div className="text-[10px] uppercase tracking-widest text-white/40">AQI now (CPCB, approx.)</div>
              <div className="font-display text-3xl font-black text-white">{fmtNum(latest.aqiCpcb)}</div>
              <div className="text-[11px] text-white/50">{latest.aqiCategory} · outlook {fmtNum(latest.forecastAqi)} (90%: {fmtNum(latest.intervals?.aqi?.low)}–{fmtNum(latest.intervals?.aqi?.high)})</div>
            </div>
            <div className="bg-white/5 rounded-xl p-4">
              <div className="text-[10px] uppercase tracking-widest text-white/40">Temperature now</div>
              <div className="font-display text-3xl font-black text-white">{fmtNum(latest.temperature, 1)}°C</div>
              <div className="text-[11px] text-white/50">outlook {fmtNum(latest.forecastTemperature, 1)}°C (90%: {fmtNum(latest.intervals?.temperature?.low, 1)}–{fmtNum(latest.intervals?.temperature?.high, 1)})</div>
            </div>
            <div className="bg-white/5 rounded-xl p-4">
              <div className="text-[10px] uppercase tracking-widest text-white/40">Humidity now</div>
              <div className="font-display text-3xl font-black text-white">{fmtNum(latest.humidity)}%</div>
              <div className="text-[11px] text-white/50">outlook {fmtNum(latest.forecastHumidity)}% · read {fmtCityDateTime(latest.observedAt)}</div>
            </div>
          </div>
        )}

        {data && (
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]">
              <thead>
                <tr className="text-white/40 text-left">
                  <th className="py-2 font-normal">Last 24 h</th>
                  <th className="py-2 font-normal text-right">Readings</th>
                  <th className="py-2 font-normal text-right">Live error (MAE)</th>
                  <th className="py-2 font-normal text-right">Bias (obs − outlook)</th>
                  <th className="py-2 font-normal text-right">Held-out error</th>
                  <th className="py-2 font-normal text-right">Inside 90% range</th>
                  <th className="py-2 font-normal text-right">Status</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.stats).map(([k, s]) => (
                  <tr key={k} className="border-t border-white/5 text-white/75">
                    <td className="py-2 capitalize">{k === 'aqi' ? 'AQI' : k}</td>
                    <td className="py-2 text-right font-mono">{s.points}</td>
                    <td className="py-2 text-right font-mono">{fmtNum(s.mae, 1)}</td>
                    <td className="py-2 text-right font-mono">{s.bias === undefined ? '—' : `${s.bias > 0 ? '+' : ''}${fmtNum(s.bias, 1)}`}</td>
                    <td className="py-2 text-right font-mono">{fmtNum(s.expectedMae ?? null, 1)}</td>
                    <td className="py-2 text-right font-mono">{pct(s.coverage ?? null)}</td>
                    <td className="py-2 text-right"><span className={`px-2 py-0.5 rounded border text-[10px] uppercase ${STATUS_STYLE[s.status] || STATUS_STYLE.no_data}`}>{s.status.replace('_', ' ')}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {chart.length > 1 && (
          <div className="h-[200px] bg-black/10 rounded-2xl border border-white/5 p-3">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chart}>
                <CartesianGrid stroke="#ffffff10" vertical={false} />
                <XAxis dataKey="time" tick={{ fill: '#ffffff55', fontSize: 10 }} />
                <YAxis tick={{ fill: '#ffffff55', fontSize: 10 }} width={40} />
                <Tooltip contentStyle={{ background: '#0d1a10', border: '1px solid #ffffff22', fontSize: 11 }} />
                <Line dataKey="observedAqi" name="Observed AQI" stroke="#46c9b0" dot={false} isAnimationActive={false} />
                <Line dataKey="forecastAqi" name="Outlook AQI" stroke="#46c9b0" strokeDasharray="4 3" dot={false} isAnimationActive={false} />
                <Line dataKey="observedTemp" name="Observed °C" stroke="#6fb8e8" dot={false} isAnimationActive={false} />
                <Line dataKey="forecastTemp" name="Outlook °C" stroke="#6fb8e8" strokeDasharray="4 3" dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {data && (
          <div className="text-[11px] text-white/40 space-y-1">
            <p>{data.rule} When the AQI or temperature bias holds over 3+ readings, the Decision Center shifts the outlook by that bias.</p>
            {data.notes.map((n) => <p key={n}>• {n}</p>)}
            {latest?.metadata?.air && <p>• Air source now: {latest.metadata.air.source}{latest.metadata.air.station ? ` (${latest.metadata.air.station})` : ''}.</p>}
            {latest?.metadata?.errors?.length ? <p>• Fallbacks used: {latest.metadata.errors.map((e) => e.provider).join(', ')} unavailable.</p> : null}
          </div>
        )}
      </div>
    </Card>
  );
};

export default DriftMonitor;
