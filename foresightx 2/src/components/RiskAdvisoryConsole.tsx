import { useEffect, useState } from 'react';
import { Loader2, RefreshCw, Send, Trash2 } from 'lucide-react';
import { backendApi, ModelConditions } from '../services/dataService';
import { HealthKey } from '../services/memberProfile';

interface Member {
  id: string;
  user_id: string;
  name: string;
  age?: number;
  phone?: string;
  ward?: string;
  preferences?: { health_conditions?: HealthKey[]; auto_risk_factors?: string[] };
}

interface Conditions {
  aqi: number;
  temp: number;
  rain: number;
  traffic: 'Low' | 'Moderate' | 'Severe';
}

interface Prediction {
  member: Member;
  score: number;
  drivers: string[];
  factors: string[];
  status: { label: 'ALERT' | 'WATCH' | 'SAFE'; color: string };
}

const HTML_KEY: Record<HealthKey, string> = {
  asthma_copd: 'asthma',
  heart: 'heart',
  diabetes: 'diabetes',
  pregnant: 'pregnant',
  limited_mobility: 'mobility',
  works_outdoors: 'outdoor',
  none: 'none',
};

const SENSITIVITY: Record<string, { aqi: number; temp: number; rain: number; traffic: number }> = {
  asthma:   { aqi: 0.55, temp: 0.08, rain: 0.03, traffic: 0.02 },
  heart:    { aqi: 0.30, temp: 0.30, rain: 0.05, traffic: 0.05 },
  diabetes: { aqi: 0.10, temp: 0.30, rain: 0.05, traffic: 0.05 },
  pregnant: { aqi: 0.35, temp: 0.25, rain: 0.10, traffic: 0.10 },
  mobility: { aqi: 0.15, temp: 0.20, rain: 0.30, traffic: 0.30 },
  outdoor:  { aqi: 0.20, temp: 0.40, rain: 0.20, traffic: 0.25 },
  none:     { aqi: 0.08, temp: 0.08, rain: 0.05, traffic: 0.05 },
};

const TRAFFIC_VAL: Record<Conditions['traffic'], number> = { Low: 20, Moderate: 55, Severe: 90 };

const HEALTH_LABEL: Record<string, string> = {
  asthma: 'Asthma/COPD',
  heart: 'Heart condition',
  diabetes: 'Diabetes',
  pregnant: 'Pregnant',
  mobility: 'Limited mobility',
  outdoor: 'Outdoor worker',
  none: 'No conditions',
  elderly_auto: 'Age 60+',
  child_auto: 'Age <12',
};

const initials = (name: string) =>
  name.split(' ').map((w) => w[0]).join('').slice(0, 2).toUpperCase();

const toHtmlKeys = (health: HealthKey[]): string[] =>
  health.filter((k) => k !== 'none').map((k) => HTML_KEY[k]);

const computeScore = (member: Member, cond: Conditions): Prediction => {
  const health = member.preferences?.health_conditions || [];
  let score = 0;
  const drivers: string[] = [];
  const factors: string[] = [];

  const effective = toHtmlKeys(health);
  const keys = effective.length ? effective : ['none'];

  const age = Number(member.age) || 0;
  if (age >= 60) factors.push('elderly_auto');
  if (age <= 12) factors.push('child_auto');

  keys.forEach((k) => {
    const s = SENSITIVITY[k] || SENSITIVITY.none;
    const aqiC = Math.max(0, cond.aqi - 50) / 250 * 100 * s.aqi;
    const tempC = Math.max(0, cond.temp - 25) / 20 * 100 * s.temp;
    const rainC = Math.min(cond.rain, 50) / 50 * 100 * s.rain;
    const trafC = TRAFFIC_VAL[cond.traffic] * s.traffic / 100 * 30;
    score += aqiC + tempC + rainC + trafC;
    if (aqiC > 8) drivers.push(`AQI ${cond.aqi}`);
    if (tempC > 8) drivers.push(`${cond.temp}°C heat`);
    if (rainC > 8) drivers.push(`${cond.rain}mm rain`);
    if (trafC > 8) drivers.push(`${cond.traffic.toLowerCase()} traffic`);
  });

  if (age >= 60) score += 8;
  if (age <= 12) score += 5;

  score = Math.min(100, Math.round(score));

  return {
    member,
    score,
    drivers: Array.from(new Set(drivers)).slice(0, 2),
    factors: [...factors, ...effective],
    status: score >= 55
      ? { label: 'ALERT', color: '#f0665f' }
      : score >= 30
        ? { label: 'WATCH', color: '#f0b849' }
        : { label: 'SAFE', color: '#4ade80' },
  };
};

const draftMessage = (member: Member, drivers: string[]): string => {
  const trigger = drivers.length ? drivers.join(' + ') : 'current conditions';
  const first = member.name.split(' ')[0];
  const health = member.preferences?.health_conditions || [];
  if (health.some((k) => ['asthma_copd', 'heart', 'pregnant'].includes(k))) {
    return `Hi ${first}, conditions in ${member.ward || 'your area'} may affect your health today (${trigger}). Please limit outdoor exposure and keep any prescribed medication on hand.`;
  }
  if (health.some((k) => ['works_outdoors', 'limited_mobility'].includes(k))) {
    return `Hi ${first}, conditions in ${member.ward || 'your area'} are difficult today (${trigger}). Take extra breaks/care if you're outdoors or travelling.`;
  }
  return `Hi ${first}, conditions in ${member.ward || 'your area'} have shifted (${trigger}). Take normal precautions today.`;
};

const RiskAdvisoryConsole = ({ initialModelConditions = null }: { initialModelConditions?: ModelConditions | null }) => {
  const [members, setMembers] = useState<Member[]>([]);
  const [cond, setCond] = useState<Conditions>({ aqi: 182, temp: 35, rain: 0, traffic: 'Severe' });
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [smsHistory, setSmsHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState('');
  const [error, setError] = useState('');
  const [toast, setToast] = useState('');

  const loadMembers = async () => {
    try {
      const response: any = await backendApi.listUsers();
      setMembers(response.data || []);
    } catch (err: any) {
      setError(err?.message || 'Could not load members.');
    }
  };

  const loadSmsHistory = async () => {
    try {
      const response: any = await backendApi.listSmsHistory(50);
      setSmsHistory(response.data || []);
    } catch (err: any) {
      setError(err?.message || 'Could not load SMS history.');
    }
  };

  useEffect(() => { loadMembers(); }, []);
  useEffect(() => { loadSmsHistory(); }, []);

  useEffect(() => {
    if (initialModelConditions) {
      setCond({
        aqi: Math.round(Number(initialModelConditions.aqi?.aqi || cond.aqi)),
        temp: Math.round(Number(initialModelConditions.weather?.temperature?.value || cond.temp)),
        rain: initialModelConditions.weather?.rainLastHourMm || 0,
        traffic: (initialModelConditions.traffic?.congestionLevel === 'heavy' || initialModelConditions.traffic?.congestionLevel === 'severe')
          ? 'Severe'
          : (initialModelConditions.traffic?.congestionLevel === 'moderate' ? 'Moderate' : 'Low'),
      });
    }
  }, [initialModelConditions]);

  const loadModelConditions = async () => {
    setLoading('model');
    setError('');
    try {
      const response: any = await backendApi.modelConditions('bangalore');
      const data = response.data;
      setCond({
        aqi: Math.round(Number(data.aqi?.aqi || 0)),
        temp: Math.round(Number(data.weather?.temperature?.value || 0)),
        rain: data.weather?.rainLastHourMm || 0,
        traffic: (data.traffic?.congestionLevel === 'heavy' || data.traffic?.congestionLevel === 'severe')
          ? 'Severe'
          : (data.traffic?.congestionLevel === 'moderate' ? 'Moderate' : 'Low'),
      });
    } catch (err: any) {
      setError(err?.message || 'Could not load model conditions.');
    } finally {
      setLoading('');
    }
  };

  const predict = () => {
    setPredictions(members.map((m) => computeScore(m, cond)));
  };

  useEffect(() => { predict(); }, [members, cond]);

  const removeMember = async (id: string) => {
    setLoading(id);
    setError('');
    try {
      await backendApi.deleteUser(id);
      setMembers((prev) => prev.filter((m) => m.id !== id));
    } catch (err: any) {
      setError(err?.message || 'Could not remove member.');
    } finally {
      setLoading('');
    }
  };

  const flagged = predictions.filter((p) => p.status.label === 'ALERT');

  const sendAlerts = async () => {
    setLoading('send');
    setError('');
    setToast('');
    try {
      let sent = 0;
      for (const p of flagged) {
        const channels = p.member.phone ? ['in_app', 'sms'] : ['in_app'];
        await backendApi.deliverNotification({
          user: { id: p.member.id, phone: p.member.phone },
          advisory: {
            userId: p.member.id,
            title: `Health advisory for ${p.member.name}`,
            message: draftMessage(p.member, p.drivers),
            severity: p.score >= 70 ? 'critical' : 'warning',
            riskScore: p.score,
            deliveryChannels: channels,
          },
          channels,
        });
        sent += 1;
      }
      setToast(`Alerts sent — ${sent} member${sent === 1 ? '' : 's'} notified by SMS + in-app.`);
      await loadSmsHistory();
    } catch (err: any) {
      setError(err?.message || 'Could not send alerts.');
    } finally {
      setLoading('');
    }
  };

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-[#1c3326] bg-[#0f1f16] p-6">
        <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.14em] text-[#4d6357]">
          <span className="h-1.5 w-1.5 rounded-full bg-[#4ade80] shadow-[0_0_8px_#4ade80]" />
          Concept — self-reported health profile, live predicted risk
        </div>
        <h1 className="mt-2 text-xl font-bold tracking-tight text-[#7ef7ba]">People tell you their risk factors once. The prediction does the rest.</h1>
        <p className="mt-1.5 max-w-[660px] text-[13.5px] leading-relaxed text-[#7b9686]">
          No more admin-assigned risk groups. Someone joins, answers a few health questions, and from then on every alert decision is computed straight from their profile against live conditions — not typed or guessed.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {/* LIVE CONDITIONS + PREDICTIONS */}
        <div className="rounded-2xl border border-[#1c3326] bg-[#0f1f16] p-6">
          <div className="mb-4 font-mono text-[11px] uppercase tracking-[0.08em] text-[#7b9686]">② Live conditions → predicted risk per member</div>

          <div className="flex flex-wrap items-end gap-3">
            {([['aqi', 'AQI', 'number'], ['temp', 'Temp °C', 'number'], ['rain', 'Rain mm', 'number']] as const).map(([key, label, type]) => (
              <div key={key} className="min-w-[100px] flex-1">
                <label className="mb-1.5 block text-[10.5px] uppercase tracking-[0.08em] text-[#4d6357]">{label}</label>
                <input
                  type={type}
                  value={cond[key]}
                  onChange={(e) => setCond({ ...cond, [key]: Number(e.target.value) || 0 })}
                  className="w-full rounded-lg border border-[#1c3326] bg-[#12261b] px-2.5 py-2 font-mono text-[13px] text-white outline-none focus:border-[#4ade80]/50"
                />
              </div>
            ))}
            <div className="min-w-[100px] flex-1">
              <label className="mb-1.5 block text-[10.5px] uppercase tracking-[0.08em] text-[#4d6357]">Traffic</label>
              <select
                value={cond.traffic}
                onChange={(e) => setCond({ ...cond, traffic: e.target.value as Conditions['traffic'] })}
                className="w-full rounded-lg border border-[#1c3326] bg-[#12261b] px-2.5 py-2 font-mono text-[13px] text-white outline-none focus:border-[#4ade80]/50"
              >
                <option>Low</option>
                <option>Moderate</option>
                <option>Severe</option>
              </select>
            </div>
            <button
              onClick={predict}
              className="whitespace-nowrap rounded-lg bg-[#7ef7ba] px-4 py-2.5 text-[12.5px] font-bold text-[#06150c]"
            >
              ▶ Predict now
            </button>
            <button
              onClick={loadModelConditions}
              disabled={loading === 'model'}
              className="flex items-center gap-1.5 whitespace-nowrap rounded-lg border border-[#2f6b45] px-4 py-2.5 text-[12px] font-bold text-[#7ef7ba] disabled:opacity-50"
            >
              {loading === 'model' ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
              Load model
            </button>
          </div>

          {error && (
            <div className="mt-4 rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-3 text-[12px] text-red-400">{error}</div>
          )}

          <div className="mt-5 overflow-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-[#1c3326] text-left font-mono text-[10.5px] uppercase tracking-[0.08em] text-[#4d6357]">
                  <th className="px-3 py-3">Member</th>
                  <th className="px-3 py-3">Health factors</th>
                  <th className="px-3 py-3">Predicted risk</th>
                  <th className="px-3 py-3">Status</th>
                  <th className="px-3 py-3" />
                </tr>
              </thead>
              <tbody>
                {predictions.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-3 py-9 text-center text-[12.5px] text-[#4d6357]">
                      <div className="mb-2 text-[28px] opacity-60">◌</div>
                      No members yet — add someone on the left to see their predicted risk here.
                    </td>
                  </tr>
                )}
                {predictions.map((p) => (
                  <tr key={p.member.id} className="border-b border-[#16281d]">
                    <td className="px-3 py-3">
                      <div className="flex items-center gap-2.5">
                        <div className="flex h-[30px] w-[30px] items-center justify-center rounded-full border border-[#1c3326] bg-[#12261b] text-[11px] font-bold text-[#4ade80]">
                          {initials(p.member.name)}
                        </div>
                        <div>
                          <div className="text-[13px] font-medium text-white">{p.member.name}</div>
                          <div className="text-[11px] text-[#4d6357]">{p.member.ward || 'Unknown ward'} · age {p.member.age ?? '-'}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-3 py-3">
                      {p.factors.map((f) => (
                        <span key={f} className="mb-0.5 mr-1 inline-block rounded border border-[#1c3326] bg-[#12261b] px-1.5 py-0.5 font-mono text-[10px] text-[#7b9686]">
                          {HEALTH_LABEL[f] || f}
                        </span>
                      ))}
                    </td>
                    <td className="px-3 py-3">
                      <div className="flex items-center gap-2">
                        <div className="h-[5px] w-16 overflow-hidden rounded bg-[#16281d]">
                          <div className="h-full rounded transition-all" style={{ width: `${p.score}%`, background: p.status.color }} />
                        </div>
                        <span className="min-w-[26px] font-mono text-[12.5px] font-bold" style={{ color: p.status.color }}>{p.score}</span>
                      </div>
                      {p.drivers.length > 0 && (
                        <div className="mt-1 font-mono text-[10.5px] text-[#4d6357]">
                          Driven by <b className="text-[#f0b849]">{p.drivers.join(', ')}</b>
                        </div>
                      )}
                    </td>
                    <td className="px-3 py-3">
                      <span
                        className="whitespace-nowrap rounded-full px-2.5 py-1 font-mono text-[10.5px] font-semibold"
                        style={{
                          color: p.status.color,
                          background: `${p.status.color}1f`,
                          border: `1px solid ${p.status.color}4d`,
                        }}
                      >
                        {p.status.label}
                      </span>
                    </td>
                    <td className="px-3 py-3 text-right">
                      <button
                        onClick={() => removeMember(p.member.id)}
                        disabled={loading === p.member.id}
                        className="flex h-6 w-6 items-center justify-center rounded-lg border border-[#1c3326] text-[13px] text-[#4d6357] transition-all hover:border-red-500/50 hover:bg-red-500/10 hover:text-[#f0665f]"
                        title="Remove member"
                      >
                        {loading === p.member.id ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-4 border-t border-[#16281d] pt-4">
            <div className="text-[13px] text-[#7b9686]">
              <b className="font-mono text-[16px] text-[#7ef7ba]">{flagged.length}</b> flagged for an advisory right now
            </div>
            <button
              onClick={sendAlerts}
              disabled={flagged.length === 0 || loading === 'send'}
              className="flex items-center gap-2 rounded-xl bg-[#7ef7ba] px-5 py-3 text-[13px] font-bold text-[#06150c] disabled:bg-[#1c3326] disabled:text-[#4d6357]"
            >
              {loading === 'send' ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
              {flagged.length ? `Send alert to ${flagged.length} flagged ${flagged.length === 1 ? 'member' : 'members'}` : 'Send alert to flagged members'}
            </button>
          </div>

          {flagged.length > 0 && (
            <div className="mt-4 rounded-xl border border-dashed border-[#1c3326] px-4 py-3.5">
              <div className="mb-2 font-mono text-[11px] uppercase tracking-[0.08em] text-[#7b9686]">Auto-drafted, per-person</div>
              {flagged.map((p) => (
                <div key={p.member.id} className="flex gap-2.5 border-b border-[#16281d] py-2 text-[12px] text-[#7b9686] last:border-b-0">
                  <b className="inline-block min-w-[100px] text-white">{p.member.name}</b>
                  {draftMessage(p.member, p.drivers)}
                </div>
              ))}
            </div>
          )}

          {toast && (
            <div className="mt-3 flex items-center gap-2 rounded-lg border border-[#2f6b45] bg-[rgba(74,222,128,0.08)] px-3.5 py-2.5 text-[12px] text-[#7ef7ba]">
              ✓ {toast}
            </div>
          )}
        </div>
      </div>

      {/* INBOX / SMS HISTORY */}
      <div className="rounded-2xl border border-[#1c3326] bg-[#0f1f16] p-6">
        <div className="mb-4 flex items-center justify-between">
          <div className="font-mono text-[11px] uppercase tracking-[0.08em] text-[#7b9686]">③ Inbox — SMS delivery history</div>
          <button
            onClick={loadSmsHistory}
            disabled={loading === 'sms'}
            className="flex items-center gap-1.5 rounded-lg border border-[#2f6b45] px-3 py-1.5 text-[11px] font-bold text-[#7ef7ba] disabled:opacity-50"
          >
            {loading === 'sms' ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
            Refresh
          </button>
        </div>

        {smsHistory.length === 0 ? (
          <div className="rounded-xl border border-dashed border-[#1c3326] px-4 py-9 text-center text-[12.5px] text-[#4d6357]">
            <div className="mb-2 text-[28px] opacity-60">📮</div>
            No SMS sent yet — trigger an alert above and real text messages go out via Twilio.
          </div>
        ) : (
          <div className="overflow-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-[#1c3326] text-left font-mono text-[10.5px] uppercase tracking-[0.08em] text-[#4d6357]">
                  <th className="px-3 py-3">Member</th>
                  <th className="px-3 py-3">Message</th>
                  <th className="px-3 py-3">Severity</th>
                  <th className="px-3 py-3">Status</th>
                  <th className="px-3 py-3">Sent at</th>
                </tr>
              </thead>
              <tbody>
                {smsHistory.map((sms) => (
                  <tr key={sms.id} className="border-b border-[#16281d]">
                    <td className="px-3 py-3">
                      <div className="text-[12.5px] font-medium text-white">{sms.member_name || 'Unknown'}</div>
                      <div className="font-mono text-[10.5px] text-[#4d6357]">{sms.member_phone || 'no phone'}</div>
                    </td>
                    <td className="max-w-[340px] px-3 py-3 text-[12px] leading-relaxed text-[#7b9686]">{sms.message}</td>
                    <td className="px-3 py-3">
                      <span
                        className={`rounded-full px-2.5 py-1 font-mono text-[10px] font-semibold ${
                          sms.severity === 'critical'
                            ? 'text-[#f0665f]'
                            : sms.severity === 'warning'
                              ? 'text-[#f0b849]'
                              : 'text-[#4ade80]'
                        }`}
                      >
                        {sms.severity || '-'}
                      </span>
                    </td>
                    <td className="px-3 py-3">
                      <span className="font-mono text-[11px] text-[#7b9686]">
                        {sms.delivery_status}
                        {sms.error_message ? (
                          <span className="block text-[10px] text-[#f0665f]">{sms.error_message}</span>
                        ) : null}
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-3 py-3 font-mono text-[10.5px] text-[#4d6357]">
                      {sms.sent_at ? new Date(sms.sent_at).toLocaleString() : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="rounded-2xl border border-[#1c3326] bg-[#0f1f16] p-6 text-[11.5px] leading-[1.7] text-[#4d6357]">
        <b className="text-[#7b9686]">Why this is different from admin-assigned groups:</b> "Elder" or "Respiratory" tags were guesses made on someone's behalf. Here the person declares asthma, a heart condition, pregnancy, mobility limits, or outdoor work themselves at sign-up — that's their consent and their data. The prediction engine reads directly from that profile plus live sensor values, so the risk score isn't manually recalculated by staff — it's generated the moment conditions change, and only crosses into an alert when it's actually warranted for that specific person.
      </div>
    </div>
  );
};

export default RiskAdvisoryConsole;
