import { useState, useEffect, useMemo } from 'react';
import { motion } from 'motion/react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import VoiceBriefing from '../components/VoiceBriefing';
import {
  CityData,
  ModelConditions,
  backendApi,
  cityDataFromModelConditions,
  getInitialData,
} from '../services/dataService';

// ── Types ──────────────────────────────────────────────────────────────────────
type WardKey = 'channasandra' | 'indiranagar';
type Severity = 'good' | 'warn' | 'bad';

interface WardData {
  name: string;
  traffic: { value: number; unit: string; level: string; pct: number };
  aqi: { value: number; level: string };
  temp: { value: number; level: string };
}

interface UserData {
  name: string;
  ward: WardKey;
  age: number;
  tags: string[];
}

const ALL_TAGS = ['Elder', 'Child', 'Respiratory', 'Commuter', 'Worker', 'Heart', 'Diabetes', 'Pregnant', 'Limited Mobility'];

interface Alert {
  severity: Severity;
  headline: string;
  body: string;
  why: string;
  time: string;
}

interface SmsAlert {
  id: string;
  title: string;
  message: string;
  severity: string;
  sentAt: string;
}

const formatSent = (iso: string) => {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  let h = d.getHours() % 12;
  if (h === 0) h = 12;
  const ampm = d.getHours() >= 12 ? 'PM' : 'AM';
  const m = String(d.getMinutes()).padStart(2, '0');
  return `${h}:${m} ${ampm}`;
};

// ── Mock data (same as HTML) ───────────────────────────────────────────────────
const wards: Record<WardKey, WardData> = {
  channasandra: {
    name: 'Channasandra',
    traffic: { value: 18780, unit: 'veh/hr', level: 'heavy', pct: 78 },
    aqi: { value: 158, level: 'very poor' },
    temp: { value: 31, level: 'high' },
  },
  indiranagar: {
    name: 'Indiranagar',
    traffic: { value: 9400, unit: 'veh/hr', level: 'moderate', pct: 22 },
    aqi: { value: 62, level: 'moderate' },
    temp: { value: 29, level: 'comfortable' },
  },
};

// ── Rule engine ───────────────────────────────────────────────────────────────
type Rule = {
  metric: 'aqi' | 'traffic' | 'temp';
  min?: number; max?: number;
  tags: string[];
  severity: Severity;
  headline: (w: WardData) => string;
  body: (w: WardData, u: UserData) => string;
  why: (w: WardData, u: UserData) => string;
};

const rules: Rule[] = [
  {
    metric: 'aqi', min: 120,
    tags: ['Respiratory', 'Elder', 'Child'],
    severity: 'bad',
    headline: (w) => `Air quality is very poor in ${w.name}`,
    body: (w, u) => `AQI is at ${w.aqi.value} right now. Given you're tagged ${matchedTags(u, ['Respiratory', 'Elder', 'Child'])}, it's worth limiting time outdoors today and keeping windows closed if you can.`,
    why: (w, u) => `Your profile includes <b>${matchedTags(u, ['Respiratory', 'Elder', 'Child'])}</b>. AQI above 120 is flagged as high-risk for these groups — a healthy adult with no tags wouldn't see this alert at the same AQI level.`,
  },
  {
    metric: 'aqi', min: 80, max: 119,
    tags: ['Respiratory', 'Elder', 'Child'],
    severity: 'warn',
    headline: (w) => `Air quality is dipping in ${w.name}`,
    body: (w, u) => `AQI is ${w.aqi.value} (moderate). Not urgent, but if you're heading out for a while, a short break indoors midday isn't a bad idea.`,
    why: (_w, u) => `Shown because you're tagged <b>${matchedTags(u, ['Respiratory', 'Elder', 'Child'])}</b> — moderate AQI is monitored more closely for these groups.`,
  },
  {
    metric: 'traffic', min: 60,
    tags: ['Commuter', 'Worker'],
    severity: 'warn',
    headline: (w) => `Heavy traffic building in ${w.name}`,
    body: (w, _u) => `Traffic is running ${w.traffic.pct}% above the usual baseline. If you're commuting in the next hour, budget extra time or check an alternate route.`,
    why: (_w, u) => `You're tagged <b>${matchedTags(u, ['Commuter', 'Worker'])}</b>, so traffic surges are surfaced for you directly.`,
  },
  {
    metric: 'temp', min: 30,
    tags: ['Elder', 'Child', 'Worker'],
    severity: 'warn',
    headline: (w) => `Temperature is climbing in ${w.name}`,
    body: (w, _u) => `It's ${w.temp.value}°C and rising. Stay hydrated, and if you can, shift outdoor tasks to earlier or later in the day.`,
    why: (_w, u) => `Heat advisories are prioritised for <b>${matchedTags(u, ['Elder', 'Child', 'Worker'])}</b> tags, since these groups are more affected by sustained heat.`,
  },
];

function matchedTags(user: UserData, ruleTags: string[]): string {
  return user.tags.filter((t) => ruleTags.includes(t)).join(' + ');
}

function metricValue(w: WardData, m: 'aqi' | 'traffic' | 'temp'): number {
  if (m === 'aqi') return w.aqi.value;
  if (m === 'traffic') return w.traffic.pct;
  return w.temp.value;
}

function buildAlerts(ward: WardData, user: UserData): Alert[] {
  const now = new Date();
  let h = now.getHours() % 12; if (h === 0) h = 12;
  const ampm = now.getHours() >= 12 ? 'PM' : 'AM';
  const m = String(now.getMinutes()).padStart(2, '0');
  const time = `${h}:${m} ${ampm}`;

  return rules.reduce<Alert[]>((out, r) => {
    const val = metricValue(ward, r.metric);
    const min = r.min ?? -Infinity;
    const max = r.max ?? Infinity;
    const tagMatch = user.tags.some((t) => r.tags.includes(t));
    if (val >= min && val <= max && tagMatch) {
      out.push({ severity: r.severity, headline: r.headline(ward), body: r.body(ward, user), why: r.why(ward, user), time });
    }
    return out;
  }, []);
}

function buildCityData(ward: WardData): CityData {
  return {
    traffic: { value: ward.traffic.value, label: ward.traffic.level, range: [0, 1], delta: '', trend: 'neutral', history: [], unit: 'vehicles/hr' },
    air: { value: ward.aqi.value, label: ward.aqi.level, range: [0, 1], delta: '', trend: 'neutral', history: [], unit: 'AQI' },
    weather: { value: ward.temp.value, label: ward.temp.level, range: [0, 1], delta: '', trend: 'neutral', history: [], unit: '°C' },
    energy: { value: 847, label: 'High', range: [0, 1], delta: '', trend: 'neutral', history: [], unit: 'MW' },
    timestamp: new Date().toLocaleString(),
    lastUpdate: '',
  };
}

const levelTagClass: Record<string, string> = {
  heavy: 'bad', 'very poor': 'bad', high: 'warn', moderate: 'warn', comfortable: 'good',
  poor: 'warn', severe: 'bad', hot: 'warn', heatwave: 'bad', warm: 'warn', cool: 'good', cold: 'warn',
};
const tagBg: Record<string, string> = {
  good: 'bg-[rgba(111,231,183,0.12)] text-[#6fe7b7]',
  warn: 'bg-[rgba(240,168,87,0.14)] text-[#f0a857]',
  bad:  'bg-[rgba(242,102,122,0.14)] text-[#f2667a]',
};
const alertBorder: Record<Severity, string> = {
  good: 'border-l-[#3f8a71]',
  warn: 'border-l-[#f0a857]',
  bad:  'border-l-[#f2667a]',
};

// ── Component ──────────────────────────────────────────────────────────────────
export default function CitizenDashboard() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  // Build the member profile from the signed-in user (no mock fallback)
  const memberTags = user?.tags?.length ? user.tags : ['Commuter'];

  const [wardKey] = useState<WardKey>(() => {
    const wardMatch = Object.keys(wards).find((w) =>
      user?.ward?.toLowerCase().includes(w.toLowerCase())
    ) as WardKey | undefined;
    return wardMatch || 'channasandra';
  });
  const [profile, setProfile] = useState<UserData>(() => ({
    name: user?.name || 'Member',
    ward: wardKey,
    age: user?.age || 30,
    tags: memberTags,
  }));
  const [draft, setDraft] = useState<UserData>(() => ({
    name: user?.name || 'Member',
    ward: wardKey,
    age: user?.age || 30,
    tags: memberTags,
  }));
  const [editing, setEditing] = useState(false);
  const [saved, setSaved] = useState(false);
  const [openWhy, setOpenWhy] = useState<number | null>(null);
  const [lastUpdated, setLastUpdated] = useState(0);
  const [modelConditions, setModelConditions] = useState<ModelConditions | null>(null);
  const [smsAlerts, setSmsAlerts] = useState<SmsAlert[]>([]);
  const [dataFetchedAt, setDataFetchedAt] = useState<number | null>(null);

  // Tick "updated Xs ago" — real elapsed time since the last live fetch
  useEffect(() => {
    if (!dataFetchedAt) return;
    const id = setInterval(() => {
      setLastUpdated(Math.max(0, Math.floor((Date.now() - dataFetchedAt) / 1000)));
    }, 1000);
    return () => clearInterval(id);
  }, [dataFetchedAt]);

  // Pull live model conditions + this member's SMS history
  useEffect(() => {
    let cancelled = false;
    const fetchLive = async () => {
      try {
        const [condRes, smsRes] = await Promise.all([
          backendApi.modelConditions('bangalore', 1),
          backendApi.listSmsHistory(50) as Promise<{ data: any[] }>,
        ]);
        if (cancelled) return;
        setModelConditions(condRes.data);
        setDataFetchedAt(Date.now());
        const phoneDigits = (user?.phone || '').replace(/\D/g, '').slice(-10);
        setSmsAlerts((smsRes.data || [])
          .filter((r: any) => String(r.member_phone || '').replace(/\D/g, '').slice(-10) === phoneDigits)
          .map((r: any) => ({
            id: r.id,
            title: r.title,
            message: r.message,
            severity: r.severity,
            sentAt: r.sent_at,
          })));
      } catch {
        // keep last known values — dashboard stays usable offline
      }
    };
    fetchLive();
    const id = setInterval(fetchLive, 60000);
    return () => { cancelled = true; clearInterval(id); };
  }, [user?.phone]);

  const baseWard = wards[wardKey];
  // Merge live trained-model conditions into the ward's snapshot, with a mock fallback.
  const ward = useMemo<WardData>(() => {
    if (!modelConditions) return baseWard;
    const trafficFlow = Math.round(Number(modelConditions.traffic?.flow ?? baseWard.traffic.value));
    const aqi = Math.round(Number(modelConditions.aqi?.aqi ?? baseWard.aqi.value));
    const temp = Math.round(Number(modelConditions.weather?.temperature?.value ?? baseWard.temp.value));
    const baseline = 8000;
    const pct = Math.max(0, Math.round(((trafficFlow - baseline) / baseline) * 100));
    const aqiLabel = aqi <= 50 ? 'good' : aqi <= 100 ? 'moderate' : aqi <= 200 ? 'poor' : aqi <= 300 ? 'very poor' : 'severe';
    const tempLabel = temp >= 38 ? 'heatwave' : temp >= 32 ? 'hot' : temp >= 24 ? 'warm' : temp >= 15 ? 'cool' : 'cold';
    return {
      name: baseWard.name,
      traffic: { value: trafficFlow, unit: 'veh/hr', level: modelConditions.traffic?.congestionLevel || 'moderate', pct },
      aqi: { value: aqi, level: aqiLabel },
      temp: { value: temp, level: tempLabel },
    };
  }, [baseWard, modelConditions]);
  const currentUser = profile;
  const alerts = buildAlerts(ward, currentUser);

  // Plain-language summary of the live conditions, used when nothing is flagged
  const conditionExplainer: Array<{ metric: string; line: string }> = (() => {
    const lines: Array<{ metric: string; line: string }> = [];
    const aqi = ward.aqi.value;
    lines.push({
      metric: 'Air quality',
      line: aqi >= 120
        ? `AQI is ${aqi} (${ward.aqi.level}) — very poor air. This can bother sensitive groups like people with respiratory issues, the elderly and children.`
        : aqi >= 80
          ? `AQI is ${aqi} (${ward.aqi.level}). Not urgent, but a bit of care is wise for sensitive groups.`
          : `AQI is ${aqi} (${ward.aqi.level}) — clean enough that we have nothing to warn you about.`,
    });
    const traffic = ward.traffic.value;
    lines.push({
      metric: 'Traffic',
      line: ward.traffic.pct >= 60
        ? `Traffic is ${traffic.toLocaleString()} vehicles/hr and surging about ${ward.traffic.pct}% above the usual baseline — expect slower commutes.`
        : `Traffic is about ${traffic.toLocaleString()} vehicles/hr (${ward.traffic.level}) — a normal level, so no delay warning.`,
    });
    const temp = ward.temp.value;
    lines.push({
      metric: 'Temperature',
      line: temp >= 30
        ? `It's ${temp}°C (${ward.temp.level}) and climbing. Heat at this level can affect the elderly, children and outdoor workers.`
        : `It's ${temp}°C (${ward.temp.level}) — below the 30°C heat alert, so nothing to flag.`,
    });
    return lines;
  })();

  // Which alert categories actually matter for this member's tags
  const watchedForYou = (() => {
    const map: Record<string, string[]> = {
      'Air quality': ['Respiratory', 'Elder', 'Child'],
      Traffic: ['Commuter', 'Worker'],
      Temperature: ['Elder', 'Child', 'Worker'],
    };
    return (Object.entries(map) as Array<[string, string[]]>)
      .filter(([, tags]) => currentUser.tags.some((t) => tags.includes(t)))
      .map(([metric]) => metric);
  })();

  const startEdit = () => {
    setDraft({ ...profile });
    setSaved(false);
    setEditing(true);
  };

  const cancelEdit = () => {
    setDraft({ ...profile });
    setEditing(false);
  };

  const saveProfile = async () => {
    setEditing(false);
    try {
      const phone = user?.phone || '';
      const wardName = wardKey === 'indiranagar' ? 'Indiranagar' : 'Channasandra';
      const payload = { name: draft.name.trim(), age: draft.age, ward: wardName, preferences: { tags: draft.tags } };
      // Resolve the backend user by phone so admin sees the update.
      const list = (await backendApi.listUsers()) as any;
      const users: any[] = Array.isArray(list) ? list : (list?.data || []);
      const phoneDigits = phone.replace(/\D/g, '').slice(-10);
      const existing = users.find((u: any) => String(u.phone || '').replace(/\D/g, '').slice(-10) === phoneDigits);
      if (existing) {
        await backendApi.updateUser(existing.id, payload);
      } else if (phoneDigits) {
        await backendApi.createUser({
          user_id: `citizen_${Date.now()}`,
          name: draft.name.trim(),
          age: draft.age,
          phone,
          city: 'bangalore',
          ward: wardName,
          status: 'active',
          preferences: { tags: draft.tags },
        });
      }
    } catch {
      // keep local state even if the backend is unreachable
    }
    setProfile({ ...draft });
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  };

  const toggleTag = (tag: string) => {
    setDraft((d) => ({
      ...d,
      tags: d.tags.includes(tag) ? d.tags.filter((t) => t !== tag) : [...d.tags, tag],
    }));
  };

  const handleLogout = () => { logout(); navigate('/login', { replace: true }); };

  const heroLine =
    alerts.length === 0 ? `All clear in ${ward.name} for you today.`
    : alerts.length === 1 ? 'One thing worth knowing today.'
    : `${alerts.length} things worth knowing today.`;

  return (
    <div
      className="min-h-screen text-[#e9f3ee] font-['Inter',sans-serif]"
      style={{
        background: 'radial-gradient(120% 100% at 15% -10%, rgba(111,231,183,0.07), transparent 55%), #0a1210',
        paddingBottom: 80,
      }}
    >
      <div className="max-w-[880px] mx-auto px-5 pt-8">

        {/* ── Top bar ─────────────────────────────────────────────── */}
        <div className="flex justify-between items-start gap-4 flex-wrap mb-[34px]">
          <div className="flex items-center gap-[9px]">
            <div className="w-[9px] h-[9px] rounded-full bg-[#6fe7b7] shadow-[0_0_10px_#6fe7b7]" />
            <div>
              <span className="font-['Fraunces',serif] font-semibold text-[1.15rem]">ForeSightX</span>
              <small className="block font-mono text-[0.63rem] text-[#5c7269] tracking-[0.14em] uppercase mt-[1px]">
                Citizen Alerts
              </small>
            </div>
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            {/* Fixed ward chip — area comes from registration, not a dropdown */}
            <div className="bg-[#10201a] border border-[#1f3831] rounded-[20px] px-4 py-[9px] font-mono text-[0.75rem] text-[#8fa69b] tracking-[0.03em]">
              <span className="text-[#6fe7b7]">{ward.name}</span>
            </div>

            {/* Signed-in member chip */}
            <div className="bg-[#10201a] border border-[#1f3831] rounded-[20px] px-4 py-[9px] font-mono text-[0.75rem] text-[#8fa69b] tracking-[0.03em]">
              <span className="text-[#6fe7b7]">{profile.name}</span>
              {user?.phone && <span className="text-[#5c7269]"> · {user.phone}</span>}
            </div>

            {/* Logout */}
            <button
              onClick={handleLogout}
              className="font-mono text-[0.7rem] tracking-widest uppercase py-[9px] px-4 rounded-[20px] border border-[#1f3831] text-[#5c7269] hover:border-[#f2667a]/40 hover:text-[#f2667a] transition-all"
            >
              Sign out
            </button>
          </div>
        </div>

        {/* ── Hero ────────────────────────────────────────────────── */}
        <motion.div
          key={wardKey + profile.name}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="bg-gradient-to-br from-[#142720] to-[#10201a] border border-[#1f3831] rounded-[20px] p-[34px_30px] mb-7"
        >
          <div className="font-mono text-[0.68rem] tracking-[0.16em] uppercase text-[#6fe7b7] mb-[14px] flex items-center gap-2">
            <span className="w-[5px] h-[5px] rounded-full bg-[#6fe7b7] inline-block" />
            {ward.name} · Updated {lastUpdated}s ago
          </div>
          <h1 className="font-['Fraunces',serif] font-normal italic text-[clamp(1.6rem,4vw,2.3rem)] leading-[1.15] m-0 mb-[10px]">
            {heroLine}
          </h1>
          <p className="text-[#8fa69b] text-[0.95rem] max-w-[56ch] mb-[26px]">
            A quick read of what's happening around you right now — no dashboards, no jargon.
          </p>

          {/* Snapshot metrics */}
          <div className="grid grid-cols-[repeat(auto-fit,minmax(130px,1fr))] gap-[14px]">
            {[
              { label: 'Traffic', value: ward.traffic.value.toLocaleString(), unit: 'veh/hr', level: ward.traffic.level },
              { label: 'Air quality', value: String(ward.aqi.value), unit: 'AQI', level: ward.aqi.level },
              { label: 'Temperature', value: `${ward.temp.value}°`, unit: 'C', level: ward.temp.level },
            ].map((m) => (
              <div key={m.label} className="bg-[#0e1a16] border border-[#1f3831] rounded-[12px] p-[14px_16px]">
                <div className="font-mono text-[0.62rem] tracking-[0.1em] uppercase text-[#5c7269] mb-2">{m.label}</div>
                <div className="font-['Fraunces',serif] text-[1.5rem] font-medium">
                  {m.value}<span className="text-[0.85rem] text-[#8fa69b] font-['Inter',sans-serif] font-normal ml-[3px]">{m.unit}</span>
                </div>
                <span className={`inline-block mt-[6px] text-[0.68rem] px-2 py-[2px] rounded-[20px] font-mono ${tagBg[levelTagClass[m.level] ?? 'good']}`}>
                  {m.level}
                </span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* ── Alerts ──────────────────────────────────────────────── */}
        <div className="flex items-baseline justify-between mb-4">
          <h2 className="font-['Fraunces',serif] font-medium text-[1.15rem] m-0">Your alerts</h2>
          <span className="font-mono text-[0.72rem] text-[#5c7269]">{alerts.length} today</span>
        </div>

        <div className="flex flex-col gap-3 mb-9">
          {alerts.length === 0 ? (
            <div className="py-8 px-5 text-[#5c7269] border border-dashed border-[#1f3831] rounded-[14px]">
              <div className="text-center">
                <div className="text-[0.9rem]">Nothing flagged for your profile in {ward.name} right now.</div>
                <div className="text-[0.8rem] mt-1">We'll let you know the moment that changes.</div>
              </div>

              <details className="mt-5 rounded-[12px] border border-[#1f3831] bg-[#0e1a16] p-[16px_18px]">
                <summary className="cursor-pointer inline-flex items-center gap-[6px] font-mono text-[0.72rem] tracking-[0.04em] text-[#6fe7b7]">
                  Why am I seeing this? <span>⌄</span>
                </summary>
                <div className="mt-4 space-y-4 text-[0.85rem] leading-relaxed text-[#8fa69b]">
                  <div>
                    <div className="font-semibold text-[#e9f3ee] mb-1">What's happening right now in {ward.name}:</div>
                    <ul className="list-disc pl-5 space-y-1">
                      {conditionExplainer.map((c) => (
                        <li key={c.metric}>
                          <span className="font-semibold text-[#6fe7b7]">{c.metric}:</span> {c.line}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <div className="font-semibold text-[#e9f3ee] mb-1">Why it matters for you:</div>
                    <p>
                      You're tagged <span className="text-[#6fe7b7]">{currentUser.tags.join(' + ') || 'no tags'}</span>.
                      {watchedForYou.length
                        ? ` We keep a closer eye on ${watchedForYou.join(' and ')} for people with your profile. None of those are at a worrying level right now.`
                        : ' None of the alert categories specifically match your tags today, so nothing is flagged.'}
                    </p>
                  </div>
                </div>
              </details>
            </div>
          ) : (
            alerts.map((a, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.07 }}
                className={`bg-[#10201a] border border-[#1f3831] border-l-[3px] rounded-[12px] p-[18px_20px] ${alertBorder[a.severity]}`}
              >
                <div className="flex justify-between gap-[14px] items-start">
                  <p className="font-semibold text-[0.96rem] m-0 mb-[6px]">{a.headline}</p>
                  <span className="font-mono text-[0.68rem] text-[#5c7269] whitespace-nowrap mt-[2px]">{a.time}</span>
                </div>
                <p className="text-[#8fa69b] text-[0.88rem] m-0 mb-3">{a.body}</p>
                <button
                  onClick={() => setOpenWhy(openWhy === i ? null : i)}
                  className="bg-transparent border border-[#1f3831] text-[#8fa69b] font-mono text-[0.68rem] tracking-[0.04em] px-[10px] py-[5px] rounded-[20px] cursor-pointer inline-flex items-center gap-[6px] hover:border-[#3f8a71] hover:text-[#6fe7b7] transition-all"
                >
                  Why am I seeing this? <span>{openWhy === i ? '⌃' : '⌄'}</span>
                </button>
                {openWhy === i && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-3 p-[12px_14px] bg-[#0e1a16] rounded-[10px] text-[0.82rem] text-[#8fa69b] border border-dashed border-[#1f3831]"
                    dangerouslySetInnerHTML={{ __html: a.why }}
                  />
                )}
              </motion.div>
            ))
          )}
        </div>

        {/* ── SMS alerts (delivered) ─────────────────────────────── */}
        <div className="flex items-baseline justify-between mb-4">
          <h2 className="font-['Fraunces',serif] font-medium text-[1.15rem] m-0">SMS alerts</h2>
          <span className="font-mono text-[0.72rem] text-[#5c7269]">{smsAlerts.length} delivered</span>
        </div>

        <div className="flex flex-col gap-3 mb-9">
          {smsAlerts.length === 0 ? (
            <div className="text-center py-10 px-5 text-[#5c7269] border border-dashed border-[#1f3831] rounded-[14px] text-[0.9rem]">
              No SMS alerts have been sent to your phone yet.
            </div>
          ) : (
            smsAlerts.map((s) => (
              <motion.div
                key={s.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-[#10201a] border border-[#1f3831] border-l-[3px] rounded-[12px] p-[18px_20px]"
              >
                <div className="flex justify-between gap-[14px] items-start">
                  <p className="font-semibold text-[0.96rem] m-0 mb-[6px]">{s.title}</p>
                  <span className="font-mono text-[0.68rem] text-[#5c7269] whitespace-nowrap mt-[2px]">{formatSent(s.sentAt)}</span>
                </div>
                <p className="text-[#8fa69b] text-[0.88rem] m-0 mb-3">{s.message}</p>
                <span className={`inline-block text-[0.68rem] px-2 py-[2px] rounded-[20px] font-mono ${s.severity === 'critical' ? tagBg.bad : tagBg.warn}`}>
                  {s.severity === 'critical' ? 'Critical · SMS sent' : 'Advisory · SMS sent'}
                </span>
              </motion.div>
            ))
          )}
        </div>

        {/* ── Profile card ────────────────────────────────────────── */}
        <div className="flex items-baseline justify-between mb-4">
          <h2 className="font-['Fraunces',serif] font-medium text-[1.15rem] m-0">Your profile</h2>
          {!editing && (
            <button
              onClick={startEdit}
              className="font-mono text-[0.7rem] tracking-widest uppercase py-[7px] px-4 rounded-[20px] border border-[#1f3831] text-[#8fa69b] hover:border-[#3f8a71] hover:text-[#6fe7b7] transition-all"
            >
              {saved ? '✓ Saved' : 'Edit profile'}
            </button>
          )}
        </div>
        <div className="bg-[#10201a] border border-[#1f3831] rounded-[16px] p-[26px_26px_22px] mb-5">
          {editing ? (
            /* ── Edit form ─────────────────────────────────────── */
            <div className="space-y-5">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-[0.68rem] tracking-[0.1em] uppercase text-[#5c7269] font-mono mb-2">Full name</label>
                  <input
                    type="text"
                    value={draft.name}
                    onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                    className="w-full bg-[#0e1a16] border border-[#1f3831] text-[#e9f3ee] text-[0.85rem] px-3 py-2.5 rounded-[10px] focus:outline-none focus:border-[#3f8a71]"
                  />
                </div>
                <div>
                  <label className="block text-[0.68rem] tracking-[0.1em] uppercase text-[#5c7269] font-mono mb-2">Age</label>
                  <input
                    type="number"
                    min={1}
                    max={120}
                    value={draft.age}
                    onChange={(e) => setDraft({ ...draft, age: Math.max(1, Math.min(120, Number(e.target.value) || 1)) })}
                    className="w-full bg-[#0e1a16] border border-[#1f3831] text-[#e9f3ee] text-[0.85rem] px-3 py-2.5 rounded-[10px] focus:outline-none focus:border-[#3f8a71]"
                  />
                </div>
              </div>

              <div>
                <label className="block text-[0.68rem] tracking-[0.1em] uppercase text-[#5c7269] font-mono mb-2">Ward / Area</label>
                <div className="bg-[#0e1a16] border border-[#1f3831] text-[#8fa69b] text-[0.85rem] px-3 py-2.5 rounded-[10px]">
                  {ward.name}
                </div>
              </div>

              <div>
                <label className="block text-[0.68rem] tracking-[0.1em] uppercase text-[#5c7269] font-mono mb-2">Tags</label>
                <div className="flex flex-wrap gap-2">
                  {ALL_TAGS.map((t) => {
                    const active = draft.tags.includes(t);
                    return (
                      <button
                        key={t}
                        type="button"
                        onClick={() => toggleTag(t)}
                        className={`text-[0.74rem] px-[13px] py-[6px] rounded-[20px] font-mono border transition-all cursor-pointer ${
                          active
                            ? 'border-[#3f8a71] bg-[rgba(111,231,183,0.12)] text-[#6fe7b7]'
                            : 'border-[#1f3831] bg-transparent text-[#5c7269] hover:text-[#8fa69b]'
                        }`}
                      >
                        {active ? '✓ ' : ''}{t}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="flex items-center gap-3 pt-1">
                <button
                  onClick={saveProfile}
                  className="font-mono text-[0.72rem] tracking-widest uppercase py-[9px] px-6 rounded-[20px] bg-[#6fe7b7] text-[#0a1210] hover:bg-[#8ef0c8] transition-all"
                >
                  Save changes
                </button>
                <button
                  onClick={cancelEdit}
                  className="font-mono text-[0.72rem] tracking-widest uppercase py-[9px] px-6 rounded-[20px] border border-[#1f3831] text-[#5c7269] hover:text-[#e9f3ee] transition-all"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            /* ── Read-only view ─────────────────────────────────── */
            <>
              <div className="flex justify-between items-start flex-wrap gap-[14px] mb-[18px]">
                <div>
                  <div className="font-['Fraunces',serif] text-[1.3rem]">{currentUser.name}</div>
                  <div className="text-[#5c7269] text-[0.8rem] font-mono mt-1">{ward.name} · Age {currentUser.age}</div>
                  {user?.phone && (
                    <div className="text-[#5c7269] text-[0.8rem] font-mono mt-1">{user.phone}</div>
                  )}
                </div>
              </div>

              <div className="text-[0.68rem] tracking-[0.1em] uppercase text-[#5c7269] font-mono mt-[18px] mb-2">You're tagged as</div>
              <div className="flex flex-wrap gap-2">
                {currentUser.tags.map((t) => (
                  <span key={t} className="text-[0.74rem] px-[13px] py-[6px] rounded-[20px] border border-[#3f8a71] bg-[rgba(111,231,183,0.12)] text-[#6fe7b7] font-mono">
                    {t}
                  </span>
                ))}
              </div>

              <div className="text-[0.68rem] tracking-[0.1em] uppercase text-[#5c7269] font-mono mt-[18px] mb-2">Alerts apply to</div>
              <div className="flex flex-wrap gap-2">
                {['Air quality', 'Traffic', 'Temperature'].map((t) => (
                  <span key={t} className="text-[0.74rem] px-[13px] py-[6px] rounded-[20px] border border-[#3f8a71] bg-[rgba(111,231,183,0.12)] text-[#6fe7b7] font-mono">
                    {t}
                  </span>
                ))}
              </div>
            </>
          )}
        </div>

        {/* ── Voice briefing ─────────────────────────────────────────── */}
        <div className="flex items-baseline justify-between mb-4">
          <h2 className="font-['Fraunces',serif] font-medium text-[1.15rem] m-0">Voice briefing</h2>
          <span className="font-mono text-[0.72rem] text-[#5c7269]">{ward.name}</span>
        </div>
        <VoiceBriefing
          data={modelConditions ? cityDataFromModelConditions(modelConditions, getInitialData()) : buildCityData(ward)}
          modelConditions={modelConditions}
        />

        <p className="text-center text-[#5c7269] text-[0.76rem] mt-9 font-['Inter',sans-serif]">
          Messages are generated for your profile and ward only — not broadcast city-wide.{' '}
          <span className="text-[#3f8a71]">Manage your tags anytime with your ward office.</span>
        </p>
      </div>
    </div>
  );
}
