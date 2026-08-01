import { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import VoiceBriefing from '../components/VoiceBriefing';
import { CityData, ModelConditions } from '../services/dataService';

// ── Types ──────────────────────────────────────────────────────────────────────
type WardKey = 'channasandra' | 'indiranagar';
type UserKey = 'yashwanth' | 'meera' | 'farooq';
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

const ALL_TAGS = ['Elder', 'Child', 'Respiratory', 'Commuter', 'Worker'];

interface Alert {
  severity: Severity;
  headline: string;
  body: string;
  why: string;
  time: string;
}

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

const users: Record<UserKey, UserData> = {
  yashwanth: { name: 'Yashwanth M', ward: 'channasandra', age: 25, tags: ['Elder', 'Commuter'] },
  meera:     { name: 'Meera R',     ward: 'indiranagar',  age: 34, tags: ['Respiratory', 'Child'] },
  farooq:    { name: 'Farooq K',    ward: 'channasandra', age: 29, tags: ['Worker', 'Commuter'] },
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

  // Determine starting ward/user keys from logged-in user
  const defaultUserKey: UserKey =
    user?.name?.toLowerCase().includes('meera') ? 'meera'
    : user?.name?.toLowerCase().includes('farooq') ? 'farooq'
    : 'yashwanth';

  const [userKey, setUserKey] = useState<UserKey>(defaultUserKey);
  const [wardKey, setWardKey] = useState<WardKey>(users[defaultUserKey].ward);
  const [profile, setProfile] = useState<UserData>(users[defaultUserKey]);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<UserData>(users[defaultUserKey]);
  const [saved, setSaved] = useState(false);
  const [openWhy, setOpenWhy] = useState<number | null>(null);
  const [lastUpdated, setLastUpdated] = useState(9);

  // Tick "updated Xs ago"
  useEffect(() => {
    const id = setInterval(() => setLastUpdated((p) => p + 3), 3000);
    return () => clearInterval(id);
  }, []);

  const ward = wards[wardKey];
  const currentUser = profile;
  const alerts = buildAlerts(ward, currentUser);

  const handleUserChange = (uk: UserKey) => {
    setUserKey(uk);
    setWardKey(users[uk].ward);
    setProfile(users[uk]);
    setDraft(users[uk]);
    setEditing(false);
    setSaved(false);
    setOpenWhy(null);
  };

  const startEdit = () => {
    setDraft({ ...profile });
    setSaved(false);
    setEditing(true);
  };

  const cancelEdit = () => {
    setDraft({ ...profile });
    setEditing(false);
  };

  const saveProfile = () => {
    setProfile({ ...draft });
    if (draft.ward !== wardKey) setWardKey(draft.ward);
    setEditing(false);
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
            {/* Ward selector */}
            <div className="relative">
              <select
                value={wardKey}
                onChange={(e) => setWardKey(e.target.value as WardKey)}
                className="appearance-none bg-[#10201a] border border-[#1f3831] text-[#e9f3ee] font-mono text-[0.75rem] tracking-[0.03em] py-[9px] pl-3 pr-8 rounded-[20px] cursor-pointer focus:outline-none"
              >
                <option value="channasandra">Channasandra</option>
                <option value="indiranagar">Indiranagar</option>
              </select>
              <span className="absolute right-3 top-1/2 -translate-y-[55%] text-[#8fa69b] pointer-events-none text-[0.8rem]">⌄</span>
            </div>

            {/* User selector */}
            <div className="relative">
              <select
                value={userKey}
                onChange={(e) => handleUserChange(e.target.value as UserKey)}
                className="appearance-none bg-[#10201a] border border-[#1f3831] text-[#e9f3ee] font-mono text-[0.75rem] tracking-[0.03em] py-[9px] pl-3 pr-8 rounded-[20px] cursor-pointer focus:outline-none"
              >
                <option value="yashwanth">Yashwanth M</option>
                <option value="meera">Meera R</option>
                <option value="farooq">Farooq K</option>
              </select>
              <span className="absolute right-3 top-1/2 -translate-y-[55%] text-[#8fa69b] pointer-events-none text-[0.8rem]">⌄</span>
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
          key={wardKey + userKey}
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
            <div className="text-center py-10 px-5 text-[#5c7269] border border-dashed border-[#1f3831] rounded-[14px] text-[0.9rem]">
              Nothing flagged for your profile in {ward.name} right now. We'll let you know the moment that changes.
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
                <label className="block text-[0.68rem] tracking-[0.1em] uppercase text-[#5c7269] font-mono mb-2">Ward</label>
                <select
                  value={draft.ward}
                  onChange={(e) => setDraft({ ...draft, ward: e.target.value as WardKey })}
                  className="appearance-none w-full bg-[#0e1a16] border border-[#1f3831] text-[#e9f3ee] text-[0.85rem] px-3 py-2.5 rounded-[10px] focus:outline-none focus:border-[#3f8a71] cursor-pointer"
                >
                  <option value="channasandra">Channasandra</option>
                  <option value="indiranagar">Indiranagar</option>
                </select>
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
        <VoiceBriefing data={buildCityData(ward)} modelConditions={null} />

        <p className="text-center text-[#5c7269] text-[0.76rem] mt-9 font-['Inter',sans-serif]">
          Messages are generated for your profile and ward only — not broadcast city-wide.{' '}
          <span className="text-[#3f8a71]">Manage your tags anytime with your ward office.</span>
        </p>
      </div>
    </div>
  );
}
