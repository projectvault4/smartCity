import { useState, FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'motion/react';
import { CheckCircle2, Loader2, ShieldCheck, X } from 'lucide-react';
import { backendApi } from '../services/dataService';
import { useAuth } from '../context/AuthContext';
import { HealthKey, HEALTH_OPTIONS, autoRiskForAge, healthKeysToTags } from '../services/memberProfile';

export default function BecomeAMember() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [phone, setPhone] = useState('');
  const [ward, setWard] = useState('');
  const [health, setHealth] = useState<HealthKey[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [member, setMember] = useState<{ name: string; ward: string; age: number } | null>(null);

  const ageNumber = Number(age);
  const autoRisk = autoRiskForAge(ageNumber);
  const tags = healthKeysToTags(health, ageNumber);

  const toggleHealth = (key: HealthKey) => {
    setHealth((prev) => {
      if (key === 'none') return prev.includes('none') ? [] : ['none'];
      const withoutNone = prev.filter((item) => item !== 'none');
      return withoutNone.includes(key)
        ? withoutNone.filter((item) => item !== key)
        : [...withoutNone, key];
    });
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (!name.trim() || !ageNumber || !ward.trim()) {
      setError('Name, age and ward / area are required.');
      return;
    }

    if (ageNumber < 1 || ageNumber > 130) {
      setError('Enter a valid age (1–130).');
      return;
    }

    setLoading(true);

    try {
      const response: any = await backendApi.createUser({
        user_id: `member_${Date.now()}`,
        name: name.trim(),
        age: ageNumber,
        phone: phone.trim() || undefined,
        ward: ward.trim(),
        city: 'Bangalore',
        preferences: {
          source: 'member_signup',
          health_conditions: health,
          auto_risk_factors: autoRisk,
        },
        status: 'active',
      });

      setMember({
        name: response.data?.name || name.trim(),
        ward: response.data?.ward || ward.trim(),
        age: response.data?.age || ageNumber,
      });

      register({
        role: 'citizen',
        name: response.data?.name || name.trim(),
        ward: response.data?.ward || ward.trim(),
        age: response.data?.age || ageNumber,
        phone: response.data?.phone || phone.trim() || undefined,
        tags,
        email: response.data?.email || '',
      });
    } catch (err: any) {
      setError(err?.message || 'Could not create your profile. Try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="text-[#6fe7b7] text-xs font-mono uppercase tracking-[2px] mb-1">
        Member Sign-up
      </div>
      <h1 className="text-white text-xl font-bold">Become a member</h1>
      <p className="text-white/40 text-sm mt-1">
        Join ForeSightX so alerts are personalised to your health profile and ward.
      </p>

      <form onSubmit={handleSubmit} className="mt-6 space-y-4">
        <div>
          <label className="block text-[10px] font-mono text-white/30 uppercase tracking-widest mb-2">
            Name
          </label>
          <input
            type="text"
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Ananya Rao"
            className="w-full bg-[#0d1a10] border border-[#1f3831] rounded-xl px-4 py-3 text-sm text-white placeholder-white/20
              focus:outline-none focus:border-[#6fe7b7]/50 focus:ring-1 focus:ring-[#6fe7b7]/20 transition-all"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-[10px] font-mono text-white/30 uppercase tracking-widest mb-2">
              Age
            </label>
            <input
              type="number"
              min={1}
              max={130}
              required
              value={age}
              onChange={(e) => setAge(e.target.value)}
              placeholder="Age"
              className="w-full bg-[#0d1a10] border border-[#1f3831] rounded-xl px-4 py-3 text-sm text-white placeholder-white/20
                focus:outline-none focus:border-[#6fe7b7]/50 focus:ring-1 focus:ring-[#6fe7b7]/20 transition-all"
            />
          </div>
          <div>
            <label className="block text-[10px] font-mono text-white/30 uppercase tracking-widest mb-2">
              Phone
            </label>
            <input
              type="tel"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              placeholder="+91"
              className="w-full bg-[#0d1a10] border border-[#1f3831] rounded-xl px-4 py-3 text-sm text-white placeholder-white/20
                focus:outline-none focus:border-[#6fe7b7]/50 focus:ring-1 focus:ring-[#6fe7b7]/20 transition-all"
            />
          </div>
        </div>

        <div>
          <label className="block text-[10px] font-mono text-white/30 uppercase tracking-widest mb-2">
            Ward / Area
          </label>
          <input
            type="text"
            required
            value={ward}
            onChange={(e) => setWard(e.target.value)}
            placeholder="e.g. Channasandra"
            className="w-full bg-[#0d1a10] border border-[#1f3831] rounded-xl px-4 py-3 text-sm text-white placeholder-white/20
              focus:outline-none focus:border-[#6fe7b7]/50 focus:ring-1 focus:ring-[#6fe7b7]/20 transition-all"
          />
        </div>

        <div>
          <label className="block text-[10px] font-mono text-white/30 uppercase tracking-widest mb-2">
            Health conditions <span className="text-white/15">(select all that apply)</span>
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {HEALTH_OPTIONS.map((option) => {
              const active = health.includes(option.key);
              const isNone = option.key === 'none';
              const disabled = isNone && health.some((key) => key !== 'none');
              return (
                <button
                  key={option.key}
                  type="button"
                  onClick={() => toggleHealth(option.key)}
                  disabled={disabled}
                  className={`flex items-center gap-2 rounded-xl border px-3 py-2.5 text-left text-xs font-bold transition-all disabled:opacity-30
                    ${active
                      ? 'border-[#6fe7b7] bg-[rgba(111,231,183,0.1)] text-[#6fe7b7]'
                      : 'border-[#1f3831] bg-[#0d1a10] text-white/40 hover:border-[#6fe7b7]/30 hover:text-white/70'}`}
                >
                  <span className={`w-4 h-4 rounded-md border flex items-center justify-center text-[10px] ${active ? 'border-[#6fe7b7] bg-[#6fe7b7] text-[#0a1210]' : 'border-white/20'}`}>
                    {active && '✓'}
                  </span>
                  {option.label}
                </button>
              );
            })}
          </div>
        </div>

        {autoRisk.length > 0 && (
          <div className="rounded-xl border border-dashed border-[#6fe7b7]/40 bg-[rgba(111,231,183,0.05)] px-4 py-3 text-xs text-white/70">
            <span className="font-bold text-[#6fe7b7]">Auto-detected:</span>{' '}
            Age {ageNumber} adds the <span className="font-bold text-white">{autoRisk.join(' + ')}</span> risk factor automatically — no need to select it.
          </div>
        )}

        {error && (
          <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-400">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full py-3.5 rounded-xl font-black text-sm uppercase tracking-widest transition-all duration-200
            bg-[#6fe7b7] text-[#0a1210] hover:bg-[#8ef0c8] shadow-[0_0_24px_rgba(111,231,183,0.2)]
            disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <span className="inline-flex items-center gap-2"><Loader2 size={16} className="animate-spin" /> Adding…</span>
          ) : (
            '+ Add as member'
          )}
        </button>

        <p className="text-[10px] leading-relaxed text-white/25 font-mono">
          Age under 12 or over 60 is auto-detected as an added risk factor — no need to select it. This profile is what future predictions use, not a group you assign to them.
        </p>
      </form>

      <AnimatePresence>
        {member && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
          >
            <motion.div
              initial={{ scale: 0.9, y: 16 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ type: 'spring', damping: 24, stiffness: 260 }}
              className="w-full max-w-sm rounded-2xl border border-[#1f3831] bg-[#10201a] p-7 shadow-2xl"
            >
              <button
                onClick={() => navigate('/citizen', { replace: true })}
                className="absolute top-4 right-4 text-white/30 hover:text-white/70"
              >
                <X size={16} />
              </button>
              <div className="flex items-center gap-3 mb-4">
                <div className="flex h-11 w-11 items-center justify-center rounded-full bg-[#6fe7b7]/15 text-[#6fe7b7]">
                  <CheckCircle2 size={22} />
                </div>
                <div>
                  <div className="text-white font-bold">You're now a member of ForeSightX</div>
                  <div className="text-xs text-white/40 font-mono mt-0.5">Welcome aboard, {member.name}</div>
                </div>
              </div>
              <div className="rounded-xl border border-white/10 bg-[#0d1a10] p-4 space-y-2 text-xs">
                <div className="flex justify-between text-white/50"><span>Ward / Area</span><span className="font-bold text-white">{member.ward}</span></div>
                <div className="flex justify-between text-white/50"><span>Age</span><span className="font-bold text-white">{member.age}</span></div>
                <div className="flex justify-between text-white/50">
                  <span>Risk factors</span>
                  <span className="font-bold text-[#6fe7b7]">
                    {health.filter((key) => key !== 'none').map((key) => HEALTH_OPTIONS.find((option) => option.key === key)?.label).filter(Boolean).join(', ') || 'None selected'}
                    {autoRisk.length ? ` + ${autoRisk.join(', ')} (auto)` : ''}
                  </span>
                </div>
              </div>
              <div className="mt-4 flex items-center gap-2 text-[10px] font-mono text-[#6fe7b7]/60">
                <ShieldCheck size={14} />
                Your profile now powers your personalised alerts.
              </div>
              <button
                onClick={() => navigate('/citizen', { replace: true })}
                className="mt-5 w-full py-3 rounded-xl font-black text-xs uppercase tracking-widest bg-[#6fe7b7] text-[#0a1210] hover:bg-[#8ef0c8] transition-all"
              >
                Done
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
