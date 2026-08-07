import { useState, FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2 } from 'lucide-react';
import { backendApi } from '../services/dataService';
import { useAuth } from '../context/AuthContext';
import { healthKeysToTags, HealthKey } from '../services/memberProfile';

const normalizePhone = (value: string) => {
  const digits = value.trim().replace(/[\s().-]/g, '').replace(/^\+/, '');
  return digits.length > 10 && digits.startsWith('91') ? digits.slice(2) : digits;
};

export default function MemberSignIn() {
  const { register } = useAuth();
  const navigate = useNavigate();

  const [phone, setPhone] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    const query = normalizePhone(phone);
    if (!query) {
      setError('Enter your registered phone number.');
      return;
    }

    setLoading(true);

    try {
      const response: any = await backendApi.listUsers();
      const users: any[] = response.data || [];

      const match = users.find((u) => normalizePhone(u.phone || '') === query);

      if (!match) {
        setError('No member found with that phone number. Become a member first.');
        return;
      }

      const healthKeys: HealthKey[] = Array.isArray(match.preferences?.health_conditions)
        ? match.preferences.health_conditions
        : [];
      const tags = healthKeysToTags(healthKeys, Number(match.age) || undefined);

      register({
        role: 'citizen',
        name: match.name,
        ward: match.ward || 'Channasandra',
        age: match.age || undefined,
        phone: match.phone || undefined,
        tags,
        email: match.email || '',
      });

      navigate('/citizen', { replace: true });
    } catch (err: any) {
      setError(err?.message || 'Could not verify your membership. Try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="text-[#6fe7b7] text-xs font-mono uppercase tracking-[2px] mb-1">
        Member Sign-in
      </div>
      <h1 className="text-white text-xl font-bold">Sign in to your ward</h1>
      <p className="text-white/40 text-sm mt-1">
        View personalised alerts and forecasts for your ward.
      </p>

      <form onSubmit={handleSubmit} className="mt-6 space-y-4">
        <div>
          <label className="block text-[10px] font-mono text-white/30 uppercase tracking-widest mb-2">
            Phone number
          </label>
          <input
            type="tel"
            required
            value={phone}
            onChange={(e) => setPhone(e.target.value)}
            placeholder="+91 98765 43210"
            className="w-full bg-[#0d1a10] border border-[#1f3831] rounded-xl px-4 py-3 text-sm text-white placeholder-white/20
              focus:outline-none focus:border-[#6fe7b7]/50 focus:ring-1 focus:ring-[#6fe7b7]/20 transition-all"
          />
        </div>

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
            <span className="inline-flex items-center gap-2"><Loader2 size={16} className="animate-spin" /> Verifying…</span>
          ) : (
            'Sign in as Member →'
          )}
        </button>

        <p className="text-[10px] leading-relaxed text-white/25 font-mono">
          Members sign in with the phone number they registered with.
        </p>
      </form>
    </div>
  );
}
