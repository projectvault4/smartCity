import { useState, FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'motion/react';
import { useAuth, UserRole } from '../context/AuthContext';

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();

  const [role, setRole] = useState<UserRole>('admin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // Auto-fill demo credentials when role tab switches
  const handleRoleSwitch = (r: UserRole) => {
    setRole(r);
    setError('');
    setEmail(r === 'admin' ? 'admin@foresightx.city' : 'yashwanth@ward.in');
    setPassword(r === 'admin' ? 'admin123' : 'citizen123');
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    await new Promise((r) => setTimeout(r, 600)); // brief splash delay
    const ok = login(email, password, role);
    setLoading(false);
    if (ok) {
      navigate(role === 'admin' ? '/dashboard' : '/citizen', { replace: true });
    } else {
      setError('Invalid credentials. Check email & password.');
    }
  };

  const HINTS = {
    admin:   { email: 'admin@foresightx.city',  pass: 'admin123',   label: 'City Admin Portal' },
    citizen: { email: 'yashwanth@ward.in',       pass: 'citizen123', label: 'Citizen Portal' },
  };

  return (
    <div className="min-h-screen bg-[#0a1210] flex items-center justify-center p-4"
      style={{ background: 'radial-gradient(120% 100% at 15% -10%, rgba(111,231,183,0.07), transparent 55%), #0a1210' }}>

      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        {/* Brand */}
        <div className="flex items-center gap-3 mb-10">
          <div className="w-2.5 h-8 rounded-full bg-[#6fe7b7] shadow-[0_0_16px_rgba(111,231,183,0.6)]" />
          <div>
            <div className="font-display text-2xl font-extrabold tracking-wide text-white">
              Fore<span className="text-[#6fe7b7]">Sight</span>X
            </div>
            <div className="text-[10px] font-mono text-white/30 uppercase tracking-[3px] mt-0.5">
              Urban Intelligence Platform
            </div>
          </div>
        </div>

        {/* Card */}
        <div className="bg-[#10201a] border border-[#1f3831] rounded-2xl p-8 shadow-2xl">

          {/* Role tabs */}
          <div className="flex gap-2 mb-8 bg-[#0d1a10] rounded-xl p-1">
            {(['admin', 'citizen'] as UserRole[]).map((r) => (
              <button
                key={r}
                onClick={() => handleRoleSwitch(r)}
                className={`flex-1 py-2.5 rounded-lg text-[11px] font-black uppercase tracking-widest transition-all duration-200
                  ${role === r
                    ? 'bg-[#6fe7b7] text-[#0a1210] shadow-[0_0_20px_rgba(111,231,183,0.25)]'
                    : 'text-white/30 hover:text-white/60'}`}
              >
                {r === 'admin' ? '⚙ Admin' : '👤 Citizen'}
              </button>
            ))}
          </div>

          {/* Role label */}
          <AnimatePresence mode="wait">
            <motion.div
              key={role}
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.2 }}
              className="mb-6"
            >
              <div className="text-[#6fe7b7] text-xs font-mono uppercase tracking-[2px] mb-1">
                {HINTS[role].label}
              </div>
              <h1 className="text-white text-xl font-bold">
                {role === 'admin' ? 'Admin sign in' : 'Sign in to your ward'}
              </h1>
              <p className="text-white/40 text-sm mt-1">
                {role === 'admin'
                  ? 'Access city-wide forecasting, anomaly detection and controls.'
                  : 'View personalised alerts and forecasts for your ward.'}
              </p>
            </motion.div>
          </AnimatePresence>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-[10px] font-mono text-white/30 uppercase tracking-widest mb-2">
                Email
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder={HINTS[role].email}
                className="w-full bg-[#0d1a10] border border-[#1f3831] rounded-xl px-4 py-3 text-sm text-white placeholder-white/20
                  focus:outline-none focus:border-[#6fe7b7]/50 focus:ring-1 focus:ring-[#6fe7b7]/20 transition-all"
              />
            </div>

            <div>
              <label className="block text-[10px] font-mono text-white/30 uppercase tracking-widest mb-2">
                Password
              </label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full bg-[#0d1a10] border border-[#1f3831] rounded-xl px-4 py-3 text-sm text-white placeholder-white/20
                  focus:outline-none focus:border-[#6fe7b7]/50 focus:ring-1 focus:ring-[#6fe7b7]/20 transition-all"
              />
            </div>

            {/* Error */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3 text-red-400 text-sm"
                >
                  <span>⚠</span> {error}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-3.5 rounded-xl font-black text-sm uppercase tracking-widest transition-all duration-200
                bg-[#6fe7b7] text-[#0a1210] hover:bg-[#8ef0c8] shadow-[0_0_24px_rgba(111,231,183,0.2)]
                disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading
                ? <span className="inline-flex items-center gap-2">
                    <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                    </svg>
                    Signing in…
                  </span>
                : `Sign in as ${role === 'admin' ? 'Admin' : 'Citizen'} →`}
            </button>
          </form>

          {/* Demo hint */}
          <div className="mt-6 p-3 bg-[#0d1a10] rounded-xl border border-dashed border-[#1f3831]">
            <div className="text-[9px] font-mono text-white/25 uppercase tracking-[2px] mb-1.5">Demo credentials</div>
            <div className="text-[11px] font-mono text-white/40">
              {HINTS[role].email} / <span className="text-[#6fe7b7]/50">{HINTS[role].pass}</span>
            </div>
          </div>
        </div>

        <p className="text-center text-[10px] text-white/20 mt-6 font-mono">
          ForeSightX · RNS Institute of Technology · VTU
        </p>
      </motion.div>
    </div>
  );
}
