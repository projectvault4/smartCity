import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Menu, X, LogOut } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  onLogout: () => void;
}

const navItems = [
  { id: 'home', label: 'City Pulse', color: 'bg-home-acc' },
  { id: 'simulation', label: 'Ripple Sim', color: 'bg-traf-acc' },
  { id: 'prediction', label: 'Forecasting', color: 'bg-air-acc' },
  { id: 'xai', label: 'Explainable AI', color: 'bg-home-acc' },
  { id: 'analytics', label: 'Multi-Domain Analytics', color: 'bg-eng-acc' },
  { id: 'anomaly', label: 'Anomaly Detection', color: 'bg-red-400' },
  { id: 'voice', label: 'Voice Briefing', color: 'bg-traf-acc' },
  { id: 'advisories', label: 'Advisories & Users', color: 'bg-home-acc' },
  { id: 'model', label: 'Model Quality', color: 'bg-white' },
];

const Sidebar = ({ activeTab, setActiveTab, onLogout }: SidebarProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const { user } = useAuth();
  const cityModes = ['traffic', 'air', 'energy', 'weather', 'home'];

  return (
    <>
      {/* MOBILE TOGGLE */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed top-6 right-6 z-[10000] lg:hidden p-3 bg-home-acc/20 border border-home-acc/30 rounded-xl backdrop-blur-xl text-home-acc"
      >
        {isOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      <AnimatePresence>
        {(isOpen || typeof window !== 'undefined' && window.innerWidth >= 1024) && (
          <motion.nav
            initial={{ x: -200 }}
            animate={{ x: 0 }}
            exit={{ x: -200 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className={`fixed top-0 left-0 w-[240px] h-screen bg-[#0d1a10] border-r border-home-acc/5 p-6 z-[9999] flex flex-col gap-1 pointer-events-auto shadow-2xl lg:translate-x-0 ${isOpen ? 'block' : 'hidden lg:flex'}`}
          >
            {/* Brand */}
            <div className="font-display text-lg font-extrabold mb-8 tracking-wider flex items-center gap-2">
              <div className="w-2 h-6 bg-home-acc rounded-full" />
              <div>Fore<span className="text-home-acc not-italic">Sight</span>X</div>
            </div>

            {/* Nav items */}
            <div className="flex-1 flex flex-col gap-1.5">
              {navItems.map((item) => {
                const isActive = activeTab === item.id || (item.id === 'home' && cityModes.includes(activeTab));

                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      setActiveTab(item.id);
                      if (window.innerWidth < 1024) setIsOpen(false);
                    }}
                    className={`flex items-center gap-3 px-4 py-3 rounded-xl text-[12px] font-bold transition-all border text-left
                      ${isActive
                        ? 'bg-home-acc/10 border-home-acc/20 text-home-acc shadow-[0_0_20px_rgba(125,249,160,0.05)]'
                        : 'border-transparent text-white/40 hover:bg-white/5 hover:text-white'
                      } ${item.id === 'model' ? 'mt-auto' : ''}`}
                  >
                    <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${isActive ? 'bg-home-acc shadow-[0_0_8px_rgba(125,249,160,0.5)]' : item.color}`} />
                    <span className="uppercase tracking-widest">{item.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Bottom: status + user + logout */}
            <div className="mt-8 pt-6 border-t border-white/5 space-y-5">
              {/* Node status */}
              <div>
                <div className="text-[10px] font-black text-white/20 uppercase tracking-[2px] mb-2">Node Status</div>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-home-acc animate-pulse" />
                  <div className="text-[10px] font-bold text-white/40 tracking-widest uppercase">Live Telemetry</div>
                </div>
              </div>

              {/* Logged-in user */}
              {user && (
                <div className="bg-home-acc/5 border border-home-acc/10 rounded-xl px-3 py-2.5">
                  <div className="text-[10px] font-mono text-home-acc/60 uppercase tracking-widest mb-0.5">Signed in as</div>
                  <div className="text-[12px] font-bold text-white/80 truncate">{user.name}</div>
                  <div className="text-[10px] font-mono text-white/30 capitalize">{user.role}</div>
                </div>
              )}

              {/* Logout button */}
              <button
                onClick={onLogout}
                className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-[12px] font-bold border border-transparent text-white/30 hover:bg-red-500/10 hover:border-red-500/20 hover:text-red-400 transition-all group"
              >
                <LogOut size={13} className="flex-shrink-0 group-hover:text-red-400" />
                <span className="uppercase tracking-widest">Sign Out</span>
              </button>
            </div>
          </motion.nav>
        )}
      </AnimatePresence>

      {/* OVERLAY FOR MOBILE */}
      {isOpen && (
        <div
          onClick={() => setIsOpen(false)}
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[9998] lg:hidden"
        />
      )}
    </>
  );
};

export default Sidebar;
