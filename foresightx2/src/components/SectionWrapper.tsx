import { motion } from 'motion/react';
import { ReactNode } from 'react';

interface SectionWrapperProps {
  id: string;
  label: string;
  theme: 'home' | 'traffic' | 'air' | 'energy' | 'weather' | 'model';
  badge?: string;
  children: ReactNode;
}

const themeClasses = {
  home: {
    bg: 'from-[#0d2e1a] to-[#0a1f14]',
    label: 'text-home-acc',
    line: 'bg-home-acc/20',
    badge: 'bg-home-acc/15 text-home-acc border-home-acc/30'
  },
  traffic: {
    bg: 'from-[#1a1200] to-[#120d00]',
    label: 'text-traf-acc',
    line: 'bg-traf-acc/20',
    badge: 'bg-traf-acc/15 text-traf-acc border-traf-acc/30'
  },
  air: {
    bg: 'from-[#0a1e2a] to-[#060e16]',
    label: 'text-air-acc',
    line: 'bg-air-acc/20',
    badge: 'bg-air-acc/15 text-air-acc border-air-acc/30'
  },
  energy: {
    bg: 'from-[#130826] to-[#08041a]',
    label: 'text-eng-acc',
    line: 'bg-eng-acc/20',
    badge: 'bg-eng-acc/15 text-eng-acc border-eng-acc/30'
  },
  weather: {
    bg: 'from-[#101e2a] to-[#080e18]',
    label: 'text-wth-acc',
    line: 'bg-wth-acc/20',
    badge: 'bg-wth-acc/15 text-wth-acc border-wth-acc/30'
  },
  model: {
    bg: 'from-[#1a1200] to-[#0d0900]',
    label: 'text-traf-acc',
    line: 'bg-traf-acc/20',
    badge: 'bg-traf-acc/15 text-traf-acc border-traf-acc/30'
  }
};

const SectionWrapper = ({ id, label, theme, badge, children }: SectionWrapperProps) => {
  const t = themeClasses[theme];
  return (
    <section id={id} className={`p-10 md:p-12 pb-14 bg-gradient-to-b ${t.bg} border-t border-white/5 transition-all duration-700`}>
      <div className="flex items-center gap-3.5 mb-7">
        <div className={`font-display text-xl font-bold ${t.label}`}>{label}</div>
        <div className={`flex-1 h-px ${t.line}`} />
        {badge && (
          <span className={`text-[10px] px-2.5 py-1 rounded-full font-semibold tracking-tight border ${t.badge}`}>
            {badge}
          </span>
        )}
      </div>
      {children}
    </section>
  );
};

export default SectionWrapper;
