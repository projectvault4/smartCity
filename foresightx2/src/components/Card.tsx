import { ReactNode } from 'react';

interface CardProps {
  key?: string | number;
  title: string;
  theme: 'home' | 'traffic' | 'air' | 'energy' | 'weather' | 'model';
  children: ReactNode;
  className?: string;
  titleClassName?: string;
}

const themeStyles = {
  home: { bg: 'bg-home-acc/6', border: 'border-home-acc/15', title: 'text-home-acc' },
  traffic: { bg: 'bg-traf-acc/6', border: 'border-traf-acc/15', title: 'text-traf-acc' },
  air: { bg: 'bg-air-acc/6', border: 'border-air-acc/15', title: 'text-air-acc' },
  energy: { bg: 'bg-eng-acc/6', border: 'border-eng-acc/15', title: 'text-eng-acc' },
  weather: { bg: 'bg-wth-acc/6', border: 'border-wth-acc/15', title: 'text-wth-acc' },
  model: { bg: 'bg-traf-acc/6', border: 'border-traf-acc/15', title: 'text-traf-acc' },
};

const Card = ({ title, theme, children, className = '', titleClassName = '' }: CardProps) => {
  const s = themeStyles[theme];
  return (
    <div className={`rounded-2xl p-5 md:p-6 border ${s.bg} ${s.border} ${className}`}>
      <h3 className={`font-display text-[17px] font-bold mb-3.5 ${s.title} ${titleClassName}`}>{title}</h3>
      <div className="text-[13px] text-white/60 leading-relaxed">
        {children}
      </div>
    </div>
  );
};

export default Card;
