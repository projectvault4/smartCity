import Card from './Card';

const StressHeatmap = () => {
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May'];
  const daysPerMonth = 30;

  const getColor = (val: number) => {
    if (val > 80) return 'bg-[#ef4444]'; // High Stress
    if (val > 60) return 'bg-[#f97316]'; // Moderate
    if (val > 40) return 'bg-[#eab308]'; // Mild
    return 'bg-[#22c55e]'; // Low Stress
  };

  return (
    <Card title="Temporal Heatmap Calendar" theme="home">
      <div className="p-2 space-y-6">
        <p className="text-sm text-white/50">GitHub-style urban stress calendar (Traffic + AQI + Energy). Color intensity indicates combined environmental pressure.</p>
        
        <div className="flex flex-col gap-3">
          {months.map((month) => (
            <div key={month} className="flex items-center gap-3">
              <div className="w-8 text-[10px] font-bold text-white/30 uppercase">{month}</div>
              <div className="flex gap-1 flex-1 overflow-auto pb-1">
                {Array.from({ length: daysPerMonth }).map((_, i) => {
                  // Simulate some interesting patterns (holidays = high, lockdown = low)
                  let val = Math.random() * 100;
                  if (month === 'Jan' && i < 7) val = 95; // New Year stress
                  if (month === 'Mar' && i > 20) val = 15; // Simulated lockdown green
                  
                  return (
                    <div 
                      key={i} 
                      className={`w-3 h-3 rounded-[2px] shrink-0 ${getColor(val)} opacity-60 hover:opacity-100 transition-opacity cursor-pointer`}
                      title={`${month} ${i+1}: Stress Index ${Math.round(val)}`}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        <div className="flex items-center gap-4 pt-4 border-t border-white/5">
           <div className="text-[10px] text-white/30 font-medium uppercase tracking-widest">Stress index:</div>
           <div className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 bg-[#22c55e] rounded-[1px]" />
              <div className="w-2.5 h-2.5 bg-[#eab308] rounded-[1px]" />
              <div className="w-2.5 h-2.5 bg-[#f97316] rounded-[1px]" />
              <div className="w-2.5 h-2.5 bg-[#ef4444] rounded-[1px]" />
              <span className="text-[10px] text-white/40 ml-1">Low → High Stress</span>
           </div>
        </div>

      </div>
    </Card>
  );
};

export default StressHeatmap;
