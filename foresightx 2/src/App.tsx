import { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import Sidebar from './components/Sidebar';
import Hero from './components/Hero';
import SectionWrapper from './components/SectionWrapper';
import Card from './components/Card';
import Sparkline from './components/Sparkline';
import PredictionPanel from './components/PredictionPanel';
import XaiPanel from './components/XaiPanel';
import MultivariatePanel from './components/MultivariatePanel';
import QueryInterface from './components/QueryInterface';
import RippleSimulator from './components/RippleSimulator';
import StressHeatmap from './components/StressHeatmap';
import DriftMonitor from './components/DriftMonitor';
import EventImpact from './components/EventImpact';
import ReportGenerator from './components/ReportGenerator';
import CitizenAdvisoryConsole from './components/CitizenAdvisoryConsole';
import AnomalyDetectionEmbed from './components/AnomalyDetectionEmbed';
import VoiceBriefing from './components/VoiceBriefing';
import { backendApi, getInitialData, updateMetric, CityData } from './services/dataService';

type DashboardView = 'overview' | 'simulation' | 'prediction' | 'xai' | 'analytics' | 'anomaly' | 'voice' | 'model';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [heroMode, setHeroMode] = useState<'home' | 'traffic' | 'air' | 'energy' | 'weather'>('home');
  const [dashboardView, setDashboardView] = useState<DashboardView>('overview');
  const [cityData, setCityData] = useState<CityData>(getInitialData());
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  // Real-time update interval
  useEffect(() => {
    const interval = setInterval(() => {
      setCityData(prev => ({
        ...prev,
        traffic: updateMetric(prev.traffic),
        air: updateMetric(prev.air),
        weather: updateMetric(prev.weather),
        energy: updateMetric(prev.energy),
        timestamp: new Date().toLocaleTimeString(),
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    let mounted = true;

    backendApi.health()
      .then(() => {
        if (mounted) setBackendStatus('online');
      })
      .catch(() => {
        if (mounted) setBackendStatus('offline');
      });

    return () => {
      mounted = false;
    };
  }, []);

  // Unified state management
  useEffect(() => {
    if (['traffic', 'air', 'energy', 'weather', 'home'].includes(activeTab)) {
      setDashboardView('overview');
      setHeroMode(activeTab === 'home' ? 'home' : activeTab as any);
    } else if (activeTab === 'simulation') {
      setDashboardView('simulation');
    } else if (activeTab === 'prediction') {
      setDashboardView('prediction');
    } else if (activeTab === 'xai') {
      setDashboardView('xai');
    } else if (activeTab === 'analytics') {
      setDashboardView('analytics');
    } else if (activeTab === 'anomaly') {
      setDashboardView('anomaly');
    } else if (activeTab === 'voice') {
      setDashboardView('voice');
    } else if (activeTab === 'model') {
      setDashboardView('model');
    }
  }, [activeTab]);

  return (
    <div className="flex bg-[#0d1a10] min-h-screen font-sans antialiased text-white">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <main className="flex-1 lg:ml-[240px]">
        <div className="fixed right-4 top-4 z-50 rounded-lg border border-white/10 bg-black/70 px-3 py-2 text-[10px] font-bold uppercase tracking-widest text-white/70 backdrop-blur-md">
          API: <span className={backendStatus === 'online' ? 'text-home-acc' : backendStatus === 'offline' ? 'text-red-400' : 'text-white/40'}>
            {backendStatus}
          </span>
        </div>
        <Hero mode={heroMode} setMode={(m) => { setHeroMode(m); setActiveTab(m); }} data={cityData} />

        <div className="p-8 md:p-12">
          
          <motion.div
            key={dashboardView + heroMode}
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {dashboardView === 'overview' && (
              <div className="space-y-12">
                {/* PRIMARY ACTIONS: SEARCH & EVENT IMPACT - STACKED */}
                <div className="space-y-8">
                   <QueryInterface />
                   <CitizenAdvisoryConsole />
                   <EventImpact />
                </div>

                {/* PULSE METRICS - RESTORED */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  {[
                    { id: 'traffic', title: 'Traffic Flow', val: cityData.traffic.value, label: cityData.traffic.label, unit: 'vehicles/hr', theme: 'traf', history: cityData.traffic.history },
                    { id: 'air', title: 'Air Quality', val: cityData.air.value, label: cityData.air.label, unit: 'AQI', theme: 'air', history: cityData.air.history },
                    { id: 'weather', title: 'Temperature', val: `${cityData.weather.value}°C`, label: cityData.weather.label, unit: 'Stable', theme: 'wth', history: cityData.weather.history },
                    { id: 'energy', title: 'Electricity', val: `${cityData.energy.value} MW`, label: cityData.energy.label, unit: 'High Demand', theme: 'eng', history: cityData.energy.history },
                  ].map(item => (
                    <div 
                      key={item.id}
                      className={`rounded-[24px] p-6 border transition-all bg-black/35 border-white/5 hover:border-white/20 hover:bg-black/50 group`}
                    >
                      <div className="flex justify-between items-center mb-4">
                        <span className="text-[10px] font-black text-white/40 uppercase tracking-widest group-hover:text-white/60 transition-colors">{item.title}</span>
                        <span className={`text-[10px] px-2.5 py-1 rounded-full font-bold border bg-${item.theme}-acc/15 text-${item.theme}-acc border-${item.theme}-acc/30`}>
                          {item.label}
                        </span>
                      </div>
                      <div className={`font-display text-[44px] font-black leading-none mb-4 text-white group-hover:scale-105 transition-transform origin-left`}>{item.val}</div>
                      <div className="h-12 w-full opacity-60">
                         <Sparkline color={item.theme === 'traf' ? '#f39c12' : item.theme === 'air' ? '#3498db' : item.theme === 'wth' ? '#9b59b6' : '#2ecc71'} points={item.history.map((h, i) => `${i * 25},${50 - (h / 100)}`).join(' ')} />
                      </div>
                      <div className="text-[10px] mt-4 font-bold text-white/30 uppercase tracking-widest">{item.unit}</div>
                    </div>
                  ))}
                </div>

                {/* ANALYTICS & REPORTS */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                   <div className="lg:col-span-2">
                       <StressHeatmap />
                   </div>
                   <div className="space-y-6">
                      <ReportGenerator data={cityData} />
                   </div>
                </div>
              </div>
            )}

            {dashboardView === 'simulation' && (
               <div className="space-y-8">
                  <RippleSimulator />
                  <Card title="Cross-Domain Knowledge Graph" theme="home">
                     <p className="text-sm text-white/60 leading-relaxed max-w-3xl">This simulator models the "Ripple Effect" of traffic on energy grids and air quality. For instance, high idling traffic (congestion) leads to an immediate 14% increase in particulate matter (AQI) and creates heat pockets that drive up local energy demand for cooling systems.</p>
                  </Card>
               </div>
            )}

            {dashboardView === 'prediction' && (
              <div className="space-y-8">
                 <PredictionPanel data={cityData} />
              </div>
            )}

            {dashboardView === 'xai' && (
              <div className="space-y-8">
                 <XaiPanel data={cityData} />
              </div>
            )}

            {dashboardView === 'analytics' && (
              <div className="space-y-8">
                 <MultivariatePanel data={cityData} />
              </div>
            )}

            {dashboardView === 'anomaly' && (
              <AnomalyDetectionEmbed />
            )}

            {dashboardView === 'voice' && (
              <VoiceBriefing />
            )}

            {dashboardView === 'model' && (
               <div className="space-y-8">
                  <DriftMonitor />
                  <SectionWrapper id="s-model" label="System Benchmarks" theme="model">
                    <div className="bg-black/35 border border-white/10 rounded-2xl p-6 overflow-auto">
                       <table className="w-full text-left">
                          <thead>
                             <tr className="border-b border-white/5 text-[10px] font-bold text-white/30 uppercase tracking-[2px]">
                                <th className="pb-4">Model Architecture</th>
                                <th className="pb-4 text-right">Convergence</th>
                                <th className="pb-4 text-right">Latency</th>
                                <th className="pb-4 text-right">Efficiency</th>
                             </tr>
                          </thead>
                          <tbody>
                             {[
                                { name: 'ForeSight BiLSTM', acc: '98.4%', lat: '12ms', eff: 'High' },
                                { name: 'Urban Transformer', acc: '97.2%', lat: '45ms', eff: 'Moderate' },
                                { name: 'Hybrid Ensemble', acc: '99.1%', lat: '120ms', eff: 'Low' },
                                { name: 'Light Adaptive', acc: '94.5%', lat: '2ms', eff: 'Ultra-High' },
                             ].map((row, i) => (
                                <tr key={i} className="border-b border-white/5 text-sm hover:bg-white/5 transition-all">
                                   <td className="py-4 font-bold text-white/80">{row.name}</td>
                                   <td className="py-4 text-right font-mono text-home-acc">{row.acc}</td>
                                   <td className="py-4 text-right font-mono text-white/40">{row.lat}</td>
                                   <td className="py-4 text-right">
                                      <span className="px-2 py-0.5 bg-white/5 rounded text-[10px] font-bold uppercase">{row.eff}</span>
                                   </td>
                                </tr>
                             ))}
                          </tbody>
                       </table>
                    </div>
                  </SectionWrapper>
               </div>
            )}
          </motion.div>
        </div>
      </main>
    </div>
  );
}
