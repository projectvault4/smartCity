import { useState, useEffect, ReactElement } from 'react';
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import { useAuth } from './context/AuthContext';
import LoginPage from './pages/LoginPage';
import CitizenDashboard from './pages/CitizenDashboard';
import Sidebar from './components/Sidebar';
import Hero from './components/Hero';
import SectionWrapper from './components/SectionWrapper';
import Sparkline from './components/Sparkline';
import CityPulseForecast from './components/CityPulseForecast';
import YearlyOutlook from './components/YearlyOutlook';
import XaiPanel from './components/XaiPanel';
import MultivariatePanel from './components/MultivariatePanel';
import QueryInterface from './components/QueryInterface';
import RippleSimulator from './components/RippleSimulator';
import StressHeatmap from './components/StressHeatmap';
import DriftMonitor from './components/DriftMonitor';
import ReportGenerator from './components/ReportGenerator';
import RiskAdvisoryConsole from './components/RiskAdvisoryConsole';
import AnomalyDetectionEmbed from './components/AnomalyDetectionEmbed';
import VoiceBriefing from './components/VoiceBriefing';
import DecisionCenter from './components/DecisionCenter';
import AlertReplay from './components/AlertReplay';
import ProofPanel from './components/ProofPanel';
import { backendApi, cityDataFromModelConditions, getInitialData, updateMetric, CityData, ModelConditions, ForecastPoint } from './services/dataService';

// ── Protected route guards ────────────────────────────────────────────────────
function RequireAdmin({ children }: { children: ReactElement }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  if (user.role !== 'admin') return <Navigate to="/citizen" replace />;
  return children;
}

function RequireCitizen({ children }: { children: ReactElement }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  if (user.role !== 'citizen') return <Navigate to="/dashboard" replace />;
  return children;
}

function RootRedirect() {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  return <Navigate to={user.role === 'admin' ? '/dashboard' : '/citizen'} replace />;
}

// ── Admin dashboard ───────────────────────────────────────────────────────────
type DashboardView = 'overview' | 'decision' | 'replay' | 'simulation' | 'prediction' | 'yearly' | 'xai' | 'analytics' | 'anomaly' | 'voice' | 'advisories' | 'model';

function AdminDashboard() {
  const { logout } = useAuth();
  const navigate = useNavigate();

  const [activeTab, setActiveTab] = useState('decision');
  const [heroMode, setHeroMode] = useState<'home' | 'traffic' | 'air' | 'energy' | 'weather'>('home');
  const [dashboardView, setDashboardView] = useState<DashboardView>('decision');
  const [cityData, setCityData] = useState<CityData>(getInitialData());
  const [modelConditions, setModelConditions] = useState<ModelConditions | null>(null);
  const [modelForecast, setModelForecast] = useState<ForecastPoint[] | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    const interval = setInterval(() => {
      if (modelConditions) return;
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
  }, [modelConditions]);

  // Connect to the backend, retrying until it responds. The Render free tier
  // sleeps when idle and can take ~30-60 s to wake, so a single check on page
  // load would leave the dashboard stuck on "offline" with fallback data.
  useEffect(() => {
    let mounted = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;

    const connect = async () => {
      try {
        await backendApi.health();
        if (!mounted) return;
        setBackendStatus('online');
        const [conditions, forecast] = await Promise.all([
          backendApi.modelConditions('bangalore'),
          backendApi.modelForecast('bangalore', 6),
        ]);
        if (!mounted) return;
        if ((conditions as any)?.data) {
          setModelConditions((conditions as any).data);
          setCityData((prev) => cityDataFromModelConditions((conditions as any).data, prev));
        }
        if (forecast?.data) setModelForecast(forecast.data);
      } catch {
        if (!mounted) return;
        // Show "checking" while the backend may still be waking up, then "offline".
        setBackendStatus(attempt < 6 ? 'checking' : 'offline');
        attempt += 1;
        // Retry every 10 s for the first minute, then every 30 s.
        timer = setTimeout(connect, attempt <= 6 ? 10000 : 30000);
      }
    };

    connect();
    return () => {
      mounted = false;
      if (timer) clearTimeout(timer);
    };
  }, []);

  // Refresh the model forecast on a rolling interval so the prediction window
  // stays anchored to the live server clock ("now -> now + horizon").
  useEffect(() => {
    const id = setInterval(() => {
      backendApi.modelForecast('bangalore', 6)
        .then((response) => setModelForecast(response.data))
        .catch(() => {});
      backendApi.modelConditions('bangalore')
        .then((response: any) => {
          if (!response?.data) return;
          setBackendStatus('online');
          setModelConditions(response.data);
          setCityData((prev) => cityDataFromModelConditions(response.data, prev));
        })
        .catch(() => setBackendStatus('offline'));
    }, 300000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (['traffic', 'air', 'energy', 'weather', 'home'].includes(activeTab)) {
      setDashboardView('overview');
      setHeroMode(activeTab === 'home' ? 'home' : activeTab as any);
    } else if (activeTab === 'decision') setDashboardView('decision');
    else if (activeTab === 'replay') setDashboardView('replay');
    else if (activeTab === 'simulation') setDashboardView('simulation');
    else if (activeTab === 'prediction') setDashboardView('prediction');
    else if (activeTab === 'yearly') setDashboardView('yearly');
    else if (activeTab === 'xai') setDashboardView('xai');
    else if (activeTab === 'analytics') setDashboardView('analytics');
    else if (activeTab === 'anomaly') setDashboardView('anomaly');
    else if (activeTab === 'voice') setDashboardView('voice');
    else if (activeTab === 'advisories') setDashboardView('advisories');
    else if (activeTab === 'model') setDashboardView('model');
  }, [activeTab]);

  const handleLogout = () => { logout(); navigate('/login', { replace: true }); };

  return (
    <div className="flex bg-[#0d1a10] min-h-screen font-sans antialiased text-white">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} onLogout={handleLogout} />

      <main className="flex-1 lg:ml-[240px]">
        {/* API status badge */}
        <div className="fixed right-4 top-4 z-50 rounded-lg border border-white/10 bg-black/70 px-3 py-2 text-[10px] font-bold uppercase tracking-widest text-white/70 backdrop-blur-md">
          API: <span className={backendStatus === 'online' ? 'text-home-acc' : backendStatus === 'offline' ? 'text-red-400' : 'text-white/40'}>
            {backendStatus}
          </span>
        </div>

        {dashboardView === 'overview' && (
          <Hero mode={heroMode} setMode={(m) => { setHeroMode(m); setActiveTab(m); }} data={cityData} />
        )}

        <div className="p-8 md:p-12">
          <motion.div
            key={dashboardView + heroMode}
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {dashboardView === 'overview' && (
              <div className="space-y-12">
                <div className="space-y-8">
                  <QueryInterface modelConditions={modelConditions} forecast={modelForecast || []} />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  {[
                    { id: 'traffic', title: 'Traffic Flow', val: cityData.traffic.value, label: cityData.traffic.label, unit: 'vehicles/hr', theme: 'traf', history: cityData.traffic.history },
                    { id: 'air', title: 'Air Quality', val: cityData.air.value, label: cityData.air.label, unit: 'AQI', theme: 'air', history: cityData.air.history },
                    { id: 'weather', title: 'Temperature', val: `${cityData.weather.value}°C`, label: cityData.weather.label, unit: 'Stable', theme: 'wth', history: cityData.weather.history },
                    { id: 'energy', title: 'Electricity', val: `${cityData.energy.value} MW`, label: cityData.energy.label, unit: 'High Demand', theme: 'eng', history: cityData.energy.history },
                  ].map(item => (
                    <div key={item.id} className="rounded-[24px] p-6 border transition-all bg-black/35 border-white/5 hover:border-white/20 hover:bg-black/50 group">
                      <div className="flex justify-between items-center mb-4">
                        <span className="text-[10px] font-black text-white/40 uppercase tracking-widest group-hover:text-white/60 transition-colors">{item.title}</span>
                        <span className={`text-[10px] px-2.5 py-1 rounded-full font-bold border bg-${item.theme}-acc/15 text-${item.theme}-acc border-${item.theme}-acc/30`}>
                          {item.label}
                        </span>
                      </div>
                      <div className="font-display text-[44px] font-black leading-none mb-4 text-white group-hover:scale-105 transition-transform origin-left">{item.val}</div>
                      <div className="h-12 w-full opacity-60">
                        <Sparkline color={item.theme === 'traf' ? '#f39c12' : item.theme === 'air' ? '#3498db' : item.theme === 'wth' ? '#9b59b6' : '#2ecc71'} points={item.history.map((h, i) => `${i * 25},${50 - (h / 100)}`).join(' ')} />
                      </div>
                      <div className="text-[10px] mt-4 font-bold text-white/30 uppercase tracking-widest">{item.unit}</div>
                    </div>
                  ))}
                </div>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                  <div className="lg:col-span-2"><StressHeatmap /></div>
                  <div className="space-y-6"><ReportGenerator data={cityData} forecast={modelForecast} /></div>
                </div>
              </div>
            )}

            {dashboardView === 'decision' && <DecisionCenter />}

            {dashboardView === 'replay' && <AlertReplay />}

            {dashboardView === 'simulation' && (
              <div className="space-y-8">
                <RippleSimulator forecast={modelForecast} />
              </div>
            )}

            {dashboardView === 'prediction' && (
              <div className="space-y-8">
                <SectionWrapper id="s-forecast" label="6-Hour Forecast" theme="home">
                  <CityPulseForecast forecast={modelForecast} />
                </SectionWrapper>
              </div>
            )}

            {dashboardView === 'yearly' && (
              <div className="space-y-8">
                <SectionWrapper id="s-yearly" label="2026 Long-Range Forecast" theme="home">
                  <YearlyOutlook city="bangalore" />
                </SectionWrapper>
              </div>
            )}

            {dashboardView === 'xai' && (
              <div className="space-y-8"><XaiPanel data={cityData} forecast={modelForecast} /></div>
            )}

            {dashboardView === 'analytics' && (
              <div className="space-y-8"><MultivariatePanel data={cityData} /></div>
            )}

            {dashboardView === 'anomaly' && (
              <AnomalyDetectionEmbed modelConditions={modelConditions} />
            )}

            {dashboardView === 'voice' && (
              <VoiceBriefing data={cityData} modelConditions={modelConditions} />
            )}

            {dashboardView === 'advisories' && (
              <RiskAdvisoryConsole initialModelConditions={modelConditions} />
            )}

            {dashboardView === 'model' && (
              <div className="space-y-8">
                <ProofPanel />
                <DriftMonitor />
              </div>
            )}
          </motion.div>
        </div>
      </main>
    </div>
  );
}

// ── Root router ───────────────────────────────────────────────────────────────
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<RootRedirect />} />
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/dashboard"
        element={<RequireAdmin><AdminDashboard /></RequireAdmin>}
      />
      <Route
        path="/citizen"
        element={<RequireCitizen><CitizenDashboard /></RequireCitizen>}
      />
      {/* Catch-all */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
