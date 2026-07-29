import { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  Bell,
  CheckCircle2,
  Database,
  Loader2,
  Play,
  RefreshCw,
  Send,
  Trash2,
  UserPlus,
  Users
} from 'lucide-react';
import Card from './Card';
import { backendApi, UserPayload } from '../services/dataService';

const makeDefaultUser = (): UserPayload => ({
  user_id: `user_${Date.now().toString().slice(-6)}`,
  name: 'Ravi Kumar',
  email: 'ravi@example.com',
  phone: '+919876543210',
  age: 68,
  city: 'Bangalore',
  ward: 'Anekal Ward',
  latitude: 12.9716,
  longitude: 77.5946,
  preferences: {
    inApp: true,
    email: false,
    sms: false
  },
  status: 'active'
});

const riskGroupOptions = [
  { key: 'elder', label: 'Elder' },
  { key: 'commuter', label: 'Commuter' },
  { key: 'resp', label: 'Respiratory' },
  { key: 'child', label: 'Child' },
  { key: 'worker', label: 'Worker' }
];

const channelOptions = [
  { key: 'in_app', label: 'In-App' },
  { key: 'email', label: 'Email' },
  { key: 'sms', label: 'SMS' }
];

const toUserForm = (user: any): UserPayload => ({
  user_id: user.user_id,
  name: user.name,
  email: user.email || '',
  phone: user.phone || '',
  age: user.age || 0,
  city: user.city || '',
  ward: user.ward || '',
  latitude: Number(user.latitude || 0),
  longitude: Number(user.longitude || 0),
  preferences: user.preferences || { inApp: true, email: false, sms: false },
  status: user.status || 'active'
});

const Field = ({
  label,
  value,
  type = 'text',
  onChange
}: {
  label: string;
  value: string | number;
  type?: string;
  onChange: (value: string) => void;
}) => (
  <label className="block">
    <span className="mb-1 block text-[9px] font-black uppercase tracking-widest text-white/30">{label}</span>
    <input
      value={value}
      type={type}
      onChange={(event) => onChange(event.target.value)}
      className="w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none focus:border-home-acc/50"
    />
  </label>
);

const Pill = ({
  active,
  children,
  onClick
}: {
  key?: string;
  active: boolean;
  children: string;
  onClick: () => void;
}) => (
  <button
    onClick={onClick}
    className={`rounded-full border px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest ${
      active
        ? 'border-home-acc bg-home-acc/20 text-home-acc'
        : 'border-white/10 bg-white/5 text-white/40 hover:text-white/70'
    }`}
  >
    {children}
  </button>
);

const CitizenAdvisoryConsole = () => {
  const [form, setForm] = useState<UserPayload>(makeDefaultUser());
  const [riskGroups, setRiskGroups] = useState(['elder', 'commuter']);
  const [channels, setChannels] = useState(['in_app']);
  const [users, setUsers] = useState<any[]>([]);
  const [selectedUserId, setSelectedUserId] = useState('');
  const [notifications, setNotifications] = useState<any[]>([]);
  const [riskAssessment, setRiskAssessment] = useState<any>(null);
  const [advisory, setAdvisory] = useState<any>(null);
  const [health, setHealth] = useState<any>(null);
  const [readiness, setReadiness] = useState<any>(null);
  const [batchResult, setBatchResult] = useState<any>(null);
  const [modelConditions, setModelConditions] = useState<any>(null);
  const [loading, setLoading] = useState('');
  const [message, setMessage] = useState('');
  const [conditions, setConditions] = useState({
    aqi: 240,
    temperature: 40,
    weather: 'Rain',
    rainLastHourMm: 8,
    traffic: 'severe'
  });

  const activeUser = useMemo(
    () => users.find((user) => user.id === selectedUserId) || null,
    [selectedUserId, users]
  );

  const deliveryUser = useMemo(() => {
    if (!activeUser) {
      return null;
    }

    if (form.user_id !== activeUser.user_id) {
      return activeUser;
    }

    return {
      ...activeUser,
      name: form.name,
      email: form.email,
      phone: form.phone,
      age: form.age,
      city: form.city,
      ward: form.ward,
      latitude: form.latitude,
      longitude: form.longitude,
      preferences: form.preferences,
      status: form.status
    };
  }, [activeUser, form]);

  const riskPayload = useMemo(() => ({
    user: deliveryUser || form,
    riskGroups,
    aqi: { aqi: conditions.aqi },
    weather: {
      weather: {
        main: conditions.weather,
        description: conditions.rainLastHourMm > 0 ? 'heavy rain' : conditions.weather
      },
      temperature: { value: conditions.temperature },
      rainLastHourMm: conditions.rainLastHourMm
    },
    traffic: { congestionLevel: conditions.traffic }
  }), [deliveryUser, form, riskGroups, conditions]);

  const showError = (error: any) => {
    setMessage(error?.message || 'Request failed');
  };

  const loadHealth = async () => {
    const [healthResponse, readinessResponse] = await Promise.allSettled([
      backendApi.health(),
      backendApi.readiness()
    ]);

    setHealth(healthResponse.status === 'fulfilled' ? healthResponse.value : { status: 'offline' });
    setReadiness(readinessResponse.status === 'fulfilled' ? readinessResponse.value : { status: 'degraded' });
  };

  const loadUsers = async (preferredUserId = selectedUserId) => {
    const response: any = await backendApi.listUsers();
    const nextUsers = response.data || [];
    setUsers(nextUsers);

    if (preferredUserId && nextUsers.some((user: any) => user.id === preferredUserId)) {
      setSelectedUserId(preferredUserId);
    } else if (!preferredUserId && nextUsers[0]) {
      setSelectedUserId(nextUsers[0].id);
    }

    return nextUsers;
  };

  const loadNotifications = async (userId = selectedUserId) => {
    if (!userId) {
      setNotifications([]);
      return;
    }

    const response: any = await backendApi.listNotifications(userId);
    setNotifications(response.data || []);
  };

  useEffect(() => {
    loadHealth().catch(showError);
    loadUsers().catch(showError);
  }, []);

  useEffect(() => {
    loadNotifications().catch(() => setNotifications([]));
  }, [selectedUserId]);

  useEffect(() => {
    if (activeUser) {
      setForm(toUserForm(activeUser));
    }
  }, [activeUser?.id]);

  const createUser = async () => {
    setLoading('create');
    setMessage('');

    try {
      const response: any = await backendApi.createUser(form);
      setMessage(`Created user ${response.data.name}`);
      setSelectedUserId(response.data.id);
      setForm(makeDefaultUser());
      await loadUsers(response.data.id);
    } catch (error) {
      showError(error);
    } finally {
      setLoading('');
    }
  };

  const updateUser = async () => {
    if (!activeUser) {
      setMessage('Select a user first.');
      return;
    }

    setLoading('update');
    setMessage('');

    try {
      const response: any = await backendApi.updateUser(activeUser.id, {
        name: form.name,
        email: form.email,
        phone: form.phone,
        age: form.age,
        city: form.city,
        ward: form.ward,
        latitude: form.latitude,
        longitude: form.longitude,
        preferences: form.preferences,
        status: form.status
      });
      setMessage(`Updated ${response.data.name}`);
      await loadUsers();
    } catch (error) {
      showError(error);
    } finally {
      setLoading('');
    }
  };

  const deleteUser = async () => {
    if (!activeUser) {
      setMessage('Select a user first.');
      return;
    }

    setLoading('delete');
    setMessage('');

    try {
      await backendApi.deleteUser(activeUser.id);
      setMessage(`Deleted ${activeUser.name}`);
      setSelectedUserId('');
      setAdvisory(null);
      setRiskAssessment(null);
      setNotifications([]);
      await loadUsers();
    } catch (error) {
      showError(error);
    } finally {
      setLoading('');
    }
  };

  const assessRisk = async () => {
    setLoading('risk');
    setMessage('');

    try {
      const response: any = await backendApi.assessRisk(riskPayload);
      setRiskAssessment(response.data);
      setMessage(`Risk calculated: ${response.data.riskLevel} (${response.data.score})`);
    } catch (error) {
      showError(error);
    } finally {
      setLoading('');
    }
  };

  const generateAdvisory = async () => {
    setLoading('advisory');
    setMessage('');

    try {
      const response: any = await backendApi.generateAdvisory(riskPayload);
      const generated = response.data?.advisories?.[0] || null;
      setRiskAssessment(response.data?.riskAssessment || null);
      setAdvisory(generated);
      setMessage(generated ? 'Advisory generated.' : 'No advisory needed for these conditions.');
    } catch (error) {
      showError(error);
    } finally {
      setLoading('');
    }
  };

  const loadModelConditions = async () => {
    setLoading('model');
    setMessage('');

    try {
      const city = String(activeUser?.city || form.city || 'bangalore').toLowerCase();
      const response: any = await backendApi.modelConditions(city);
      const data = response.data;
      setModelConditions(data);
      setConditions({
        aqi: Math.round(data.aqi?.aqi || 0),
        temperature: Math.round(data.weather?.temperature?.value || 0),
        weather: data.weather?.weather?.main || 'Clear',
        rainLastHourMm: data.weather?.rainLastHourMm || 0,
        traffic: data.traffic?.congestionLevel || 'moderate'
      });
      setMessage(`Loaded trained-model forecast for ${data.city}: ${data.forecastFor}`);
    } catch (error) {
      showError(error);
    } finally {
      setLoading('');
    }
  };

  const deliverAdvisory = async () => {
    if (!activeUser || !advisory) {
      setMessage('Select a saved user and generate an advisory first.');
      return;
    }

    setLoading('deliver');
    setMessage('');

    try {
      if (channels.includes('sms') && !deliveryUser?.phone) {
        setMessage('Enter a phone number before sending SMS.');
        return;
      }

      await backendApi.deliverNotification({
        user: deliveryUser,
        advisory: {
          ...advisory,
          userId: deliveryUser.id,
          deliveryChannels: channels
        },
        channels
      });
      setMessage(`Delivery attempted for ${channels.join(', ')}.`);
      await loadNotifications(activeUser.id);
    } catch (error) {
      showError(error);
    } finally {
      setLoading('');
    }
  };

  const markRead = async (notificationId: string) => {
    if (!activeUser) return;

    setLoading(notificationId);

    try {
      await backendApi.markNotificationRead(activeUser.id, notificationId);
      await loadNotifications(activeUser.id);
    } catch (error) {
      showError(error);
    } finally {
      setLoading('');
    }
  };

  const runBatch = async () => {
    setLoading('batch');
    setMessage('');

    try {
      const response: any = await backendApi.runAdvisoryBatch();
      setBatchResult(response.data);
      setMessage(`Batch finished: ${response.data.status}`);
      await loadUsers();
      await loadNotifications();
    } catch (error) {
      showError(error);
    } finally {
      setLoading('');
    }
  };

  const sendPersonalizedSmsBatch = async () => {
    setLoading('bulkSms');
    setMessage('');

    try {
      const response: any = await backendApi.runAdvisoryBatch({
        startedBy: 'manual_sms_console',
        channels: ['sms'],
        sharedConditions: {
          aqi: { aqi: conditions.aqi },
          weather: {
            weather: {
              main: conditions.weather,
              description: conditions.rainLastHourMm > 0 ? 'heavy rain' : conditions.weather
            },
            temperature: { value: conditions.temperature },
            rainLastHourMm: conditions.rainLastHourMm
          },
          traffic: { congestionLevel: conditions.traffic }
        }
      });

      setBatchResult(response.data);
      setMessage(`Personalized SMS batch finished: ${response.data.notifications_sent} sent for ${response.data.users_processed} users.`);
      await loadUsers();
      await loadNotifications();
    } catch (error) {
      showError(error);
    } finally {
      setLoading('');
    }
  };

  const loadSelectedIntoForm = () => {
    if (!activeUser) return;

    setForm(toUserForm(activeUser));
  };

  const toggleValue = (value: string, setter: (next: string[]) => void, current: string[]) => {
    setter(current.includes(value) ? current.filter((item) => item !== value) : [...current, value]);
  };

  return (
    <div className="space-y-6">
      <Card title="Backend Operations" theme="model">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="rounded-xl border border-white/10 bg-black/30 p-4">
            <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-white/30">
              <Activity size={14} className="text-home-acc" />
              API
            </div>
            <div className="mt-2 text-2xl font-black text-white">{health?.status || 'checking'}</div>
            <div className="text-[10px] text-white/30">Port 5001</div>
          </div>
          <div className="rounded-xl border border-white/10 bg-black/30 p-4">
            <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-white/30">
              <Database size={14} className="text-air-acc" />
              Ready
            </div>
            <div className="mt-2 text-2xl font-black text-white">{readiness?.status || 'checking'}</div>
            <div className="text-[10px] text-white/30">PostgreSQL + Redis</div>
          </div>
          <button
            onClick={() => loadHealth().catch(showError)}
            className="flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 p-4 text-xs font-black uppercase tracking-widest text-white/60 hover:text-white"
          >
            <RefreshCw size={16} />
            Refresh Health
          </button>
          <button
            onClick={runBatch}
            disabled={loading === 'batch'}
            className="flex items-center justify-center gap-2 rounded-xl border border-traf-acc/30 bg-traf-acc/15 p-4 text-xs font-black uppercase tracking-widest text-traf-acc disabled:opacity-50"
          >
            {loading === 'batch' ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
            Run 15m Job
          </button>
          <button
            onClick={sendPersonalizedSmsBatch}
            disabled={loading === 'bulkSms'}
            className="flex items-center justify-center gap-2 rounded-xl border border-home-acc/40 bg-home-acc/15 p-4 text-xs font-black uppercase tracking-widest text-home-acc disabled:opacity-50"
          >
            {loading === 'bulkSms' ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
            SMS All Users
          </button>
        </div>
        {batchResult && (
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            {[
              ['Status', batchResult.status],
              ['Users', batchResult.users_processed],
              ['Advisories', batchResult.advisories_generated],
              ['Notifications', batchResult.notifications_sent]
            ].map(([label, value]) => (
              <div key={label} className="rounded-lg bg-black/30 p-3">
                <div className="text-white/30 uppercase tracking-widest text-[9px]">{label}</div>
                <div className="mt-1 font-black text-white">{value ?? '-'}</div>
              </div>
            ))}
          </div>
        )}
        {modelConditions && (
          <div className="mt-4 rounded-xl border border-home-acc/20 bg-home-acc/5 p-4 text-xs">
            <div className="font-black uppercase tracking-widest text-home-acc">Model Forecast Auto Mode</div>
            <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3">
              <div>AQI: <span className="font-bold text-white">{Math.round(modelConditions.aqi?.aqi || 0)}</span></div>
              <div>Temp: <span className="font-bold text-white">{Math.round(modelConditions.weather?.temperature?.value || 0)}C</span></div>
              <div>Traffic: <span className="font-bold text-white">{modelConditions.traffic?.congestionLevel}</span></div>
              <div>For: <span className="font-bold text-white">{modelConditions.forecastFor}</span></div>
            </div>
          </div>
        )}
      </Card>

      <Card title="Users, Risk Engine, Advisories, Notifications" theme="home">
        <div className="grid grid-cols-1 2xl:grid-cols-4 gap-6">
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-[10px] font-black text-white/40 uppercase tracking-widest">
              <UserPlus size={15} className="text-home-acc" />
              Create / Edit User
            </div>
            <Field label="User ID" value={form.user_id} onChange={(value) => setForm({ ...form, user_id: value })} />
            <Field label="Name" value={form.name} onChange={(value) => setForm({ ...form, name: value })} />
            <Field label="Email" value={form.email || ''} onChange={(value) => setForm({ ...form, email: value })} />
            <Field label="Phone" value={form.phone || ''} onChange={(value) => setForm({ ...form, phone: value })} />
            <div className="grid grid-cols-3 gap-2">
              <Field label="Age" type="number" value={form.age || 0} onChange={(value) => setForm({ ...form, age: Number(value) })} />
              <Field label="Lat" type="number" value={form.latitude || 0} onChange={(value) => setForm({ ...form, latitude: Number(value) })} />
              <Field label="Lon" type="number" value={form.longitude || 0} onChange={(value) => setForm({ ...form, longitude: Number(value) })} />
            </div>
            <Field label="City" value={form.city || ''} onChange={(value) => setForm({ ...form, city: value })} />
            <Field label="Ward" value={form.ward || ''} onChange={(value) => setForm({ ...form, ward: value })} />
            <div className="grid grid-cols-2 gap-2">
              <button onClick={createUser} disabled={loading === 'create'} className="rounded-lg bg-home-acc px-4 py-3 text-xs font-black uppercase tracking-widest text-black disabled:opacity-50">
                {loading === 'create' ? 'Creating' : 'Create'}
              </button>
              <button onClick={updateUser} disabled={!activeUser || loading === 'update'} className="rounded-lg border border-air-acc/40 bg-air-acc/15 px-4 py-3 text-xs font-black uppercase tracking-widest text-air-acc disabled:opacity-50">
                Update
              </button>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-[10px] font-black text-white/40 uppercase tracking-widest">
                <Users size={15} className="text-air-acc" />
                Users
              </div>
              <button onClick={() => loadUsers().catch(showError)} className="text-[9px] font-bold uppercase text-white/40 hover:text-white">Refresh</button>
            </div>
            <select
              value={selectedUserId}
              onChange={(event) => {
                setSelectedUserId(event.target.value);
                setAdvisory(null);
                setRiskAssessment(null);
              }}
              className="w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none focus:border-air-acc/50"
            >
              <option value="">Select user</option>
              {users.map((user) => (
                <option key={user.id} value={user.id}>{user.name} - {user.city || 'No city'}</option>
              ))}
            </select>
            {activeUser && (
              <div className="rounded-xl border border-white/10 bg-black/35 p-4 text-xs">
                <div className="font-black text-white">{activeUser.name}</div>
                <div className="mt-1 text-white/40">{activeUser.email || 'No email'} / {activeUser.phone || 'No phone'}</div>
                <div className="mt-1 text-white/40">{activeUser.ward || 'No ward'}, {activeUser.city || 'No city'}</div>
                <div className="mt-3 flex gap-2">
                  <button onClick={loadSelectedIntoForm} className="rounded-md border border-white/10 px-3 py-1.5 text-[10px] font-bold uppercase text-white/60 hover:text-white">Edit</button>
                  <button onClick={deleteUser} disabled={loading === 'delete'} className="flex items-center gap-1 rounded-md border border-red-400/30 px-3 py-1.5 text-[10px] font-bold uppercase text-red-300 disabled:opacity-50">
                    <Trash2 size={12} />
                    Delete
                  </button>
                </div>
              </div>
            )}
            <div className="text-[10px] font-black uppercase tracking-widest text-white/30">Risk Groups</div>
            <div className="flex flex-wrap gap-2">
              {riskGroupOptions.map((group) => (
                <Pill key={group.key} active={riskGroups.includes(group.key)} onClick={() => toggleValue(group.key, setRiskGroups, riskGroups)}>
                  {group.label}
                </Pill>
              ))}
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Field label="AQI" type="number" value={conditions.aqi} onChange={(value) => setConditions({ ...conditions, aqi: Number(value) })} />
              <Field label="Temp C" type="number" value={conditions.temperature} onChange={(value) => setConditions({ ...conditions, temperature: Number(value) })} />
              <Field label="Rain mm" type="number" value={conditions.rainLastHourMm} onChange={(value) => setConditions({ ...conditions, rainLastHourMm: Number(value) })} />
              <label className="block">
                <span className="mb-1 block text-[9px] font-black uppercase tracking-widest text-white/30">Traffic</span>
                <select value={conditions.traffic} onChange={(event) => setConditions({ ...conditions, traffic: event.target.value })} className="w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none">
                  {['light', 'moderate', 'heavy', 'severe'].map((item) => <option key={item}>{item}</option>)}
                </select>
              </label>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-center gap-2 text-[10px] font-black text-white/40 uppercase tracking-widest">
              <Activity size={15} className="text-traf-acc" />
              Risk + Advisory
            </div>
            <div className="grid grid-cols-2 gap-2">
              <button onClick={loadModelConditions} disabled={loading === 'model'} className="col-span-2 rounded-lg border border-home-acc/40 bg-home-acc/15 px-4 py-3 text-xs font-black uppercase tracking-widest text-home-acc disabled:opacity-50">
                {loading === 'model' ? 'Loading Model' : 'Load From Trained Model'}
              </button>
              <button onClick={assessRisk} disabled={loading === 'risk'} className="rounded-lg border border-traf-acc/40 bg-traf-acc/15 px-4 py-3 text-xs font-black uppercase tracking-widest text-traf-acc disabled:opacity-50">
                Risk
              </button>
              <button onClick={generateAdvisory} disabled={loading === 'advisory'} className="rounded-lg border border-wth-acc/40 bg-wth-acc/15 px-4 py-3 text-xs font-black uppercase tracking-widest text-wth-acc disabled:opacity-50">
                Advisory
              </button>
            </div>
            {riskAssessment && (
              <div className="rounded-xl border border-white/10 bg-black/35 p-4">
                <div className="text-[10px] font-black uppercase tracking-widest text-white/30">Risk Score</div>
                <div className="mt-1 text-4xl font-black text-white">{riskAssessment.score}</div>
                <div className="mt-1 text-xs font-bold uppercase tracking-widest text-red-300">{riskAssessment.riskLevel} / {riskAssessment.severity}</div>
                <div className="mt-3 space-y-2">
                  {(riskAssessment.factors || []).map((factor: any) => (
                    <div key={factor.code} className="rounded-lg bg-white/5 p-2 text-xs text-white/60">
                      <span className="font-bold text-white">{factor.title}</span> +{factor.score}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {advisory && (
              <div className="rounded-xl border border-white/10 bg-black/35 p-4">
                <div className="text-sm font-black text-white">{advisory.title}</div>
                <p className="mt-2 text-xs leading-relaxed text-white/60">{advisory.message}</p>
                <div className="mt-3 text-[10px] font-bold uppercase tracking-widest text-red-300">
                  {advisory.severity} - Score {advisory.riskScore}
                </div>
              </div>
            )}
            <div className="text-[10px] font-black uppercase tracking-widest text-white/30">Channels</div>
            <div className="flex flex-wrap gap-2">
              {channelOptions.map((channel) => (
                <Pill key={channel.key} active={channels.includes(channel.key)} onClick={() => toggleValue(channel.key, setChannels, channels)}>
                  {channel.label}
                </Pill>
              ))}
            </div>
            <button
              onClick={deliverAdvisory}
              disabled={!activeUser || !advisory || loading === 'deliver'}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-home-acc px-4 py-3 text-xs font-black uppercase tracking-widest text-black disabled:opacity-50"
            >
              {loading === 'deliver' ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
              Send / Store
            </button>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-[10px] font-black text-white/40 uppercase tracking-widest">
                <Bell size={15} className="text-wth-acc" />
                In-App Inbox
              </div>
              <button onClick={() => loadNotifications().catch(showError)} className="text-[9px] font-bold uppercase text-white/40 hover:text-white">Refresh</button>
            </div>
            <div className="max-h-[560px] space-y-3 overflow-auto pr-1">
              {notifications.length === 0 && (
                <div className="rounded-xl border border-white/10 bg-black/30 p-4 text-xs text-white/40">
                  No in-app notifications yet.
                </div>
              )}
              {notifications.map((item) => (
                <div key={item.id} className="rounded-xl border border-white/10 bg-black/35 p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="text-sm font-black text-white">{item.title}</div>
                    {!item.is_read && (
                      <button onClick={() => markRead(item.id)} className="text-home-acc">
                        {loading === item.id ? <Loader2 size={14} className="animate-spin" /> : <CheckCircle2 size={14} />}
                      </button>
                    )}
                  </div>
                  <p className="mt-2 text-xs leading-relaxed text-white/60">{item.message}</p>
                  <div className="mt-3 text-[10px] font-bold uppercase tracking-widest text-white/30">
                    {item.status} - {new Date(item.created_at).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {message && (
          <div className="mt-5 rounded-lg border border-white/10 bg-white/5 px-4 py-3 text-xs font-medium text-white/70">
            {message}
          </div>
        )}
      </Card>
    </div>
  );
};

export default CitizenAdvisoryConsole;
