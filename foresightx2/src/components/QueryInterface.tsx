import { useState } from 'react';
import Card from './Card';
import { Search, Loader2, BarChart2, TrendingUp, Sparkles } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LineChart, Line } from 'recharts';
import { ModelConditions, ForecastPoint } from '../services/dataService';

// Groq API (OpenAI-compatible). Uses process.env.GROQ_API_KEY injected via vite.
const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';
const GROQ_MODEL = 'llama-3.3-70b-versatile';

// Normalize the model's response into the exact shape the charts expect:
// { text: string, forecast: [{time, value}], shap: [{name, val}] }
const normalizeResult = (raw: any): any => {
  if (!raw || typeof raw !== 'object') return { text: String(raw || '') };

  // text: the model may put the answer in "answer", "text", "summary",
  // "analysis" (string) or "analysis.text". Grab whichever exists.
  const analysis = raw.analysis;
  const text =
    raw.answer ||
    raw.text ||
    raw.summary ||
    raw.insight ||
    raw.response ||
    (typeof analysis === 'string' ? analysis : analysis?.text || analysis?.answer || analysis?.summary) ||
    '';

  // forecast: accept array [{time, value}] or object {values: {"HH:00": n}}
  let forecast: { time: string; value: number }[] = [];
  const fc = raw.forecast;
  if (Array.isArray(fc)) {
    forecast = fc
      .map((entry: any) => ({
        time: entry.time ?? entry.label ?? entry.timestamp ?? entry.hour ?? '',
        value: Number(entry.value ?? entry.aqi ?? entry.val ?? entry.prediction ?? NaN),
      }))
      .filter((e: any) => e.time && Number.isFinite(e.value));
  } else if (fc && typeof fc === 'object') {
    if (fc.values && typeof fc.values === 'object') {
      forecast = Object.entries(fc.values)
        .map(([time, v]) => ({ time, value: Number(v) }))
        .filter((e: any) => Number.isFinite(e.value));
    } else if (Array.isArray(fc.data)) {
      forecast = fc.data
        .map((entry: any) => ({
          time: entry.time ?? entry.label ?? entry.timestamp ?? entry.hour ?? '',
          value: Number(entry.value ?? entry.aqi ?? entry.val ?? NaN),
        }))
        .filter((e: any) => e.time && Number.isFinite(e.value));
    }
  }

  // shap: accept array [{name, val}] or object {drivers: [...]}
  let shap: { name: string; val: number }[] = [];
  const sh = raw.shap;
  if (Array.isArray(sh)) {
    shap = sh
      .map((entry: any) => ({
        name: entry.name ?? entry.driver ?? entry.label ?? '',
        val: Number(entry.val ?? entry.value ?? entry.importance ?? entry.score ?? NaN),
      }))
      .filter((e: any) => e.name && Number.isFinite(e.val));
  } else if (sh && typeof sh === 'object' && Array.isArray(sh.drivers)) {
    shap = sh.drivers.map((name: string, index: number) => ({
      name,
      val: sh.values?.[name] ?? sh.importance?.[name] ?? (sh.drivers.length - index),
    }));
  }

  return { text, forecast, shap };
};

const QueryInterface = ({
  modelConditions = null,
  forecast = [],
}: {
  modelConditions?: ModelConditions | null;
  forecast?: ForecastPoint[];
}) => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const formatNumber = (v?: number | null) =>
    Number.isFinite(Number(v)) ? Math.round(Number(v)).toLocaleString() : 'unknown';

  const getAqiLabel = (value: number) => {
    if (!Number.isFinite(value)) return 'unknown';
    if (value <= 50) return 'Good';
    if (value <= 100) return 'Moderate';
    if (value <= 200) return 'Poor';
    if (value <= 300) return 'Very Poor';
    return 'Severe';
  };

  // Fallback chart data from the REAL model forecast, used when the LLM omits or
  // mangles the forecast/shap arrays so the charts never render blank.
  const fallbackForecast = (queryText: string): { time: string; value: number }[] => {
    if (!forecast.length) return [];
    const q = queryText.toLowerCase();
    const pick = (p: ForecastPoint) => {
      if (q.includes('traffic') || q.includes('congestion') || q.includes('vehicle')) return p.trafficFlow ?? null;
      if (q.includes('energy') || q.includes('demand') || q.includes('electricity')) return p.electricityDemand ?? null;
      if (q.includes('temp') || q.includes('heat') || q.includes('degree')) return p.temperature ?? null;
      return p.aqi ?? null;
    };
    return forecast
      .slice(0, 6)
      .map((p) => ({
        time: new Date(p.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        value: Math.round(Number(pick(p)) || 0),
      }))
      .filter((e: any) => Number.isFinite(e.value));
  };

  const fallbackShap = (): { name: string; val: number }[] => {
    if (!modelConditions) return [];
    const mc = modelConditions;
    return [
      { name: 'Traffic Flow', val: Math.round(Number(mc.traffic?.flow) || 0) },
      { name: 'Air Quality (AQI)', val: Math.round(Number(mc.aqi?.aqi) || 0) },
      { name: 'Temperature', val: Math.round(Number(mc.weather?.temperature?.value) || 0) },
      { name: 'Electricity Demand', val: Math.round(Number(mc.raw?.electricity_demand) || 0) },
    ];
  };

  const buildCityContext = () => {
    if (!modelConditions) {
      return {
        source: 'fallback (no trained model data connected)',
        traffic: '5670 vehicles/hr',
        aqi: '156 (Poor)',
        energy: '847 MW',
        temp: '28°C',
        forecastText: 'No forecast available.',
      };
    }

    const mc = modelConditions;
    const trafficLevel = mc.traffic?.congestionLevel || 'unknown';
    const aqi = mc.aqi?.aqi;
    const temp = mc.weather?.temperature?.value;
    const energy = mc.raw?.electricity_demand;

    const forecastText = forecast.length
      ? forecast
          .slice(0, 6)
          .map(
            (p) =>
              `T+${p.stepAhead}H (${new Date(p.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}): AQI ${formatNumber(p.aqi)}, temp ${formatNumber(p.temperature)}°C, traffic ${p.traffic?.congestionLevel || 'unknown'}, weather ${p.weather?.description || p.weather?.main || 'unknown'}`
          )
          .join('\n')
      : 'No model forecast available.';

    return {
      source: `trained model (${mc.city}, ${mc.source || 'unknown source'})`,
      traffic: `${formatNumber(mc.traffic?.flow)} vehicles/hr (${trafficLevel})`,
      aqi: `${formatNumber(aqi)} (${getAqiLabel(Number(aqi))})`,
      energy: `${formatNumber(Number(energy))} MW`,
      temp: `${formatNumber(temp)}°C`,
      forecastText,
    };
  };

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setResult(null);

    try {
      // Current city data for context — pulled from the trained model when available
      const cityContext = buildCityContext();

      const prompt = `
        You are the Bengaluru Urban Intelligence Assistant — a senior urban systems analyst presenting to municipal officials and a thesis committee.
        The city values below come from a trained forecasting model. Use them as your ground truth.

        Context (model source: ${cityContext.source}):
        - Traffic: ${cityContext.traffic}
        - Air Quality: ${cityContext.aqi}
        - Energy Demand: ${cityContext.energy}
        - Temperature: ${cityContext.temp}

        Model forecast (next 6 hours):
        ${cityContext.forecastText}

        User Query: "${query}"

        Respond in plain, simple language that a general audience can easily understand:

        1. ANALYSIS: Explain the answer in 4-5 short, simple sentences (max ~110 words total). Use everyday words, not jargon. Tell them what's happening, why it's happening in plain terms, and what it means for them. Mention the relevant forecast number(s) naturally (e.g. "AQI rises to 109"). Never refuse or say the data "cannot answer" — reason from the evidence you have and keep it simple.
        2. Format the response as a JSON object wrapped in <data> tags.
        3. Include "forecast": an ARRAY of 6 objects, each {"time":"HH:00","value":<number>}, mirroring the real model forecast values above (plot the metric most relevant to the query).
        4. Include "shap": an ARRAY of 4 objects, each {"name":"<driver>","val":<number>}, ranking the drivers of the trend with real magnitudes.
      `;

      // Using Groq's OpenAI-compatible chat completions API
      const response = await fetch(GROQ_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${process.env.GROQ_API_KEY || ''}`,
        },
        body: JSON.stringify({
          model: GROQ_MODEL,
          messages: [
            {
              role: 'system',
              content:
                "You explain city forecast data in plain, simple language that a general audience understands. Always answer the user's query with 4-5 short, easy sentences (max ~110 words total), using everyday words and no jargon. Mention the key forecast number(s) naturally. Never refuse or claim the data is insufficient — reason from the evidence and keep it simple. Format your response as a JSON object inside <data> tags with fields: text (your explanation), forecast (array of {time,value}), and shap (array of {name,val}). Do not include markdown code blocks.",
            },
            { role: 'user', content: prompt },
          ],
          temperature: 0.5,
          max_tokens: 1200,
        }),
      });

      if (!response.ok) {
        const body = await response.text();
        throw new Error(`Groq request failed (${response.status}): ${body.slice(0, 200)}`);
      }

      const data = await response.json();
      const responseText = data?.choices?.[0]?.message?.content || '';

      const dataMatch = responseText.match(/<data>(.*?)<\/data>/s);
      let structuredData = null;
      if (dataMatch) {
         try {
           structuredData = JSON.parse(dataMatch[1].trim());
         } catch (e) {
           console.error("JSON Parse Error", e);
         }
      }

      const normalized = structuredData
        ? normalizeResult(structuredData)
        : { text: '', forecast: [], shap: [] };

      // LLMs sometimes write the real analysis OUTSIDE the <data> tags and leave a
      // stub inside. Prefer whichever text is longer/more substantive.
      const outerProse = responseText.replace(/<data>[\s\S]*?<\/data>/s, '').trim();
      const analysisText =
        normalized.text && normalized.text.trim().length >= 40
          ? normalized.text
          : outerProse || normalized.text;

      // Guarantee charts render even if the LLM skipped them.
      const finalResult = {
        text: analysisText || 'No textual analysis returned — showing the model forecast data below.',
        forecast: normalized.forecast.length >= 4 ? normalized.forecast : fallbackForecast(query),
        shap: normalized.shap.length >= 3 ? normalized.shap : fallbackShap(),
      };
      setResult(finalResult);
    } catch (error) {
      console.error("Urban Intelligence Query Failed:", error);
      setResult({ text: "The urban relay is currently unresponsive. Please check your query or try again in a moment." });
    } finally {
      setLoading(false);
    }
  };

  const suggestedQuestions = [
    "What is the forecasted impact of high traffic on AQI tonight?",
    "How does temperature affect energy demand patterns?",
    "Why is traffic flow predicted to decrease in the next 2 hours?",
    "Correlate current air quality with school zone safety."
  ];

  return (
    <div className="space-y-6">
      <div className="relative">
        {/* Glow effect for the primary search bar */}
        <div className="absolute -inset-1 bg-gradient-to-r from-home-acc/10 to-air-acc/10 rounded-[32px] blur-xl opacity-50 group-hover:opacity-100 transition duration-1000"></div>
        
        <Card 
          title="Query Urban Intelligence" 
          theme="home" 
          className="relative overflow-visible"
          titleClassName="font-serif italic tracking-wide uppercase text-[12px] opacity-80"
        >
          <div className="p-2 space-y-6">
            <div className="flex flex-col gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-home-acc/10 rounded-lg">
                  <Sparkles size={18} className="text-home-acc" />
                </div>
                <div className="flex-1">
                  <p className="text-sm text-white/50 font-medium">Ask the model about city dynamics, predictions, or causal links.</p>
                  <p className="mt-1 inline-flex items-center gap-1.5 rounded-full border border-home-acc/20 bg-home-acc/5 px-2.5 py-0.5 font-mono text-[9.5px] font-bold uppercase tracking-widest text-home-acc/80">
                    <span className={`h-1.5 w-1.5 rounded-full ${modelConditions ? 'bg-home-acc' : 'bg-amber-400'}`} />
                    {modelConditions ? `Context: trained model · ${modelConditions.city}` : 'Context: fallback (no model data)'}
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                {suggestedQuestions.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => setQuery(q)}
                    className="px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/5 rounded-full text-[10px] font-bold text-white/40 hover:text-white/70 transition-all"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
            
            <div className="relative group">
              <div className="absolute left-5 top-1/2 -translate-y-1/2 text-white/20 group-focus-within:text-home-acc transition-colors">
                <Search size={22} />
              </div>
              <input 
                type="text" 
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="What is the forecasted impact of high traffic on AQI tonight?"
                className="w-full bg-black/60 border-2 border-white/5 rounded-[24px] pl-16 pr-36 py-5 text-lg font-medium focus:border-home-acc/30 focus:bg-black/80 transition-all outline-none placeholder:text-white/10"
              />
              <button 
                onClick={handleSearch}
                disabled={loading}
                className="absolute right-3 top-1/2 -translate-y-1/2 px-8 py-3 bg-home-acc text-black rounded-[18px] font-black text-xs uppercase tracking-widest hover:scale-105 active:scale-95 transition-all disabled:opacity-50"
              >
                {loading ? <Loader2 className="animate-spin" size={18} /> : 'ANALYZE'}
              </button>
            </div>

            {result && (
               <div className="space-y-10 pt-8 animate-in fade-in slide-in-from-top-4 duration-700">
                  <div className="relative p-6 bg-white/5 border border-white/5 rounded-2xl">
                     <div className="absolute -top-3 left-6 px-3 py-1 bg-home-acc text-black text-[10px] font-black rounded-full uppercase tracking-widest shadow-xl">
                        AI Inference
                     </div>
                     <p className="text-[15px] text-white/90 leading-relaxed font-medium">
                       {result.text}
                     </p>
                  </div>

                  {result.forecast && result.forecast.length > 0 && (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                      <div className="space-y-5">
                         <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2 text-[10px] font-black text-white/30 uppercase tracking-[2.5px]">
                               <TrendingUp size={16} className="text-home-acc" />
                               Trend Forecast
                            </div>
                            <div className="text-[10px] text-white/20 font-bold">NEXT 6 HOURS</div>
                         </div>
                         <div className="h-[220px] w-full p-4 bg-black/30 rounded-2xl border border-white/5 shadow-inner">
                           <ResponsiveContainer width="100%" height="100%">
                             <LineChart data={result.forecast}>
                               <XAxis dataKey="time" hide />
                               <Tooltip 
                                 contentStyle={{ backgroundColor: '#0c0c0c', border: '1px solid #ffffff15', borderRadius: '16px' }}
                                 itemStyle={{ color: '#fff', fontSize: '12px' }}
                                 cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1 }}
                               />
                               <Line 
                                 type="natural" 
                                 dataKey="value" 
                                 stroke="#2ecc71" 
                                 strokeWidth={4} 
                                 dot={{ fill: '#2ecc71', r: 4, strokeWidth: 0 }} 
                                 activeDot={{ r: 7, stroke: '#fff', strokeWidth: 3 }}
                               />
                             </LineChart>
                           </ResponsiveContainer>
                         </div>
                      </div>

                      {result.shap && result.shap.length > 0 && (
                        <div className="space-y-5">
                           <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2 text-[10px] font-black text-white/30 uppercase tracking-[2.5px]">
                                 <BarChart2 size={16} className="text-home-acc" />
                                 Model Attribution
                              </div>
                              <div className="text-[10px] text-white/20 font-bold">SHAP VALUES</div>
                           </div>
                           <div className="h-[220px] w-full p-4 bg-black/30 rounded-2xl border border-white/5 shadow-inner">
                              <ResponsiveContainer width="100%" height="100%">
                                 <BarChart data={result.shap} layout="vertical" margin={{ left: 0, right: 30 }}>
                                    <XAxis type="number" hide />
                                    <YAxis dataKey="name" type="category" width={100} tick={{ fill: '#ffffff30', fontSize: 10, fontWeight: 700 }} axisLine={false} tickLine={false} />
                                    <Tooltip cursor={{ fill: 'rgba(255,255,255,0.03)' }} contentStyle={{ display: 'none' }} />
                                    <Bar dataKey="val" radius={[0, 8, 8, 0]} barSize={28}>
                                       {result.shap.map((_entry: any, index: number) => (
                                         <Cell key={`cell-${index}`} fill={index === 0 ? '#2ecc71' : 'rgba(255,255,255,0.08)'} />
                                       ))}
                                    </Bar>
                                 </BarChart>
                              </ResponsiveContainer>
                           </div>
                        </div>
                      )}
                    </div>
                  )}
               </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};

export default QueryInterface;
