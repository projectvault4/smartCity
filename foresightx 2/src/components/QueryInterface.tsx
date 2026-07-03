import { useState } from 'react';
import Card from './Card';
import { Search, Loader2, BarChart2, TrendingUp, Sparkles } from 'lucide-react';
import { GoogleGenAI } from "@google/genai";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LineChart, Line } from 'recharts';

// Initialize Gemini with the SDK pattern from gemini-api skill
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

const QueryInterface = () => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setResult(null);
    
    try {
      // Current city data for context
      const cityContext = {
        traffic: "5670 vehicles/hr",
        aqi: "156 (Poor)",
        energy: "847 MW",
        temp: "28°C"
      };

      const prompt = `
        You are the Bengaluru Urban Intelligence Assistant. 
        Context:
        - Traffic: ${cityContext.traffic}
        - Air Quality: ${cityContext.aqi}
        - Energy Demand: ${cityContext.energy}
        - Temperature: ${cityContext.temp}
        
        User Query: "${query}"
        
        1. Provide a professional, concise urban analysis (2-3 sentences).
        2. Format the response as a JSON object wrapped in <data> tags.
        3. Include "forecast" (next 6 hours, values as numbers, labels as "HH:00").
        4. Include "shap" (top 4 drivers of the model's prediction).
      `;

      // Using the correct SDK call pattern with system instruction
      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: prompt,
        config: {
          systemInstruction: "Format all your responses as a JSON object inside <data> tags. Even if analysis is natural language, put it in the 'text' field of the JSON. Do not include markdown code blocks around the JSON inside the data tags.",
          responseMimeType: "text/plain"
        }
      });

      const responseText = response.text || "";
      
      const dataMatch = responseText.match(/<data>(.*?)<\/data>/s);
      let structuredData = null;
      if (dataMatch) {
         try {
           structuredData = JSON.parse(dataMatch[1].trim());
         } catch (e) {
           console.error("JSON Parse Error", e);
         }
      }

      if (structuredData) {
        setResult(structuredData);
      } else {
        setResult({ text: responseText.replace(/<data>.*?<\/data>/s, '').trim() });
      }
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
                <p className="text-sm text-white/50 font-medium">Ask the model about city dynamics, predictions, or causal links.</p>
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

                  {result.forecast && (
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

                      {result.shap && (
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
