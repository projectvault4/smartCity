import { useEffect, useMemo, useRef, useState } from 'react';
import { Mic, Pause, Play, Volume2 } from 'lucide-react';
import Card from './Card';
import { CityData, ModelConditions } from '../services/dataService';

const pickVoices = (): SpeechSynthesisVoice[] => {
  const all = window.speechSynthesis?.getVoices?.() || [];
  const english = all.filter((v) => v.lang.toLowerCase().startsWith('en'));
  const pool = english.length ? english : all;

  const seen = new Set<string>();
  const distinct = pool.filter((v) => {
    const key = v.name.toLowerCase().replace(/[^a-z]/g, '');
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  return distinct.slice(0, 3);
};

const fallbackBriefings = {
  today: {
    label: 'Today',
    text: "Good morning, Bengaluru. Here is today's outlook. Traffic is expected to be light, around six thousand one hundred and seventy four vehicles per hour. Air quality is moderate, with an A Q I of fifty five. Temperature will be cool, around twenty four degrees. Electricity demand is projected low, at four thousand and sixty four megawatts. Overall, a calm day across all four domains."
  },
  tomorrow: {
    label: 'Tomorrow',
    text: "Good morning, Bengaluru. Here is tomorrow's projected outlook. Treat this one as a forward estimate rather than a confirmed reading. Traffic is projected to pick up into the moderate range during commute hours. Air quality may drift slightly higher as temperatures rise. Electricity demand is expected to climb into the moderate band by afternoon. Keep an eye on the Ripple Simulation module if temperatures move sharply."
  }
};

type BriefingKey = keyof typeof fallbackBriefings;

const formatNumber = (value?: number | null) => (
  Number.isFinite(Number(value)) ? Math.round(Number(value)).toLocaleString() : 'unknown'
);

const buildModelBriefing = (data: CityData, modelConditions: ModelConditions | null) => {
  if (!modelConditions) {
    return fallbackBriefings;
  }

  const city = modelConditions.city.charAt(0).toUpperCase() + modelConditions.city.slice(1);
  const traffic = formatNumber(modelConditions.traffic?.flow);
  const trafficLevel = modelConditions.traffic?.congestionLevel || data.traffic.label;
  const aqi = formatNumber(modelConditions.aqi?.aqi);
  const temp = formatNumber(modelConditions.weather?.temperature?.value);
  const energy = formatNumber(Number(modelConditions.raw?.electricity_demand ?? data.energy.value));
  const weather = modelConditions.weather?.weather?.description || modelConditions.weather?.weather?.main || data.weather.label;
  const forecastTime = modelConditions.forecastFor
    ? new Date(modelConditions.forecastFor).toLocaleString()
    : 'the next forecast step';

  return {
    today: {
      label: 'Model T+1H',
      text: `Good morning, ${city}. This voice briefing is connected to the trained dataset model. For ${forecastTime}, traffic is forecast at ${traffic} vehicles per hour, in the ${trafficLevel} range. Air quality is forecast at A Q I ${aqi}. Temperature is forecast near ${temp} degrees Celsius with ${weather}. Electricity demand is forecast at ${energy} megawatts. Alerts and advisories should use these same model conditions.`
    },
    tomorrow: {
      label: 'Alert Summary',
      text: `For ${city}, the alert engine should treat the trained model forecast as the active source. The current model signals are traffic ${trafficLevel}, A Q I ${aqi}, temperature ${temp} degrees, and electricity demand ${energy} megawatts. If sensitive users are selected, generate advisories from these model values before delivering in app, email, or S M S notifications.`
    }
  };
};

const VoiceBriefing = ({ data, modelConditions }: { data: CityData; modelConditions: ModelConditions | null }) => {
  const [currentBriefing, setCurrentBriefing] = useState<BriefingKey>('today');
  const [speaking, setSpeaking] = useState(false);
  const [activeWord, setActiveWord] = useState(-1);
  const [barHeights, setBarHeights] = useState([4, 4, 4, 4, 4, 4, 4]);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);
  const eqInterval = useRef<number | null>(null);
  const wordInterval = useRef<number | null>(null);

  useEffect(() => {
    if (!window.speechSynthesis) return;

    const refresh = () => {
      const options = pickVoices();
      setVoices(options);
      if (!selectedVoice && options.length) {
        setSelectedVoice(options[0]);
      }
    };

    refresh();
    window.speechSynthesis.addEventListener('voiceschanged', refresh);
    return () => window.speechSynthesis.removeEventListener('voiceschanged', refresh);
  }, []);

  const briefings = useMemo(() => buildModelBriefing(data, modelConditions), [data, modelConditions]);
  const words = useMemo(() => briefings[currentBriefing].text.split(' '), [briefings, currentBriefing]);

  const stopBriefing = () => {
    setSpeaking(false);
    setActiveWord(-1);
    setBarHeights([4, 4, 4, 4, 4, 4, 4]);

    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }

    if (eqInterval.current) {
      window.clearInterval(eqInterval.current);
      eqInterval.current = null;
    }

    if (wordInterval.current) {
      window.clearInterval(wordInterval.current);
      wordInterval.current = null;
    }
  };

  const playBriefing = () => {
    if (speaking) {
      stopBriefing();
      return;
    }

    setSpeaking(true);
    setActiveWord(-1);

    eqInterval.current = window.setInterval(() => {
      setBarHeights((heights) => heights.map(() => 4 + Math.random() * 24));
    }, 140);

    let wordIndex = 0;
    wordInterval.current = window.setInterval(() => {
      if (wordIndex < words.length) {
        setActiveWord(wordIndex);
        wordIndex += 1;
      } else if (wordInterval.current) {
        window.clearInterval(wordInterval.current);
        wordInterval.current = null;
      }
    }, 180);

    if (window.speechSynthesis) {
      const utterance = new SpeechSynthesisUtterance(briefings[currentBriefing].text);
      if (selectedVoice) {
        utterance.voice = selectedVoice;
      }
      utterance.rate = 0.98;
      utterance.onend = stopBriefing;
      utterance.onerror = stopBriefing;
      window.speechSynthesis.speak(utterance);
    } else {
      window.setTimeout(stopBriefing, words.length * 180 + 500);
    }
  };

  useEffect(() => {
    stopBriefing();
  }, [currentBriefing, selectedVoice]);

  useEffect(() => stopBriefing, []);

  return (
    <div className="space-y-6">
      <Card title="AI Voice Briefing" theme="traffic">
        <div className="mb-5 text-sm text-white/45">
          A spoken briefing generated from the trained dataset model forecast.
        </div>

        <div className="mb-5 rounded-xl border border-white/10 bg-black/30 p-4">
          <div className="mb-3 flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-white/35">
            <Volume2 size={14} className="text-traf-acc" />
            Briefing For
          </div>
          <div className="flex flex-wrap gap-2">
            {(Object.entries(briefings) as [BriefingKey, typeof briefings[BriefingKey]][]).map(([key, briefing]) => (
              <button
                key={key}
                onClick={() => setCurrentBriefing(key)}
                className={`rounded-full border px-4 py-2 text-[10px] font-bold uppercase tracking-widest transition ${
                  currentBriefing === key
                    ? 'border-traf-acc bg-traf-acc/15 text-traf-acc'
                    : 'border-white/10 bg-white/5 text-white/40 hover:text-white/70'
                }`}
              >
                {briefing.label}
              </button>
            ))}
          </div>

          <div className="mt-4 border-t border-white/10 pt-3">
            <div className="mb-2.5 flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-white/35">
              <Mic size={13} className="text-traf-acc" />
              Choose a voice
            </div>
            {voices.length === 0 ? (
              <div className="text-[11px] text-white/35">No extra voices available — using your browser default.</div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {voices.map((voice) => (
                  <button
                    key={voice.name}
                    onClick={() => setSelectedVoice(voice)}
                    className={`rounded-full border px-4 py-2 text-[10px] font-bold tracking-widest transition ${
                      selectedVoice?.name === voice.name
                        ? 'border-traf-acc bg-traf-acc/15 text-traf-acc'
                        : 'border-white/10 bg-white/5 text-white/40 hover:text-white/70'
                    }`}
                  >
                    {voice.name}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-white/10 bg-black/35 p-5">
          <div className="flex flex-wrap items-center gap-6">
            <button
              onClick={playBriefing}
              className={`flex h-[72px] w-[72px] items-center justify-center rounded-full bg-gradient-to-br from-traf-acc to-amber-500 text-[#1a1200] shadow-lg shadow-traf-acc/20 transition hover:scale-105 ${
                speaking ? 'animate-pulse' : ''
              }`}
              aria-label={speaking ? 'Stop voice briefing' : 'Play voice briefing'}
            >
              {speaking ? <Pause size={28} fill="currentColor" /> : <Play size={28} fill="currentColor" />}
            </button>

            <div className="flex h-8 items-end gap-1">
              {barHeights.map((height, index) => (
                <div
                  key={index}
                  className="w-1 rounded-full bg-traf-acc transition-[height] duration-150"
                  style={{ height }}
                />
              ))}
            </div>

            <p className="min-w-[240px] flex-1 text-sm leading-8 text-white/60">
              {words.map((word, index) => (
                <span
                  key={`${word}-${index}`}
                  className={index === activeWord ? 'font-bold text-traf-acc' : undefined}
                >
                  {word}{' '}
                </span>
              ))}
            </p>
          </div>
        </div>

        <div className="mt-5 rounded-xl border border-traf-acc/20 bg-traf-acc/5 px-4 py-3 text-xs leading-relaxed text-white/45">
          Source: {modelConditions?.source || 'fallback forecast'} · Uses your browser's built-in text-to-speech.
        </div>
      </Card>
    </div>
  );
};

export default VoiceBriefing;
