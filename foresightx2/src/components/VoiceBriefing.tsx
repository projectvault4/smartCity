import { useEffect, useMemo, useRef, useState } from 'react';
import { Mic, Pause, Play, Volume2 } from 'lucide-react';
import Card from './Card';
import { backendApi, Briefing, CityData, ModelConditions, PROFILE_LABELS } from '../services/dataService';

// Evidence-based voice briefing: what is different from a normal hour, why
// (SHAP drivers), how sure (90% range / probability) and what to do for the
// listener's profile. Same decision object as the alerts and SMS.

type Lang = 'en' | 'kn';
const PROFILES = ['general', 'commuter', 'outdoor_worker', 'respiratory', 'elderly', 'child'];

// macOS ships joke voices that would derail a demo.
const NOVELTY = /^(albert|bad news|bahh|bells|boing|bubbles|cellos|good news|jester|organ|superstar|trinoids|whisper|wobble|zarvox|junior|ralph|fred|kathy)\b/i;

const voicesFor = (lang: Lang): SpeechSynthesisVoice[] => {
  const all = (window.speechSynthesis?.getVoices?.() || []).filter((v) => !NOVELTY.test(v.name));
  const prefix = lang === 'kn' ? 'kn' : 'en';
  const matching = all.filter((v) => v.lang.toLowerCase().startsWith(prefix));
  if (lang === 'en') {
    // Prefer Indian English voices, then any English voice.
    return [...matching.filter((v) => v.lang === 'en-IN'), ...matching.filter((v) => v.lang !== 'en-IN')].slice(0, 6);
  }
  return matching;
};

const VoiceBriefing = (_props: { data?: CityData; modelConditions?: ModelConditions | null }) => {
  const [lang, setLang] = useState<Lang>('en');
  const [profile, setProfile] = useState('commuter');
  const [briefing, setBriefing] = useState<Briefing | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [activeSentence, setActiveSentence] = useState(-1);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);
  const cancelled = useRef(false);

  useEffect(() => {
    let mounted = true;
    setBriefing(null);
    backendApi.briefing(profile === 'general' ? [] : [profile], lang)
      .then((r) => { if (mounted) { setBriefing(r.data); setError(null); } })
      .catch((e) => { if (mounted) setError(e.message); });
    return () => { mounted = false; };
  }, [profile, lang]);

  useEffect(() => {
    if (!window.speechSynthesis) return undefined;
    const refresh = () => {
      const list = voicesFor(lang);
      setVoices(list);
      setSelectedVoice(list[0] || null);
    };
    refresh();
    window.speechSynthesis.addEventListener('voiceschanged', refresh);
    return () => window.speechSynthesis.removeEventListener('voiceschanged', refresh);
  }, [lang]);

  const sentences = useMemo(() => briefing?.sentences || [], [briefing]);

  const stop = () => {
    cancelled.current = true;
    window.speechSynthesis?.cancel();
    setSpeaking(false);
    setActiveSentence(-1);
  };

  const play = () => {
    if (speaking) { stop(); return; }
    if (!window.speechSynthesis || !sentences.length) return;
    cancelled.current = false;
    setSpeaking(true);
    // Speak sentence by sentence so the current one can be highlighted.
    const speakAt = (i: number) => {
      if (cancelled.current || i >= sentences.length) { setSpeaking(false); setActiveSentence(-1); return; }
      setActiveSentence(i);
      const u = new SpeechSynthesisUtterance(sentences[i].replace(/AQI/g, 'A Q I'));
      u.lang = lang === 'kn' ? 'kn-IN' : 'en-IN';
      if (selectedVoice) u.voice = selectedVoice;
      u.rate = 0.97;
      u.onend = () => speakAt(i + 1);
      u.onerror = () => { setSpeaking(false); setActiveSentence(-1); };
      window.speechSynthesis.speak(u);
    };
    speakAt(0);
  };

  useEffect(() => stop, []);
  useEffect(() => { stop(); }, [profile, lang]);

  return (
    <div className="space-y-6">
      <Card title="AI Voice Briefing" theme="traffic">
        <div className="mb-5 text-sm text-white/50">
          Spoken from the same decision engine as the alerts: what is unusual, why, how sure, and what to do for you.
        </div>

        <div className="mb-5 rounded-xl border border-white/10 bg-black/30 p-4 space-y-4">
          <div>
            <div className="mb-2 flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-white/35"><Volume2 size={14} className="text-traf-acc" />Briefing for</div>
            <div className="flex flex-wrap gap-2">
              {PROFILES.map((p) => (
                <button key={p} onClick={() => setProfile(p)}
                  className={`rounded-full border px-4 py-2 text-[10px] font-bold uppercase tracking-widest ${profile === p ? 'border-traf-acc bg-traf-acc/15 text-traf-acc' : 'border-white/10 bg-white/5 text-white/40 hover:text-white/70'}`}>
                  {PROFILE_LABELS[p]}
                </button>
              ))}
            </div>
          </div>
          <div>
            <div className="mb-2 text-[10px] font-black uppercase tracking-widest text-white/35">Language</div>
            <div className="flex gap-2">
              {(['en', 'kn'] as Lang[]).map((l) => (
                <button key={l} onClick={() => setLang(l)}
                  className={`rounded-full border px-4 py-2 text-[11px] font-bold ${lang === l ? 'border-traf-acc bg-traf-acc/15 text-traf-acc' : 'border-white/10 bg-white/5 text-white/40'}`}>
                  {l === 'en' ? 'English' : 'ಕನ್ನಡ (Kannada)'}
                </button>
              ))}
            </div>
          </div>
          <div className="border-t border-white/10 pt-3">
            <div className="mb-2 flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-white/35"><Mic size={13} className="text-traf-acc" />Voice</div>
            {voices.length === 0 ? (
              <div className="text-[11px] text-white/40">
                {lang === 'kn'
                  ? 'This browser has no Kannada voice installed. The Kannada text is shown below; Chrome on Android usually includes a Kannada voice.'
                  : 'Using your browser default voice.'}
              </div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {voices.map((v) => (
                  <button key={v.name} onClick={() => setSelectedVoice(v)}
                    className={`rounded-full border px-4 py-2 text-[10px] font-bold tracking-widest ${selectedVoice?.name === v.name ? 'border-traf-acc bg-traf-acc/15 text-traf-acc' : 'border-white/10 bg-white/5 text-white/40'}`}>
                    {v.name}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-white/10 bg-black/35 p-5">
          {error && <p className="text-red-300 text-sm mb-3">Briefing unavailable: {error}</p>}
          <div className="flex flex-wrap items-start gap-6">
            <button onClick={play} disabled={!sentences.length || (lang === 'kn' && voices.length === 0)}
              className={`flex h-[72px] w-[72px] items-center justify-center rounded-full bg-gradient-to-br from-traf-acc to-amber-500 text-[#1a1200] shadow-lg disabled:opacity-40 ${speaking ? 'animate-pulse' : ''}`}
              aria-label={speaking ? 'Stop voice briefing' : 'Play voice briefing'}>
              {speaking ? <Pause size={28} fill="currentColor" /> : <Play size={28} fill="currentColor" />}
            </button>
            <div className="min-w-[240px] flex-1 space-y-2">
              {!briefing && !error && <p className="text-white/40 text-sm">Building briefing…</p>}
              {sentences.map((s, i) => (
                <p key={i} className={`text-sm leading-relaxed ${i === activeSentence ? 'text-traf-acc font-semibold' : 'text-white/65'}`}>{s}</p>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-5 rounded-xl border border-traf-acc/20 bg-traf-acc/5 px-4 py-3 text-xs leading-relaxed text-white/45">
          Source: decision engine (2026 outlook, 90% ranges backtested on held-out data, SHAP drivers). Uses the browser's text-to-speech.
          {lang === 'kn' && briefing ? ` ${briefing.kannadaNote}` : ''}
        </div>
      </Card>
    </div>
  );
};

export default VoiceBriefing;
