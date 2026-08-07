import { useState } from 'react';
import Card from './Card';
import { FileText, Download, Loader2, CheckCircle } from 'lucide-react';
import { jsPDF } from 'jspdf';
import { CityData, ForecastPoint } from '../services/dataService';

type Trend = 'up' | 'down' | 'stable';

const trendFromSeries = (values: number[] | null | undefined): Trend => {
  if (!values || values.length < 2) return 'stable';
  const first = values[0];
  const last = values[values.length - 1];
  const pct = Math.abs(first) > 0 ? (last - first) / first : last - first;
  if (pct > 0.02) return 'up';
  if (pct < -0.02) return 'down';
  return 'stable';
};

const seriesFromForecast = (forecast: ForecastPoint[], pick: (p: ForecastPoint) => number | null): number[] =>
  forecast.map((p) => pick(p)).filter((v): v is number => v !== null);

const fmtTrend = (t: Trend) => (t === 'up' ? 'rising' : t === 'down' ? 'easing' : 'stable');

const ReportGenerator = ({
  data,
  forecast
}: {
  data: CityData;
  forecast?: ForecastPoint[] | null;
}) => {
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);

  const predicted = forecast && forecast.length > 0 ? forecast : null;

  const trafficTrend: Trend = predicted
    ? trendFromSeries(seriesFromForecast(predicted, (p) => p.trafficFlow))
    : data.traffic.trend === 'up' ? 'up' : data.traffic.trend === 'down' ? 'down' : 'stable';
  const airTrend: Trend = predicted
    ? trendFromSeries(seriesFromForecast(predicted, (p) => p.aqi))
    : data.air.trend === 'up' ? 'up' : data.air.trend === 'down' ? 'down' : 'stable';
  const weatherTrend: Trend = predicted
    ? trendFromSeries(seriesFromForecast(predicted, (p) => p.temperature))
    : data.weather.trend === 'up' ? 'up' : data.weather.trend === 'down' ? 'down' : 'stable';
  const energyTrend: Trend = predicted
    ? trendFromSeries(seriesFromForecast(predicted, (p) => p.electricityDemand))
    : data.energy.trend === 'up' ? 'up' : data.energy.trend === 'down' ? 'down' : 'stable';

  const trafficToAir = (() => {
    if (trafficTrend === 'up' && airTrend === 'down') {
      return 'Road volume climbs across the forecast window, yet the model still eases AQI — wind dispersion and secondary factors outweigh added tailpipe emissions in this window.';
    }
    if (trafficTrend === 'up' && airTrend === 'up') {
      return 'Rising road volume concentrates emissions at junctions, and the model carries that directly into the AQI forecast.';
    }
    if (trafficTrend === 'down' && airTrend === 'down') {
      return 'Easing traffic pressure trims road emissions, which the model factors into the near-term AQI call.';
    }
    return 'The model weighs how road emissions interact with wind and temperature before committing to the AQI call.';
  })();

  const tempToEnergy = (() => {
    if (weatherTrend === 'up' && energyTrend === 'up') {
      return 'Rising temperature shifts the baseline for electricity demand as cooling needs adjust upward.';
    }
    if (weatherTrend === 'down' && energyTrend === 'down') {
      return 'Dropping temperature trims cooling load, letting the model ease grid demand across the window.';
    }
    if (weatherTrend === 'up' && energyTrend === 'down') {
      return 'Temperature rises, but the model still eases grid demand — evening hours and reduced activity pull more weight than the cooling signal.';
    }
    return 'The model ties grid load to temperature swings, with cooling and industrial draw moving together.';
  })();

  const reportBody = predicted
    ? `Based on the trained-model 6-hour forecast (T+1 through T+6 from ${predicted[0].timestamp}), Bengaluru traffic is ${fmtTrend(trafficTrend)}, starting near ${Math.round(predicted[0].trafficFlow || 0)} and reaching ${Math.round(predicted[predicted.length - 1].trafficFlow || 0)} vehicles/hr. AQI is ${fmtTrend(airTrend)} (${Math.round(predicted[0].aqi || 0)} moving to ${Math.round(predicted[predicted.length - 1].aqi || 0)}), temperature is ${fmtTrend(weatherTrend)} (${Math.round(predicted[0].temperature || 0)}C toward ${Math.round(predicted[predicted.length - 1].temperature || 0)}C), and grid energy demand is ${fmtTrend(energyTrend)} (${Math.round(predicted[0].electricityDemand || 0)} MW toward ${Math.round(predicted[predicted.length - 1].electricityDemand || 0)} MW). Causal links: ${trafficToAir} ${tempToEnergy}`
    : `Based on high-fidelity telemetry from the B-NODE sensors, Bengaluru is exhibiting ${data.traffic.label.toLowerCase()} traffic flow (${data.traffic.value} veh/hr). Multivariate analysis indicates a causal synchronization between traffic idling in the Central Business District and a localized AQI spike of ${data.air.value}. We project energy demand of ${data.energy.value} MW as residential cooling systems respond to the sustained ambient temperature of ${data.weather.value}C.`;

  const generateReport = async () => {
    setLoading(true);
    setDone(false);

    try {
      await new Promise(resolve => setTimeout(resolve, 900));

      const doc = new jsPDF();

      // Branding Header
      doc.setFillColor(13, 26, 16); // Match app background
      doc.rect(0, 0, 210, 40, 'F');
      doc.setTextColor(34, 197, 94); // Home Acc
      doc.setFontSize(26);
      doc.text("ForeSightX", 20, 25);
      doc.setTextColor(255, 255, 255);
      doc.setFontSize(10);
      doc.text("AUTONOMOUS URBAN PULSE ENGINE", 20, 32);

      doc.setTextColor(0, 0, 0);
      doc.setFontSize(18);
      doc.text("Executive Urban Intelligence Report", 20, 55);

      doc.setFontSize(10);
      doc.setTextColor(100, 100, 100);
      doc.text(`REFERENCE ID: BLR-${Math.floor(Date.now() / 100000)}`, 20, 65);
      doc.text(`TIMESTAMP: ${new Date().toLocaleString()}`, 20, 70);
      doc.text(predicted ? `MODEL FORECAST: ${predicted[0].timestamp}` : 'SOURCE: B-NODE LIVE TELEMETRY', 20, 75);

      // Metrics Table-like structure
      doc.setDrawColor(230, 230, 230);
      doc.line(20, 82, 190, 82);

      doc.setFontSize(12);
      doc.setTextColor(0, 0, 0);
      doc.text("KEY PERFORMANCE INDICATORS", 20, 92);

      const metrics = [
        { label: "Traffic Flow Density", value: `${Math.round(data.traffic.value)} veh/hr`, status: data.traffic.label },
        { label: "Atmospheric Quality Index", value: `${Math.round(data.air.value)} AQI`, status: data.air.label },
        { label: "Grid Energy Demand", value: `${Math.round(data.energy.value)} MW`, status: data.energy.label },
        { label: "Ambient Temperature", value: `${Math.round(data.weather.value)} Celsius`, status: "STABLE" }
      ];

      metrics.forEach((m, i) => {
        const y = 107 + (i * 12);
        doc.setFontSize(10);
        doc.setTextColor(80, 80, 80);
        doc.text(m.label + ":", 20, y);
        doc.setTextColor(0, 0, 0);
        doc.text(m.value, 80, y);
        doc.setFontSize(9);
        doc.setTextColor(m.status === 'Poor' ? 200 : 0, m.status === 'Stable' ? 150 : 0, 0);
        doc.text(m.status.toUpperCase(), 160, y);
      });

      doc.line(20, 157, 190, 157);

      doc.setFontSize(12);
      doc.setTextColor(0, 0, 0);
      doc.text("INTELLIGENT URBAN ANALYSIS", 20, 167);

      doc.setFontSize(10);
      doc.setTextColor(60, 60, 60);
      const splitText = doc.splitTextToSize(reportBody, 170);
      doc.text(splitText, 20, 177);

      doc.setFillColor(245, 245, 245);
      doc.rect(20, 210, 170, 40, 'F');
      doc.setTextColor(0, 0, 0);
      doc.setFontSize(10);
      doc.text("RECOMMENDED ACTIONS:", 30, 222);
      doc.setFontSize(9);
      doc.setTextColor(80, 80, 80);
      const actionLines = predicted
        ? [
            `1. ${trafficTrend === 'up' ? 'Pre-position traffic management for rising arterial flow across the forecast window.' : 'Maintain signal timing — road volume is projected to ease or hold steady.'}`,
            `2. ${energyTrend === 'up' ? 'Verify power load balancing for grid clusters facing rising demand.' : 'Grid demand is easing; no load-balancing intervention is projected.'}`,
            `3. ${airTrend === 'up' ? 'Issue automated advisory for school zones in high-AQI perimeters.' : 'Keep routine AQI advisory cadence — air quality is not projected to worsen.'}`
          ]
        : [
            "1. Implement adaptive signal timing at Silk Board and KR Puram.",
            "2. Verify power load balancing for Southern distribution clusters.",
            "3. Issue automated advisory for school zones in high AQI perimeters."
          ];
      doc.text(actionLines, 30, 230);

      doc.setFontSize(8);
      doc.setTextColor(150, 150, 150);
      doc.text("Generated by ForeSightX Autonomous Urban Pulse Engine. (c) 2026", 70, 285);

      doc.save("Urban_Intelligence_Report.pdf");

      setDone(true);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="City Intelligence Report — Auto Generated" theme="home">
      <div className="p-4 space-y-6 flex flex-col items-center text-center">
        <div className="w-16 h-16 bg-home-acc/10 rounded-full flex items-center justify-center mb-2">
           <FileText size={32} className="text-home-acc" />
        </div>
        <div>
           <h3 className="font-display text-xl font-bold mb-2">Deep Urban Analysis</h3>
           <p className="text-sm text-white/50 max-w-sm mb-6">Click below to use the AI engine to generate a 200-word intelligent urban summary based on multi-domain telemetry.</p>
        </div>

        {predicted && (
          <div className="w-full max-w-sm rounded-xl border border-home-acc/20 bg-home-acc/5 px-4 py-3 text-left text-[11px] text-white/55">
            <div className="mb-1 font-black uppercase tracking-widest text-home-acc/80">Trained model forecast</div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              <span>Traffic: {fmtTrend(trafficTrend)}</span>
              <span>AQI: {fmtTrend(airTrend)}</span>
              <span>Temp: {fmtTrend(weatherTrend)}</span>
              <span>Energy: {fmtTrend(energyTrend)}</span>
            </div>
          </div>
        )}

        <button 
          onClick={generateReport}
          disabled={loading}
          className="flex items-center gap-3 px-8 py-4 bg-home-acc text-black font-extrabold rounded-xl hover:bg-home-acc/90 transition-all disabled:opacity-50"
        >
          {loading ? (
             <>
               <Loader2 size={20} className="animate-spin" />
               GENERATING...
             </>
          ) : done ? (
             <>
               <CheckCircle size={20} />
               REPORT READY
             </>
          ) : (
             <>
               <Download size={20} />
               GENERATE CITY REPORT
             </>
          )}
        </button>

        {done && (
           <p className="text-[11px] text-home-acc font-bold animate-pulse uppercase tracking-[2px]">
             Report downloaded successfully as PDF
           </p>
        )}
      </div>
    </Card>
  );
};

export default ReportGenerator;
