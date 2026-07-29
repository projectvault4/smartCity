import Card from './Card';

const AnomalyDetectionEmbed = () => (
  <div className="space-y-6">
    <Card title="Anomaly Detection" theme="air">
      <div className="overflow-hidden rounded-xl border border-white/10 bg-black/40">
        <iframe
          title="ForeSightX Anomaly Detection"
          src="/anomaly-detection.html"
          className="h-[calc(100vh-180px)] min-h-[720px] w-full border-0"
        />
      </div>
    </Card>
  </div>
);

export default AnomalyDetectionEmbed;
