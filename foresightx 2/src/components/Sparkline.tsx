interface SparklineProps {
  color: string;
  points: string;
  predictedPoints?: string;
}

const Sparkline = ({ color, points, predictedPoints = "" }: SparklineProps) => {
  return (
    <svg className="w-full h-[60px] overflow-visible" viewBox="0 0 260 60">
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="2"
        points={points}
      />
      {predictedPoints && (
        <polyline
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          strokeDasharray="5,3"
          opacity="0.45"
          points={predictedPoints}
        />
      )}
      <text x="4" y="58" fontSize="9" fill={color} opacity="0.8">Actual</text>
      {predictedPoints && <text x="70" y="58" fontSize="9" fill={color} opacity="0.4">-- Predicted</text>}
    </svg>
  );
};

export default Sparkline;
