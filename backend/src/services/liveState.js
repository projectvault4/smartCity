// Shared in-process state between the live check (writer) and the decision
// engine (reader). Kept separate to avoid a require cycle.

let latestStats = null;
let updatedAt = 0;

const setLiveStats = (stats) => {
  latestStats = stats;
  updatedAt = Date.now();
};

// Observed-minus-forecast bias per target from the last 24 h of readings,
// only when there are enough points to trust it.
const getLiveBias = ({ minPoints = 3, maxAgeMs = 3 * 60 * 60 * 1000 } = {}) => {
  if (!latestStats || Date.now() - updatedAt > maxAgeMs) return {};
  const out = {};
  ['aqi', 'temperature'].forEach((target) => {
    const s = latestStats[target];
    if (s && s.points >= minPoints && Number.isFinite(s.bias)) {
      out[target] = { bias: s.bias, points: s.points };
    }
  });
  return out;
};

module.exports = { setLiveStats, getLiveBias };
