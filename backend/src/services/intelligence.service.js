const fs = require('fs');
const path = require('path');

const config = require('../config/env');

// Artifacts are produced offline by build_intelligence.py (repo root) and baked
// into the backend image under outputs/intelligence/.
const artifactDir = () => path.join(config.modelForecast.projectRoot, 'outputs', 'intelligence');

const cache = new Map();

const loadArtifact = (name) => {
  const file = path.join(artifactDir(), name);
  if (!fs.existsSync(file)) {
    const error = new Error(`Intelligence artifact not found: ${name}. Run build_intelligence.py.`);
    error.statusCode = 503;
    throw error;
  }
  const { mtimeMs } = fs.statSync(file);
  const hit = cache.get(file);
  if (hit && hit.mtimeMs === mtimeMs) return hit.data;
  const data = JSON.parse(fs.readFileSync(file, 'utf8'));
  cache.set(file, { mtimeMs, data });
  return data;
};

const getPolicy = () => loadArtifact('policy.json');
const getManifest = () => loadArtifact('manifest.json');
const getBenchmark = () => loadArtifact('benchmark.json');
const getConformal = () => loadArtifact('conformal.json');
const getShapSummary = () => loadArtifact('shap_summary.json');
const getWhatIf = () => loadArtifact('whatif.json');
const getAnomalyEval = () => loadArtifact('anomaly_eval.json');
const getOutlookUncertainty = () => loadArtifact('outlook_uncertainty.json');

// The replay has ~500 hours; the player pages through it.
const getReplay = ({ offset = 0, limit } = {}) => {
  const replay = loadArtifact('replay.json');
  const { hours, ...rest } = replay;
  const start = Math.max(0, Number(offset) || 0);
  const end = limit ? start + Math.max(1, Number(limit)) : hours.length;
  return { ...rest, totalHours: hours.length, offset: start, hours: hours.slice(start, end) };
};

const getProof = () => {
  const manifest = getManifest();
  const benchmark = getBenchmark();
  const conformal = getConformal();
  const anomaly = getAnomalyEval();
  const replay = getReplay({ limit: 1 });
  return {
    manifest,
    benchmark,
    conformal,
    anomaly,
    replay: { window: replay.window, cooldownHours: replay.cooldown_hours, scorecard: replay.scorecard, events: replay.events }
  };
};

module.exports = {
  artifactDir,
  loadArtifact,
  getPolicy,
  getManifest,
  getBenchmark,
  getConformal,
  getShapSummary,
  getWhatIf,
  getAnomalyEval,
  getOutlookUncertainty,
  getReplay,
  getProof
};
