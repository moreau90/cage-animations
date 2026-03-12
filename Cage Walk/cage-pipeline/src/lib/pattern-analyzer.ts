/**
 * Pattern Analyzer — queries Supabase for run families and finds
 * parameter-score correlations across mutation batches.
 */
import pino from 'pino';
import { getSupabase } from '../config/supabase.js';

const log = pino({ name: 'pattern-analyzer' });

export interface Correlation {
  parameter: string;
  metric: string;
  correlation: number;
  direction: 'positive' | 'negative';
  sampleCount: number;
}

export interface Insight {
  pattern: string;
  confidence: number;
  supporting_runs: string[];
  correlations: Correlation[];
}

/**
 * Analyze a family of runs (parent + children) to find parameter-score correlations.
 *
 * @param parentRunId - The parent run that spawned child mutations
 * @returns Array of insights found
 */
export async function analyzeRunFamily(parentRunId: string): Promise<Insight[]> {
  const sb = getSupabase();

  // Get child runs
  const { data: childRuns, error: runsErr } = await sb
    .from('runs')
    .select('id, status, decision')
    .eq('parent_run_id', parentRunId);

  if (runsErr) throw new Error(`Failed to query child runs: ${runsErr.message}`);
  if (!childRuns || childRuns.length < 2) {
    log.info({ parentRunId, childCount: childRuns?.length ?? 0 }, 'Too few children for analysis');
    return [];
  }

  const childIds = childRuns.map(r => r.id);

  // Get configs for all children (contains mutation details)
  const { data: configs } = await sb
    .from('run_configs')
    .select('run_id, config')
    .in('run_id', childIds);

  // Get scores for all children
  const { data: scores } = await sb
    .from('scores')
    .select('run_id, metric, value')
    .in('run_id', childIds);

  if (!configs || !scores || scores.length === 0) {
    return [];
  }

  // Build per-run data
  const configMap = new Map<string, Record<string, unknown>>();
  for (const cfg of configs) {
    configMap.set(cfg.run_id, cfg.config as Record<string, unknown>);
  }

  const scoreMap = new Map<string, Map<string, number>>();
  for (const s of scores) {
    if (!scoreMap.has(s.run_id)) scoreMap.set(s.run_id, new Map());
    scoreMap.get(s.run_id)!.set(s.metric, s.value);
  }

  // Extract numeric parameters from mutation configs
  const paramValues = new Map<string, number[]>(); // param name → values
  const paramRunIds = new Map<string, string[]>();  // param name → runIds
  const metricsSet = new Set<string>();

  for (const runId of childIds) {
    const cfg = configMap.get(runId);
    if (!cfg) continue;

    const mutations = cfg.mutations as Record<string, unknown> | undefined;
    if (!mutations) continue;

    // Extract scalar mutation parameters
    for (const [key, val] of Object.entries(mutations)) {
      if (typeof val === 'number') {
        if (!paramValues.has(key)) {
          paramValues.set(key, []);
          paramRunIds.set(key, []);
        }
        paramValues.get(key)!.push(val);
        paramRunIds.get(key)!.push(runId);
      }
    }

    // Also extract top-level numeric config values
    for (const [key, val] of Object.entries(cfg)) {
      if (key === 'mutations' || key === 'mutation_strategy') continue;
      if (typeof val === 'number') {
        if (!paramValues.has(key)) {
          paramValues.set(key, []);
          paramRunIds.set(key, []);
        }
        paramValues.get(key)!.push(val);
        paramRunIds.get(key)!.push(runId);
      }
    }

    // Collect metric names
    const runScores = scoreMap.get(runId);
    if (runScores) {
      for (const metric of runScores.keys()) {
        metricsSet.add(metric);
      }
    }
  }

  // Compute Pearson correlation for each (parameter, metric) pair
  const correlations: Correlation[] = [];

  for (const [paramName, pValues] of paramValues) {
    if (pValues.length < 3) continue; // need at least 3 data points
    const runIds = paramRunIds.get(paramName)!;

    for (const metric of metricsSet) {
      // Align parameter values with metric values
      const xs: number[] = [];
      const ys: number[] = [];

      for (let i = 0; i < runIds.length; i++) {
        const mVal = scoreMap.get(runIds[i])?.get(metric);
        if (mVal !== undefined && isFinite(mVal)) {
          xs.push(pValues[i]);
          ys.push(mVal);
        }
      }

      if (xs.length < 3) continue;

      const r = pearsonCorrelation(xs, ys);
      if (isNaN(r) || !isFinite(r)) continue;
      if (Math.abs(r) < 0.3) continue; // weak correlations not interesting

      correlations.push({
        parameter: paramName,
        metric,
        correlation: r,
        direction: r > 0 ? 'positive' : 'negative',
        sampleCount: xs.length,
      });
    }
  }

  // Sort by absolute correlation strength
  correlations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));

  // Build insights from top correlations
  const insights: Insight[] = [];
  const seen = new Set<string>();

  for (const corr of correlations.slice(0, 10)) {
    const key = `${corr.parameter}→${corr.metric}`;
    if (seen.has(key)) continue;
    seen.add(key);

    const dir = corr.direction === 'positive' ? 'increases' : 'decreases';
    insights.push({
      pattern: `Higher ${corr.parameter} ${dir} ${corr.metric} (r=${corr.correlation.toFixed(3)}, n=${corr.sampleCount})`,
      confidence: Math.abs(corr.correlation),
      supporting_runs: childIds,
      correlations: [corr],
    });
  }

  // Also check accept/reject patterns
  const acceptedRunIds = childRuns.filter(r => r.decision === 'accept').map(r => r.id);
  const rejectedRunIds = childRuns.filter(r => r.decision === 'reject').map(r => r.id);

  if (acceptedRunIds.length > 0 && rejectedRunIds.length > 0) {
    insights.push({
      pattern: `${acceptedRunIds.length}/${childIds.length} candidates accepted, ${rejectedRunIds.length} rejected`,
      confidence: 0.5,
      supporting_runs: childIds,
      correlations: [],
    });
  }

  return insights;
}

/** Pearson correlation coefficient */
function pearsonCorrelation(xs: number[], ys: number[]): number {
  const n = xs.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
  for (let i = 0; i < n; i++) {
    sumX += xs[i];
    sumY += ys[i];
    sumXY += xs[i] * ys[i];
    sumX2 += xs[i] * xs[i];
    sumY2 += ys[i] * ys[i];
  }
  const denom = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
  if (denom < 1e-12) return 0;
  return (n * sumXY - sumX * sumY) / denom;
}

/**
 * Write insights to the lessons_learned table.
 */
export async function writeInsights(
  runId: string,
  insights: Insight[],
): Promise<number> {
  if (insights.length === 0) return 0;

  const sb = getSupabase();
  const rows = insights.map(ins => ({
    run_id: runId,
    pattern: ins.pattern,
    confidence: ins.confidence,
    supporting_runs: ins.supporting_runs,
    payload: { correlations: ins.correlations },
  }));

  const { error } = await sb.from('lessons_learned').insert(rows);
  if (error) {
    log.error({ error: error.message }, 'Failed to write insights');
    return 0;
  }

  return rows.length;
}
