/**
 * Adaptive Mutations
 *
 * Reads lessons_learned from previous rounds and adjusts mutation config
 * for the next round. Implements:
 * - Correlation-based parameter biasing
 * - Strategy rotation on staleness
 * - Magnitude decay over rounds
 */
import pino from 'pino';
import { getSupabase } from '../config/supabase.js';

const log = pino({ name: 'adaptive-mutations' });

/** Available mutation strategies to rotate through (directed strategies first) */
const STRATEGIES = [
  'centering_correction', 'directed_correction',
  'joint_jitter', 'finger_z_offset', 'sharpening_sweep', 'alpha_sweep', 'ground_correction',
];

export interface AdaptationResult {
  strategy: string;
  mutationConfig: Record<string, unknown>;
  candidateCount: number;
  reasoning: string[];
}

/**
 * Adapt mutation config for the next round based on lessons learned.
 *
 * @param searchId - Top-level search run ID
 * @param currentStrategy - Current mutation strategy name
 * @param currentConfig - Current mutation config
 * @param round - Current round number (1-based, about to start)
 * @param candidateCount - Current candidate count
 * @returns Adapted strategy, config, candidate count, and reasoning
 */
export async function adaptMutationConfig(
  searchId: string,
  currentStrategy: string,
  currentConfig: Record<string, unknown>,
  round: number,
  candidateCount: number,
): Promise<AdaptationResult> {
  const reasoning: string[] = [];
  const newConfig = { ...currentConfig };
  let strategy = currentStrategy;

  // Query lessons_learned for high-confidence correlations from this search
  const sb = getSupabase();
  const { data: lessons } = await sb
    .from('lessons_learned')
    .select('pattern, confidence, payload')
    .eq('run_id', searchId)
    .order('confidence', { ascending: false });

  // Apply correlation-based biasing
  if (lessons && lessons.length > 0) {
    for (const lesson of lessons) {
      if (lesson.confidence < 0.5) continue;

      const payload = lesson.payload as { correlations?: Array<{
        parameter: string;
        metric: string;
        correlation: number;
        direction: string;
      }> };

      if (!payload?.correlations) continue;

      for (const corr of payload.correlations) {
        // If "higher X → worse metric" (positive correlation with error metrics), bias X downward
        const isErrorMetric = corr.metric.includes('diff') || corr.metric.includes('strain')
          || corr.metric.includes('deviation') || corr.metric.includes('offset')
          || corr.metric.includes('mismatch');

        if (isErrorMetric && corr.direction === 'positive' && corr.parameter in newConfig) {
          const val = newConfig[corr.parameter];
          if (typeof val === 'number') {
            const factor = 1 - 0.3 * corr.correlation; // reduce by up to 30%
            newConfig[corr.parameter] = val * factor;
            reasoning.push(`Reduced ${corr.parameter} by ${((1 - factor) * 100).toFixed(0)}% (r=${corr.correlation.toFixed(2)} with ${corr.metric})`);
          }
        }

        // If "higher X → better metric" (negative correlation with error), bias X upward
        if (isErrorMetric && corr.direction === 'negative' && corr.parameter in newConfig) {
          const val = newConfig[corr.parameter];
          if (typeof val === 'number') {
            const factor = 1 + 0.2 * Math.abs(corr.correlation);
            newConfig[corr.parameter] = val * factor;
            reasoning.push(`Increased ${corr.parameter} by ${((factor - 1) * 100).toFixed(0)}% (r=${corr.correlation.toFixed(2)} with ${corr.metric})`);
          }
        }
      }
    }
  }

  // ─── Centering-aware fallback logic ───
  // If AI is unavailable, use metric-driven strategy switching
  if (strategy === 'centering_correction' || strategy === 'directed_correction') {
    // Query latest centering/finger scores to decide if we should switch strategy
    const { data: latestScores } = await sb
      .from('scores')
      .select('metric, value')
      .eq('run_id', searchId)
      .eq('worker', 'loop-controller')
      .order('created_at', { ascending: false })
      .limit(50);

    // Find the latest round's centering and finger data
    let worstCentering = 999;
    let worstFinger = 999;
    for (const s of latestScores ?? []) {
      if (s.metric.includes('centering') && s.metric.includes('total_mm')) {
        worstCentering = Math.min(worstCentering, s.value); // we want the latest worst
      }
    }
    // Also check the main worst scores from the latest round
    const { data: childScores } = await sb
      .from('scores')
      .select('metric, value')
      .in('metric', ['worst_centering_offset_mm', 'worst_centerline_deviation_mm'])
      .order('created_at', { ascending: false })
      .limit(10);

    for (const s of childScores ?? []) {
      if (s.metric === 'worst_centering_offset_mm') worstCentering = Math.min(worstCentering, s.value);
      if (s.metric === 'worst_centerline_deviation_mm') worstFinger = Math.min(worstFinger, s.value);
    }

    // Strategy progression based on convergence
    if (worstCentering < 2) {
      strategy = 'finger_z_offset';
      reasoning.push(`Centering converged (worst=${worstCentering.toFixed(1)}mm < 2mm) — switching to finger correction`);
    }
  }

  // Finger convergence → fine jitter
  if (strategy === 'finger_z_offset') {
    const { data: fingerScores } = await sb
      .from('scores')
      .select('metric, value')
      .in('metric', ['worst_centerline_deviation_mm'])
      .order('created_at', { ascending: false })
      .limit(5);
    const latestFinger = fingerScores?.[0]?.value ?? 999;
    if (latestFinger < 1) {
      strategy = 'joint_jitter';
      newConfig.magnitude_mm = 0.3;
      reasoning.push(`Fingers converged (worst=${latestFinger.toFixed(1)}mm < 1mm) — switching to fine jitter 0.3mm`);
    }
  }

  // Check for strategy staleness: rotate if same strategy for 2+ rounds with no improvement
  const { data: recentHistory } = await sb
    .from('diagnostics')
    .select('payload')
    .eq('worker', 'loop-controller')
    .eq('diagnostic_type', 'loop_decision')
    .order('created_at', { ascending: false })
    .limit(3);

  if (recentHistory && recentHistory.length >= 2) {
    const recentStrategies = recentHistory
      .map(d => (d.payload as { strategy?: string })?.strategy)
      .filter(Boolean);

    const allSame = recentStrategies.every(s => s === currentStrategy);
    if (allSame && recentStrategies.length >= 2) {
      const currentIdx = STRATEGIES.indexOf(currentStrategy);
      const nextIdx = (currentIdx + 1) % STRATEGIES.length;
      strategy = STRATEGIES[nextIdx];
      reasoning.push(`Rotated strategy from ${currentStrategy} to ${strategy} (stale for ${recentStrategies.length} rounds)`);
    }
  }

  // Reduce magnitude as rounds increase: magnitude *= 0.7^(round-1)
  if ('magnitude_mm' in newConfig && typeof newConfig.magnitude_mm === 'number') {
    const decay = Math.pow(0.7, round - 1);
    const original = newConfig.magnitude_mm;
    newConfig.magnitude_mm = Math.max(0.1, newConfig.magnitude_mm * decay);
    reasoning.push(`Magnitude decay: ${original.toFixed(2)}mm → ${(newConfig.magnitude_mm as number).toFixed(2)}mm (round ${round}, factor ${decay.toFixed(3)})`);
  }

  if (reasoning.length === 0) {
    reasoning.push('No adaptations applied (no strong correlations found)');
  }

  log.info({ round, strategy, adaptations: reasoning.length }, 'Mutation config adapted');

  return {
    strategy,
    mutationConfig: newConfig,
    candidateCount,
    reasoning,
  };
}
