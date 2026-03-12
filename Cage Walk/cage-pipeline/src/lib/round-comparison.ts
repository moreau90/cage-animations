/**
 * Round Comparison — computes before/after per-joint centering deltas
 * between two child runs within a search.
 */
import pino from 'pino';
import { getSupabase } from '../config/supabase.js';

const log = pino({ name: 'round-comparison' });

export interface JointDelta {
  before: number;
  after: number;
  delta: number;
  improved: boolean;
}

export interface RoundComparison {
  /** Per-joint centering deltas (total offset mm) */
  jointChanges: Record<string, JointDelta>;
  /** Joints that improved by > 0.1mm */
  improved: string[];
  /** Joints that worsened by > 0.1mm */
  worsened: string[];
  /** Overall composite delta (negative = better) */
  compositeDelta: number;
  /** Human-readable summary */
  summary: string;
}

/**
 * Compare centering scores between two runs (previous best vs current best).
 */
export async function compareRounds(
  previousBestId: string,
  currentBestId: string,
): Promise<RoundComparison> {
  const sb = getSupabase();

  // Query centering scores for both runs
  const [{ data: prevScores }, { data: currScores }] = await Promise.all([
    sb.from('scores').select('metric, value')
      .eq('run_id', previousBestId)
      .like('metric', 'centering_%_mm'),
    sb.from('scores').select('metric, value')
      .eq('run_id', currentBestId)
      .like('metric', 'centering_%_mm'),
  ]);

  const prevMap = new Map<string, number>();
  for (const s of prevScores ?? []) prevMap.set(s.metric, s.value);

  const currMap = new Map<string, number>();
  for (const s of currScores ?? []) currMap.set(s.metric, s.value);

  // Build per-joint comparison using total offset metrics
  const jointChanges: Record<string, JointDelta> = {};
  const improved: string[] = [];
  const worsened: string[] = [];

  // Extract joint names from metrics like centering_{joint}_total_mm
  const jointNames = new Set<string>();
  for (const metric of [...prevMap.keys(), ...currMap.keys()]) {
    const match = metric.match(/^centering_(.+)_total_mm$/);
    if (match) jointNames.add(match[1]);
  }

  // If either run has no centering data, skip comparison
  const prevHasData = [...prevMap.keys()].some(k => k.match(/^centering_.+_total_mm$/));
  const currHasData = [...currMap.keys()].some(k => k.match(/^centering_.+_total_mm$/));

  if (!prevHasData || !currHasData) {
    const reason = !prevHasData ? 'previous run has no centering scores' : 'current run has no centering scores';
    log.info({ reason }, 'Skipping round comparison — missing data');
    return {
      jointChanges: {},
      improved: [],
      worsened: [],
      compositeDelta: 0,
      summary: `Comparison skipped: ${reason}`,
    };
  }

  let totalBefore = 0;
  let totalAfter = 0;
  let jointCount = 0;

  for (const joint of jointNames) {
    const before = prevMap.get(`centering_${joint}_total_mm`) ?? 0;
    const after = currMap.get(`centering_${joint}_total_mm`) ?? 0;
    // Skip joints missing from either run
    if (!prevMap.has(`centering_${joint}_total_mm`) || !currMap.has(`centering_${joint}_total_mm`)) continue;
    const delta = after - before;

    jointChanges[joint] = { before, after, delta, improved: delta < -0.1 };

    if (delta < -0.1) improved.push(joint);
    if (delta > 0.1) worsened.push(joint);

    totalBefore += before;
    totalAfter += after;
    jointCount++;
  }

  const compositeDelta = jointCount > 0 ? (totalAfter - totalBefore) / jointCount : 0;

  // Also check worst centering offset
  const prevWorst = prevMap.get('worst_centering_offset_mm') ?? 0;
  const currWorst = currMap.get('worst_centering_offset_mm') ?? 0;

  const parts: string[] = [];
  if (improved.length > 0) parts.push(`${improved.length} joints improved (${improved.join(', ')})`);
  if (worsened.length > 0) parts.push(`${worsened.length} joints worsened (${worsened.join(', ')})`);
  parts.push(`worst: ${prevWorst.toFixed(1)}→${currWorst.toFixed(1)}mm`);
  parts.push(`avg delta: ${compositeDelta >= 0 ? '+' : ''}${compositeDelta.toFixed(2)}mm`);

  const summary = parts.join('; ');
  log.info({ improved: improved.length, worsened: worsened.length, compositeDelta: compositeDelta.toFixed(2) }, 'Round comparison');

  return { jointChanges, improved, worsened, compositeDelta, summary };
}
