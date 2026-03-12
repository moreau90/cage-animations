import { getSupabase } from '../config/supabase.js';
import type { ScoreEntry } from '../types/job-payloads.js';

/** Write a batch of scores to Supabase */
export async function writeScores(scores: ScoreEntry[]): Promise<void> {
  if (scores.length === 0) return;
  const sb = getSupabase();
  const { error } = await sb.from('scores').insert(scores);
  if (error) throw new Error(`Failed to write scores: ${error.message}`);
}

/** Write a diagnostic payload to Supabase */
export async function writeDiagnostic(params: {
  runId: string;
  worker: string;
  diagnosticType: string;
  payload: Record<string, unknown>;
}): Promise<void> {
  const sb = getSupabase();
  const { error } = await sb.from('diagnostics').insert({
    run_id: params.runId,
    worker: params.worker,
    diagnostic_type: params.diagnosticType,
    payload: params.payload,
  });
  if (error) throw new Error(`Failed to write diagnostic: ${error.message}`);
}

/** Read all scores for a run */
export async function getScoresForRun(runId: string): Promise<ScoreEntry[]> {
  const sb = getSupabase();
  const { data, error } = await sb
    .from('scores')
    .select('run_id, worker, metric, value, confidence, unit')
    .eq('run_id', runId);
  if (error) throw new Error(`Failed to read scores: ${error.message}`);
  return data as ScoreEntry[];
}
