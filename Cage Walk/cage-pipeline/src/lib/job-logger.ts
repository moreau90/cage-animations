import { getSupabase } from '../config/supabase.js';

/** Log job start to job_history */
export async function logJobStart(params: {
  runId: string;
  jobName: string;
  jobId: string;
}): Promise<string> {
  const sb = getSupabase();
  const { data, error } = await sb
    .from('job_history')
    .insert({
      run_id: params.runId,
      job_name: params.jobName,
      job_id: params.jobId,
      status: 'started',
      started_at: new Date().toISOString(),
    })
    .select('id')
    .single();
  if (error) throw new Error(`Failed to log job start: ${error.message}`);
  return data.id;
}

/** Log job completion to job_history */
export async function logJobEnd(
  historyId: string,
  status: 'completed' | 'failed',
  startedAt: Date,
  errorMessage?: string
): Promise<void> {
  const sb = getSupabase();
  const now = new Date();
  const durationMs = now.getTime() - startedAt.getTime();

  const { error } = await sb
    .from('job_history')
    .update({
      status,
      completed_at: now.toISOString(),
      duration_ms: durationMs,
      error_message: errorMessage ?? null,
    })
    .eq('id', historyId);
  if (error) throw new Error(`Failed to log job end: ${error.message}`);
}
