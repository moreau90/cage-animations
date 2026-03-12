import { getSupabase } from '../config/supabase.js';
import type { RunRow } from '../types/db-types.js';

/** Create a new run in Supabase */
export async function createRun(params: {
  goal: string;
  parentRunId?: string;
  config?: Record<string, unknown>;
}): Promise<string> {
  const sb = getSupabase();

  const { data: run, error } = await sb
    .from('runs')
    .insert({
      goal: params.goal,
      status: 'pending',
      parent_run_id: params.parentRunId ?? null,
    })
    .select('id')
    .single();

  if (error) throw new Error(`Failed to create run: ${error.message}`);

  // Save config if provided
  if (params.config) {
    const { error: cfgError } = await sb
      .from('run_configs')
      .insert({ run_id: run.id, config: params.config });
    if (cfgError) throw new Error(`Failed to save run config: ${cfgError.message}`);
  }

  return run.id;
}

/** Update run status */
export async function updateRunStatus(
  runId: string,
  status: RunRow['status'],
  decision?: string | null,
  decisionReasons?: string[] | null
): Promise<void> {
  const sb = getSupabase();
  const update: Record<string, unknown> = {
    status,
    updated_at: new Date().toISOString(),
  };
  if (decision !== undefined) update.decision = decision;
  if (decisionReasons !== undefined) update.decision_reasons = decisionReasons;

  const { error } = await sb.from('runs').update(update).eq('id', runId);
  if (error) throw new Error(`Failed to update run ${runId}: ${error.message}`);
}

/** Register an artifact in Supabase */
export async function registerArtifact(params: {
  runId: string;
  name: string;
  artifactType: 'float32' | 'json' | 'glb' | 'png';
  path: string;
  sizeBytes?: number;
}): Promise<void> {
  const sb = getSupabase();
  const { error } = await sb.from('artifacts').insert({
    run_id: params.runId,
    name: params.name,
    artifact_type: params.artifactType,
    path: params.path,
    size_bytes: params.sizeBytes ?? null,
  });
  if (error) throw new Error(`Failed to register artifact: ${error.message}`);
}
