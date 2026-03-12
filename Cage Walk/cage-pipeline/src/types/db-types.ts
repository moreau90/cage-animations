/** Supabase row types matching the schema in 001_initial_schema.sql */

export interface RunRow {
  id: string;
  goal: string;
  status: 'pending' | 'running' | 'completed' | 'rejected' | 'error';
  decision: string | null;
  decision_reasons: string[] | null;
  parent_run_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface RunConfigRow {
  id: string;
  run_id: string;
  config: Record<string, unknown>;
  created_at: string;
}

export interface ArtifactRow {
  id: string;
  run_id: string;
  name: string;
  artifact_type: 'float32' | 'json' | 'glb' | 'png';
  path: string;
  size_bytes: number | null;
  created_at: string;
}

export interface ScoreRow {
  id: string;
  run_id: string;
  worker: string;
  metric: string;
  value: number;
  confidence: number;
  unit: string;
  created_at: string;
}

export interface DiagnosticRow {
  id: string;
  run_id: string;
  worker: string;
  diagnostic_type: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface JobHistoryRow {
  id: string;
  run_id: string;
  job_name: string;
  job_id: string;
  status: 'started' | 'completed' | 'failed';
  started_at: string;
  completed_at: string | null;
  error_message: string | null;
  duration_ms: number | null;
}

export interface LessonLearnedRow {
  id: string;
  run_id: string | null;
  category: string;
  insight: string;
  created_at: string;
}

export interface EmbeddingIndexRow {
  id: string;
  source_type: string;
  source_id: string;
  content: string;
  embedding: number[];
  created_at: string;
}
