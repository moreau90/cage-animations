-- Cage Pipeline: Initial Schema
-- Run this in Supabase SQL Editor

-- Enable UUID generation
create extension if not exists "uuid-ossp";

-- ─── runs ───
create table if not exists runs (
  id uuid primary key default uuid_generate_v4(),
  goal text not null,
  status text not null default 'pending'
    check (status in ('pending', 'running', 'completed', 'rejected', 'error')),
  decision text,
  decision_reasons text[],
  parent_run_id uuid references runs(id),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_runs_status on runs(status);
create index if not exists idx_runs_parent on runs(parent_run_id);

-- ─── run_configs ───
create table if not exists run_configs (
  id uuid primary key default uuid_generate_v4(),
  run_id uuid not null references runs(id) on delete cascade,
  config jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create index if not exists idx_run_configs_run on run_configs(run_id);

-- ─── artifacts ───
create table if not exists artifacts (
  id uuid primary key default uuid_generate_v4(),
  run_id uuid not null references runs(id) on delete cascade,
  name text not null,
  artifact_type text not null
    check (artifact_type in ('float32', 'json', 'glb', 'png')),
  path text not null,
  size_bytes bigint,
  created_at timestamptz not null default now()
);

create index if not exists idx_artifacts_run on artifacts(run_id);

-- ─── scores ───
create table if not exists scores (
  id uuid primary key default uuid_generate_v4(),
  run_id uuid not null references runs(id) on delete cascade,
  worker text not null,
  metric text not null,
  value double precision not null,
  confidence double precision not null default 1.0,
  unit text not null,
  created_at timestamptz not null default now()
);

create index if not exists idx_scores_run on scores(run_id);
create index if not exists idx_scores_metric on scores(metric);

-- ─── diagnostics ───
create table if not exists diagnostics (
  id uuid primary key default uuid_generate_v4(),
  run_id uuid not null references runs(id) on delete cascade,
  worker text not null,
  diagnostic_type text not null,
  payload jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create index if not exists idx_diagnostics_run on diagnostics(run_id);

-- ─── job_history ───
create table if not exists job_history (
  id uuid primary key default uuid_generate_v4(),
  run_id uuid not null references runs(id) on delete cascade,
  job_name text not null,
  job_id text not null,
  status text not null default 'started'
    check (status in ('started', 'completed', 'failed')),
  started_at timestamptz not null default now(),
  completed_at timestamptz,
  error_message text,
  duration_ms integer,
  created_at timestamptz not null default now()
);

create index if not exists idx_job_history_run on job_history(run_id);

-- ─── lessons_learned (Phase 2) ───
create table if not exists lessons_learned (
  id uuid primary key default uuid_generate_v4(),
  run_id uuid references runs(id) on delete set null,
  category text not null,
  insight text not null,
  created_at timestamptz not null default now()
);

create index if not exists idx_lessons_category on lessons_learned(category);

-- ─── embeddings_index (Phase 2+) ───
create table if not exists embeddings_index (
  id uuid primary key default uuid_generate_v4(),
  source_type text not null,
  source_id text not null,
  content text not null,
  embedding text,  -- will be changed to vector(1536) when pgvector is enabled in Phase 2+
  created_at timestamptz not null default now()
);

create index if not exists idx_embeddings_source on embeddings_index(source_type, source_id);
