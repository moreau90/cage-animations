-- Phase 2: Schema updates for generation/pattern-analysis pipeline
-- Run this in Supabase SQL Editor after 001_initial_schema.sql

-- ─── run_configs: add mutation tracking columns ───
alter table run_configs
  add column if not exists mutations jsonb,
  add column if not exists mutation_strategy text;

-- ─── lessons_learned: restructure for pattern analysis ───
-- Drop old columns (category/insight) and add new ones
-- Using IF EXISTS to be idempotent

-- Add new columns
alter table lessons_learned
  add column if not exists pattern text,
  add column if not exists confidence double precision default 0.5,
  add column if not exists supporting_runs uuid[],
  add column if not exists payload jsonb default '{}';

-- Make category and insight optional (they existed in Phase 1 schema)
-- We can't drop NOT NULL easily if data exists, so just set defaults
alter table lessons_learned
  alter column category set default 'pattern_analysis',
  alter column insight set default '';

-- Index for finding lessons by confidence
create index if not exists idx_lessons_confidence on lessons_learned(confidence desc);

-- Index for finding mutation runs
create index if not exists idx_run_configs_strategy on run_configs(mutation_strategy)
  where mutation_strategy is not null;

-- ─── scores: add composite index for pattern analysis queries ───
create index if not exists idx_scores_run_worker on scores(run_id, worker);

-- ─── artifacts: add support for png type (already in check constraint) ───
-- No change needed — constraint already allows 'png'
