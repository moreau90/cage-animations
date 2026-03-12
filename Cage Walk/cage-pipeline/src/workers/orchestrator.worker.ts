import { Worker } from 'bullmq';
import pino from 'pino';
import { getRedisConnection } from '../config/redis.js';
import { getQueue } from '../queues/index.js';
import { JOB_NAMES, type OrchestratorPayload, type OrchestratorResult } from '../types/job-payloads.js';
import { getScoresForRun, writeDiagnostic } from '../lib/score-writer.js';
import { updateRunStatus } from '../lib/run-manager.js';
import { logJobStart, logJobEnd } from '../lib/job-logger.js';
import { decide } from '../lib/decide.js';
import { getSupabase } from '../config/supabase.js';

const log = pino({ name: 'orchestrator' });

const worker = new Worker<OrchestratorPayload, OrchestratorResult>(
  JOB_NAMES.ORCHESTRATOR,
  async (job) => {
    const { runId } = job.data;
    const startedAt = new Date();
    let historyId: string | undefined;

    try {
      historyId = await logJobStart({ runId, jobName: JOB_NAMES.ORCHESTRATOR, jobId: job.id! });
      log.info({ runId, jobId: job.id }, 'Starting orchestrator');

      // Read all scores from both QA workers
      const scores = await getScoresForRun(runId);
      log.info({ runId, scoreCount: scores.length }, 'Loaded scores');

      // Make decision
      const { decision, reasons } = decide(scores);

      // Update run status
      const status = decision === 'reject' ? 'rejected' : 'completed';
      await updateRunStatus(runId, status, decision, reasons);

      // Write decision diagnostic
      await writeDiagnostic({
        runId,
        worker: JOB_NAMES.ORCHESTRATOR,
        diagnosticType: 'decision',
        payload: {
          decision,
          reasons,
          scoreCount: scores.length,
          scores: scores.map(s => ({ metric: s.metric, value: s.value, unit: s.unit })),
        },
      });

      const result: OrchestratorResult = { runId, decision, reasons };

      // ─── Sibling-completion detection ───
      // If this run has a parent, check if all siblings are done.
      // If so, enqueue pattern-analysis for the parent.
      try {
        const sb = getSupabase();
        const { data: thisRun } = await sb
          .from('runs')
          .select('parent_run_id')
          .eq('id', runId)
          .single();

        if (thisRun?.parent_run_id) {
          const parentId = thisRun.parent_run_id;

          // Query all siblings (children of the same parent)
          const { data: siblings } = await sb
            .from('runs')
            .select('id, status')
            .eq('parent_run_id', parentId);

          const allDone = siblings?.every(s =>
            s.status === 'completed' || s.status === 'rejected' || s.status === 'error'
          );

          if (allDone && siblings && siblings.length > 0) {
            log.info({ parentId, siblingCount: siblings.length }, 'All siblings done — enqueuing pattern analysis');

            // Use deterministic jobId to prevent duplicate enqueue from race conditions
            const dedupJobId = `pattern-analysis-${parentId}`;
            const paQueue = getQueue(JOB_NAMES.PATTERN_ANALYSIS);
            await paQueue.add(
              JOB_NAMES.PATTERN_ANALYSIS,
              { runId: parentId, familyRootId: parentId },
              { jobId: dedupJobId },
            );
          }
        }
      } catch (siblingErr) {
        // Non-fatal: log and continue
        const msg = siblingErr instanceof Error ? siblingErr.message : String(siblingErr);
        log.warn({ runId, err: msg }, 'Sibling-completion check failed (non-fatal)');
      }

      log.info({ runId, decision, reasons }, 'Orchestrator complete');
      await logJobEnd(historyId, 'completed', startedAt);
      return result;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.error({ runId, err: msg }, 'Orchestrator failed');
      if (historyId) await logJobEnd(historyId, 'failed', startedAt, msg);
      // Mark run as error
      try { await updateRunStatus(runId, 'error'); } catch { /* best effort */ }
      throw err;
    }
  },
  {
    connection: getRedisConnection(),
    concurrency: 1,
  }
);

worker.on('ready', () => log.info('Orchestrator worker ready'));
worker.on('failed', (job, err) => log.error({ jobId: job?.id, err: err.message }, 'Job failed'));

const shutdown = async () => {
  log.info('Shutting down orchestrator worker...');
  await worker.close();
  process.exit(0);
};
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

export { decide } from '../lib/decide.js';
