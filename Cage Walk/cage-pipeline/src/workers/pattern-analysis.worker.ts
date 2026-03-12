/**
 * Pattern Analysis Worker
 *
 * Runs after a batch of mutation candidates complete.
 * Queries Supabase for the run family, computes parameter-score
 * correlations, and writes insights to lessons_learned.
 */
import { Worker } from 'bullmq';
import pino from 'pino';
import { getRedisConnection } from '../config/redis.js';
import {
  JOB_NAMES,
  type PatternAnalysisPayload,
  type PatternAnalysisResult,
  type ScoreEntry,
} from '../types/job-payloads.js';
import { writeScores, writeDiagnostic } from '../lib/score-writer.js';
import { logJobStart, logJobEnd } from '../lib/job-logger.js';
import { updateRunStatus } from '../lib/run-manager.js';
import { analyzeRunFamily, writeInsights } from '../lib/pattern-analyzer.js';
import { getQueue } from '../queues/index.js';
import { getSupabase } from '../config/supabase.js';

const log = pino({ name: 'pattern-analysis' });

const worker = new Worker<PatternAnalysisPayload, PatternAnalysisResult>(
  JOB_NAMES.PATTERN_ANALYSIS,
  async (job) => {
    const { runId, familyRootId } = job.data;
    const startedAt = new Date();
    let historyId: string | undefined;

    try {
      historyId = await logJobStart({ runId, jobName: JOB_NAMES.PATTERN_ANALYSIS, jobId: job.id! });
      await updateRunStatus(runId, 'running');
      log.info({ runId, familyRootId }, 'Starting pattern analysis');

      // Analyze the run family (use familyRootId if provided, else runId is the parent)
      const parentId = familyRootId ?? runId;
      const insights = await analyzeRunFamily(parentId);

      log.info({ insightCount: insights.length }, 'Analysis complete');

      // Write insights to lessons_learned
      const insightsWritten = await writeInsights(runId, insights);

      // Extract top correlations for the result
      const topCorrelations = insights
        .flatMap(ins => ins.correlations)
        .slice(0, 10)
        .map(c => ({
          parameter: c.parameter,
          metric: c.metric,
          correlation: c.correlation,
        }));

      // Write scores
      const scores: ScoreEntry[] = [
        { run_id: runId, worker: JOB_NAMES.PATTERN_ANALYSIS, metric: 'insights_found', value: insights.length, confidence: 1.0, unit: 'count' },
        { run_id: runId, worker: JOB_NAMES.PATTERN_ANALYSIS, metric: 'correlations_found', value: topCorrelations.length, confidence: 1.0, unit: 'count' },
      ];
      if (topCorrelations.length > 0) {
        scores.push({
          run_id: runId, worker: JOB_NAMES.PATTERN_ANALYSIS,
          metric: 'strongest_correlation', value: Math.abs(topCorrelations[0].correlation),
          confidence: 1.0, unit: 'r',
        });
      }
      await writeScores(scores);

      await writeDiagnostic({
        runId,
        worker: JOB_NAMES.PATTERN_ANALYSIS,
        diagnosticType: 'pattern_analysis',
        payload: { insights, topCorrelations },
      });

      const result: PatternAnalysisResult = {
        runId,
        insightsWritten,
        topCorrelations,
      };

      // ─── Loop-controller trigger ───
      // Check if the parent run has loop config — if so, trigger loop controller
      try {
        const sb = getSupabase();
        const { data: cfgRow } = await sb
          .from('run_configs')
          .select('config')
          .eq('run_id', parentId)
          .single();

        const config = cfgRow?.config as Record<string, unknown> | undefined;
        const loopConfig = config?.loop as {
          searchId: string;
          round: number;
          maxRounds: number;
          staleRounds: number;
          baseStrategy: string;
          baseMutationConfig: Record<string, unknown>;
          candidateCount: number;
        } | undefined;

        if (loopConfig) {
          log.info({ parentId, round: loopConfig.round }, 'Loop config found — enqueuing loop controller');

          const lcQueue = getQueue(JOB_NAMES.LOOP_CONTROLLER);
          const dedupJobId = `loop-controller-${parentId}-round-${loopConfig.round}`;
          await lcQueue.add(
            JOB_NAMES.LOOP_CONTROLLER,
            {
              searchId: loopConfig.searchId,
              parentRunId: parentId,
              round: loopConfig.round,
              maxRounds: loopConfig.maxRounds,
              staleRounds: loopConfig.staleRounds,
              baseStrategy: loopConfig.baseStrategy,
              baseMutationConfig: loopConfig.baseMutationConfig,
              candidateCount: loopConfig.candidateCount,
            },
            { jobId: dedupJobId },
          );
        }
      } catch (loopErr) {
        const msg = loopErr instanceof Error ? loopErr.message : String(loopErr);
        log.warn({ runId, err: msg }, 'Loop-controller trigger check failed (non-fatal)');
      }

      log.info({ runId, insightsWritten, topCorrelations: topCorrelations.length }, 'Pattern analysis complete');
      await logJobEnd(historyId, 'completed', startedAt);
      return result;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.error({ runId, err: msg }, 'Pattern analysis failed');
      if (historyId) await logJobEnd(historyId, 'failed', startedAt, msg);
      throw err;
    }
  },
  {
    connection: getRedisConnection(),
    concurrency: 1,
  },
);

worker.on('ready', () => log.info('Pattern analysis worker ready'));
worker.on('failed', (job, err) => log.error({ jobId: job?.id, err: err.message }, 'Job failed'));

const shutdown = async () => {
  log.info('Shutting down pattern-analysis worker...');
  await worker.close();
  process.exit(0);
};
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
