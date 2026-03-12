#!/usr/bin/env tsx
import fs from 'node:fs/promises';
import path from 'node:path';
import pino from 'pino';
import { getFlowProducer, getQueue, JOB_NAMES } from '../queues/index.js';
import { createRun, registerArtifact } from '../lib/run-manager.js';
import { saveJSON, saveFloat32, getFileSize } from '../lib/artifact-store.js';
import { closeQueues } from '../queues/index.js';
import { closeRedis } from '../config/redis.js';

const log = pino({ name: 'submit-run' });

interface SubmitOptions {
  goal: string;
  jointsPath: string;
  refSkeletonPath?: string;
  deformedPosPath?: string;
  restPosPath?: string;
  boneWeightsPath?: string;
  boneTransformsPath?: string;
  adjacencyPath?: string;
  alpha?: number;
  glbPath?: string;
  keyframesPath?: string;
  mutate?: string; // "strategy:param=value,param=value"
  mutateCount?: number;
  mutateSeed?: number;
}

async function copyFileArtifact(
  runId: string,
  srcPath: string,
  name: string,
  type: 'json' | 'float32'
): Promise<string> {
  const raw = await fs.readFile(srcPath);

  let destPath: string;
  if (type === 'float32') {
    const f32 = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    destPath = await saveFloat32(runId, name, f32);
  } else {
    const data = JSON.parse(raw.toString('utf-8'));
    destPath = await saveJSON(runId, name, data);
  }

  const size = await getFileSize(destPath);
  await registerArtifact({ runId, name, artifactType: type, path: destPath, sizeBytes: size });
  return destPath;
}

/** Parse mutation string: "joint_jitter:magnitude_mm=2" → { strategy, params } */
function parseMutateFlag(mutateStr: string): { strategy: string; params: Record<string, unknown> } {
  const [strategy, ...paramParts] = mutateStr.split(':');
  const params: Record<string, unknown> = {};
  for (const part of paramParts) {
    for (const kv of part.split(',')) {
      const [key, val] = kv.split('=');
      if (key && val) {
        // Try to parse as number
        const num = Number(val);
        params[key] = isNaN(num) ? val : num;
      }
    }
  }
  return { strategy, params };
}

async function submitRun(opts: SubmitOptions): Promise<string> {
  log.info({ goal: opts.goal }, 'Creating run');

  const runId = await createRun({
    goal: opts.goal,
    config: {
      alpha: opts.alpha ?? 1.0,
      ...(opts.glbPath ? { glbPath: path.resolve(opts.glbPath) } : {}),
    },
  });
  log.info({ runId }, 'Run created');

  // Copy joints artifact (required for skeleton QA)
  const jointsArtifactPath = await copyFileArtifact(runId, opts.jointsPath, 'placed_joints', 'json');

  // ─── Mutation mode: generation → processing → QA ───
  if (opts.mutate) {
    const { strategy, params } = parseMutateFlag(opts.mutate);

    // Store keyframes as artifact if provided
    if (opts.keyframesPath) {
      await copyFileArtifact(runId, opts.keyframesPath, 'keyframes', 'json');
    }
    if (opts.boneWeightsPath) {
      await copyFileArtifact(runId, opts.boneWeightsPath, 'bone_weights', 'json');
    }

    // Enqueue generation job
    const genQueue = getQueue(JOB_NAMES.GENERATION);
    await genQueue.add(JOB_NAMES.GENERATION, {
      runId,
      strategy,
      mutationConfig: {
        ...params,
        ...(opts.mutateSeed !== undefined ? { seed: opts.mutateSeed } : {}),
      },
      candidateCount: opts.mutateCount ?? 5,
    });

    log.info({ runId, strategy, count: opts.mutateCount ?? 5 }, 'Generation job enqueued');
    return runId;
  }

  // ─── Processing mode: processing → QA flow ───
  if (opts.glbPath && opts.keyframesPath) {
    const processingQueue = getQueue(JOB_NAMES.PROCESSING);

    // Copy keyframes
    const keyframesArtifactPath = await copyFileArtifact(runId, opts.keyframesPath, 'keyframes', 'json');
    let weightsArtifactPath: string | undefined;
    if (opts.boneWeightsPath) {
      weightsArtifactPath = await copyFileArtifact(runId, opts.boneWeightsPath, 'bone_weights', 'json');
    }

    await processingQueue.add(JOB_NAMES.PROCESSING, {
      runId,
      glbPath: path.resolve(opts.glbPath),
      jointsPath: jointsArtifactPath,
      keyframesPath: keyframesArtifactPath,
      boneWeightsPath: weightsArtifactPath,
      alpha: opts.alpha ?? 1.0,
    });

    log.info({ runId }, 'Processing job enqueued');
    return runId;
  }

  // ─── Direct QA mode: skeleton-qa + optional deformation-qa → orchestrator ───

  // Build skeleton QA payload
  const skelPayload: Record<string, unknown> = {
    runId,
    jointsPath: jointsArtifactPath,
  };
  if (opts.refSkeletonPath) {
    const refPath = await copyFileArtifact(runId, opts.refSkeletonPath, 'ref_skeleton', 'json');
    skelPayload.refSkeletonPath = refPath;
  }

  // Build children array — always includes skeleton QA
  const children: Array<{ name: string; queueName: string; data: Record<string, unknown> }> = [
    { name: JOB_NAMES.SKELETON_QA, queueName: JOB_NAMES.SKELETON_QA, data: skelPayload },
  ];

  // Add finger-alignment + joint-centering if GLB path available
  if (opts.glbPath) {
    const resolvedGlb = path.resolve(opts.glbPath);
    children.push({
      name: JOB_NAMES.FINGER_ALIGNMENT,
      queueName: JOB_NAMES.FINGER_ALIGNMENT,
      data: { runId, glbPath: resolvedGlb, jointsPath: jointsArtifactPath },
    });
    children.push({
      name: JOB_NAMES.JOINT_CENTERING,
      queueName: JOB_NAMES.JOINT_CENTERING,
      data: { runId, glbPath: resolvedGlb, jointsPath: jointsArtifactPath },
    });
  }

  // Add deformation QA if artifacts provided
  if (opts.deformedPosPath && opts.restPosPath && opts.boneWeightsPath && opts.boneTransformsPath && opts.adjacencyPath) {
    const deformedPath = await copyFileArtifact(runId, opts.deformedPosPath, 'deformed_pos', 'float32');
    const restPath = await copyFileArtifact(runId, opts.restPosPath, 'rest_pos', 'float32');
    const weightsPath = await copyFileArtifact(runId, opts.boneWeightsPath, 'bone_weights', 'json');
    const btPath = await copyFileArtifact(runId, opts.boneTransformsPath, 'bone_transforms', 'json');
    const adjPath = await copyFileArtifact(runId, opts.adjacencyPath, 'adjacency', 'json');

    children.push({
      name: JOB_NAMES.DEFORMATION_QA,
      queueName: JOB_NAMES.DEFORMATION_QA,
      data: {
        runId,
        deformedPosPath: deformedPath,
        restPosPath: restPath,
        boneWeightsPath: weightsPath,
        jointsPath: jointsArtifactPath,
        boneTransformsPath: btPath,
        adjacencyPath: adjPath,
        alpha: opts.alpha ?? 1.0,
      },
    });
  }

  // Submit flow: orchestrator waits for children
  const flow = getFlowProducer();
  const tree = await flow.add({
    name: JOB_NAMES.ORCHESTRATOR,
    queueName: JOB_NAMES.ORCHESTRATOR,
    data: { runId },
    children,
  });

  log.info({
    runId,
    orchestratorJobId: tree.job.id,
    childCount: children.length,
  }, 'Flow submitted');

  return runId;
}

// CLI entry point
async function main() {
  const args = process.argv.slice(2);

  if (args.length < 2) {
    console.log(`Usage: submit-run <goal> <joints-path> [options]

Options:
  --ref <path>            Reference skeleton JSON
  --deformed <path>       Deformed positions (Float32Array binary)
  --rest <path>           Rest positions (Float32Array binary)
  --weights <path>        Bone weights JSON
  --transforms <path>     Bone transforms JSON
  --adjacency <path>      Mesh adjacency JSON
  --alpha <number>        Alpha blend factor (default: 1.0)
  --glb <path>            Mesh GLB file (enables finger/centering QA)
  --keyframes <path>      Extracted keyframes JSON (enables processing)
  --mutate <strategy>     Mutation strategy (e.g., "joint_jitter:magnitude_mm=2")
  --mutate-count <n>      Number of candidates (default: 5)
  --mutate-seed <n>       PRNG seed for reproducible mutations

Examples:
  # Direct QA (skeleton only)
  submit-run "baseline check" placed_joints.json

  # Full QA with GLB (skeleton + finger + centering)
  submit-run "baseline check" placed_joints.json --glb mesh.glb

  # Processing pipeline (FK/LBS → QA)
  submit-run "test run" placed_joints.json --glb mesh.glb --keyframes kf.json

  # Mutation search
  submit-run "jitter search" placed_joints.json --glb mesh.glb --keyframes kf.json \\
    --mutate "joint_jitter:magnitude_mm=2" --mutate-count 10
`);
    process.exit(1);
  }

  const goal = args[0];
  const jointsPath = path.resolve(args[1]);

  // Parse optional flags
  const opts: SubmitOptions = { goal, jointsPath };
  for (let i = 2; i < args.length; i++) {
    switch (args[i]) {
      case '--ref': opts.refSkeletonPath = path.resolve(args[++i]); break;
      case '--deformed': opts.deformedPosPath = path.resolve(args[++i]); break;
      case '--rest': opts.restPosPath = path.resolve(args[++i]); break;
      case '--weights': opts.boneWeightsPath = path.resolve(args[++i]); break;
      case '--transforms': opts.boneTransformsPath = path.resolve(args[++i]); break;
      case '--adjacency': opts.adjacencyPath = path.resolve(args[++i]); break;
      case '--alpha': opts.alpha = parseFloat(args[++i]); break;
      case '--glb': opts.glbPath = args[++i]; break;
      case '--keyframes': opts.keyframesPath = path.resolve(args[++i]); break;
      case '--mutate': opts.mutate = args[++i]; break;
      case '--mutate-count': opts.mutateCount = parseInt(args[++i]); break;
      case '--mutate-seed': opts.mutateSeed = parseInt(args[++i]); break;
    }
  }

  try {
    const runId = await submitRun(opts);
    console.log(`Run submitted: ${runId}`);
  } catch (err) {
    console.error('Failed to submit run:', err);
    process.exit(1);
  } finally {
    await closeQueues();
    await closeRedis();
  }
}

main();
