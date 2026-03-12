/**
 * Processing Worker
 *
 * Runs the FK → LBS deformation pipeline server-side.
 * Loads mesh from GLB, joints from placed_joints.json, keyframes from extracted FBX data.
 * Produces deformation artifacts (deformed positions, bone transforms, weights, adjacency)
 * that downstream QA workers can consume.
 */
import { Worker } from 'bullmq';
import pino from 'pino';
import {
  FBX_BONE_MAP,
  FK_CHILDREN,
  BONE_SEGMENTS,
  JOINT_PRIMARY_CHILD,
  computeJointRotationMatrices,
  computeFKChain,
  computeGroundCorrection,
  computePerBoneMatrices,
  applyPerBoneLBS,
  buildMeshAdjacency,
  getJointQuaternionAtTime,
  getRootPositionAtTime,
} from 'cage-core';
import type { Vec3, Quat, JointMap, JointPosData } from 'cage-core';
import { getRedisConnection } from '../config/redis.js';
import {
  JOB_NAMES,
  type ProcessingPayload,
  type ProcessingResult,
  type ScoreEntry,
} from '../types/job-payloads.js';
import { loadJSON, saveJSON, saveFloat32, getFileSize } from '../lib/artifact-store.js';
import { readGLB } from '../lib/glb-reader.js';
import { writeScores, writeDiagnostic } from '../lib/score-writer.js';
import { logJobStart, logJobEnd } from '../lib/job-logger.js';
import { updateRunStatus, registerArtifact } from '../lib/run-manager.js';

const log = pino({ name: 'processing' });

type PlacedJointsFile = {
  P?: { [name: string]: { P: number[] } };
  primary?: { [name: string]: number[] };
  [key: string]: unknown;
};

function extractJoints(data: PlacedJointsFile): JointMap {
  const joints: JointMap = {};
  if (data.P) {
    for (const [name, entry] of Object.entries(data.P)) {
      // Handle both formats: plain [x,y,z] array or {P: [x,y,z]} object
      if (Array.isArray(entry) && entry.length >= 3) {
        joints[name] = [entry[0], entry[1], entry[2]] as Vec3;
      } else if (entry && typeof entry === 'object' && 'P' in entry) {
        const p = (entry as { P: number[] }).P;
        if (p && p.length >= 3) {
          joints[name] = [p[0], p[1], p[2]] as Vec3;
        }
      }
    }
  }
  if (data.primary) {
    for (const [name, pos] of Object.entries(data.primary)) {
      if (pos && pos.length >= 3 && !joints[name]) {
        joints[name] = [pos[0], pos[1], pos[2]] as Vec3;
      }
    }
  }
  return joints;
}

/** Build bone index → joint name mapping from FBX_BONE_MAP */
function buildBoneNameToIdx(): { [boneIdx: number]: string } {
  const map: { [boneIdx: number]: string } = {};
  for (let i = 0; i < FBX_BONE_MAP.length; i++) {
    map[i] = FBX_BONE_MAP[i].name;
  }
  return map;
}

/**
 * Run FK+LBS for a single animation frame.
 * Returns the deformed positions and bone transforms.
 */
function processFrame(
  time: number,
  keyframes: JointPosData,
  restJoints: JointMap,
  meshRestPos: Float32Array,
  boneWeights: (number[] | null)[],
  alpha: number,
): { deformedPos: Float32Array; boneTransforms: ReturnType<typeof computePerBoneMatrices>; strain: number } {
  const boneNameToIdx = buildBoneNameToIdx();
  const boneCount = FBX_BONE_MAP.length;

  // Sample quaternions at this time
  const currentQuats: { [jn: string]: Quat } = {};
  for (const jn of Object.keys(keyframes.rest_quats)) {
    const q = getJointQuaternionAtTime(keyframes, jn, time);
    if (q) currentQuats[jn] = q;
  }

  // Compute rotation matrices (delta from rest)
  const jointRMat = computeJointRotationMatrices(keyframes.rest_quats, currentQuats);

  // Get root position
  const rootPos = getRootPositionAtTime(keyframes, time) ?? restJoints['hips'] ?? [0, 0, 0];

  // Forward kinematics
  const fkPos = computeFKChain(rootPos as Vec3, restJoints, jointRMat, FK_CHILDREN);

  // Ground correction
  const groundCorr = computeGroundCorrection(fkPos, 0);

  // Per-bone transforms
  const boneTransforms = computePerBoneMatrices(
    boneNameToIdx, jointRMat, restJoints, fkPos, groundCorr, boneCount,
  );

  // LBS deformation
  const deformedPos = new Float32Array(meshRestPos.length);
  applyPerBoneLBS(meshRestPos, deformedPos, boneWeights, boneTransforms, alpha);

  // Quick strain metric: average vertex displacement
  let totalDisp = 0;
  const nVerts = meshRestPos.length / 3;
  for (let i = 0; i < nVerts; i++) {
    const dx = deformedPos[i * 3] - meshRestPos[i * 3];
    const dy = deformedPos[i * 3 + 1] - meshRestPos[i * 3 + 1];
    const dz = deformedPos[i * 3 + 2] - meshRestPos[i * 3 + 2];
    totalDisp += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  const strain = totalDisp / nVerts;

  return { deformedPos, boneTransforms, strain };
}

const worker = new Worker<ProcessingPayload, ProcessingResult>(
  JOB_NAMES.PROCESSING,
  async (job) => {
    const { runId, glbPath, jointsPath, keyframesPath, boneWeightsPath, alpha = 1.0 } = job.data;
    const startedAt = new Date();
    let historyId: string | undefined;

    try {
      historyId = await logJobStart({ runId, jobName: JOB_NAMES.PROCESSING, jobId: job.id! });
      await updateRunStatus(runId, 'running');
      log.info({ runId, jobId: job.id }, 'Starting processing');

      // Load inputs
      const mesh = await readGLB(glbPath);
      const jointsData = await loadJSON<PlacedJointsFile>(jointsPath);
      const restJoints = extractJoints(jointsData);
      const keyframes = await loadJSON<JointPosData>(keyframesPath);

      // Load or use empty bone weights
      let boneWeights: (number[] | null)[];
      if (boneWeightsPath) {
        boneWeights = await loadJSON<(number[] | null)[]>(boneWeightsPath);
      } else {
        boneWeights = new Array(mesh.nVerts).fill(null);
      }

      log.info({
        nVerts: mesh.nVerts,
        nTris: mesh.nTris,
        duration: keyframes.duration,
        fps: keyframes.fps,
        joints: Object.keys(restJoints).length,
      }, 'Inputs loaded');

      // Sample frames across the animation
      const nSamples = Math.min(20, Math.ceil(keyframes.duration * keyframes.fps));
      const frameTimes: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        frameTimes.push((i / nSamples) * keyframes.duration);
      }

      // Process each frame, track worst strain
      let worstStrain = 0;
      let worstFrameIdx = 0;
      let worstDeformed: Float32Array | null = null;
      let worstBoneTransforms: ReturnType<typeof computePerBoneMatrices> | null = null;

      for (let fi = 0; fi < frameTimes.length; fi++) {
        const time = frameTimes[fi];
        const result = processFrame(time, keyframes, restJoints, mesh.positions, boneWeights, alpha);

        if (result.strain > worstStrain) {
          worstStrain = result.strain;
          worstFrameIdx = fi;
          worstDeformed = result.deformedPos;
          worstBoneTransforms = result.boneTransforms;
        }

        await job.updateProgress(Math.round((fi + 1) / frameTimes.length * 100));
      }

      if (!worstDeformed || !worstBoneTransforms) {
        throw new Error('No frames processed');
      }

      // Build adjacency
      const adjacency = buildMeshAdjacency(mesh.positions, mesh.indices);
      const adjacencyJSON = {
        edgeA: Array.from(adjacency.edgeA),
        edgeB: Array.from(adjacency.edgeB),
        edgeRestLen: Array.from(adjacency.edgeRestLen),
        nEdges: adjacency.nEdges,
        nVerts: adjacency.nVerts,
      };

      // Serialize bone transforms for storage
      const boneTransformsJSON = worstBoneTransforms.map(bt => {
        if (!bt) return null;
        return { R: Array.from(bt.R), t: Array.from(bt.t) };
      });

      // Save artifacts
      const deformedPath = await saveFloat32(runId, 'deformed_pos', worstDeformed);
      const restPath = await saveFloat32(runId, 'rest_pos', mesh.positions);
      const weightsPath = await saveJSON(runId, 'bone_weights', boneWeights);
      const btPath = await saveJSON(runId, 'bone_transforms', boneTransformsJSON);
      const adjPath = await saveJSON(runId, 'adjacency', adjacencyJSON);
      const jointsArtifactPath = await saveJSON(runId, 'placed_joints', jointsData);

      // Register artifacts
      for (const [name, artPath, type] of [
        ['deformed_pos', deformedPath, 'float32'],
        ['rest_pos', restPath, 'float32'],
        ['bone_weights', weightsPath, 'json'],
        ['bone_transforms', btPath, 'json'],
        ['adjacency', adjPath, 'json'],
        ['placed_joints', jointsArtifactPath, 'json'],
      ] as const) {
        const size = await getFileSize(artPath);
        await registerArtifact({ runId, name, artifactType: type, path: artPath, sizeBytes: size });
      }

      // Write processing scores
      const scores: ScoreEntry[] = [
        { run_id: runId, worker: JOB_NAMES.PROCESSING, metric: 'frames_processed', value: frameTimes.length, confidence: 1.0, unit: 'count' },
        { run_id: runId, worker: JOB_NAMES.PROCESSING, metric: 'worst_frame_strain', value: worstStrain * 1000, confidence: 1.0, unit: 'mm' },
        { run_id: runId, worker: JOB_NAMES.PROCESSING, metric: 'worst_frame_index', value: worstFrameIdx, confidence: 1.0, unit: 'index' },
      ];
      await writeScores(scores);

      await writeDiagnostic({
        runId,
        worker: JOB_NAMES.PROCESSING,
        diagnosticType: 'processing_summary',
        payload: {
          nVerts: mesh.nVerts,
          nTris: mesh.nTris,
          framesProcessed: frameTimes.length,
          worstFrameIdx,
          worstStrain: worstStrain * 1000,
          alpha,
          boneCount: FBX_BONE_MAP.length,
        },
      });

      const result: ProcessingResult = {
        runId,
        framesProcessed: frameTimes.length,
        worstFrameIdx,
        artifactPaths: {
          deformedPos: deformedPath,
          restPos: restPath,
          boneWeights: weightsPath,
          boneTransforms: btPath,
          adjacency: adjPath,
        },
      };

      log.info({
        runId,
        framesProcessed: frameTimes.length,
        worstStrain: (worstStrain * 1000).toFixed(2),
        worstFrameIdx,
      }, 'Processing complete');

      await logJobEnd(historyId, 'completed', startedAt);
      return result;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.error({ runId, err: msg }, 'Processing failed');
      if (historyId) await logJobEnd(historyId, 'failed', startedAt, msg);
      throw err;
    }
  },
  {
    connection: getRedisConnection(),
    concurrency: 1,
  },
);

worker.on('ready', () => log.info('Processing worker ready'));
worker.on('failed', (job, err) => log.error({ jobId: job?.id, err: err.message }, 'Job failed'));

const shutdown = async () => {
  log.info('Shutting down processing worker...');
  await worker.close();
  process.exit(0);
};
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
