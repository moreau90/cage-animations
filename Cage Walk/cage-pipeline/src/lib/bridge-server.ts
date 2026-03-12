/**
 * Bridge Server — serves Cage Walk as static files + API endpoints
 * for runtime data export from the browser.
 *
 * Usage:
 *   npx tsx src/lib/bridge-server.ts
 *   # or via PM2 in ecosystem.config.cjs
 */
import express from 'express';
import fs from 'node:fs/promises';
import path from 'node:path';
import pino from 'pino';
import { getEnv } from '../config/env.js';
import { saveJSON, saveFloat32 } from './artifact-store.js';

const log = pino({ name: 'bridge-server' });

export interface BridgeServerOptions {
  port?: number;
  cageWalkDir?: string;
}

export interface RuntimeExport {
  runId: string;
  /** Deformed vertex positions as flat number[] */
  deformedPositions?: number[];
  /** Rest vertex positions as flat number[] */
  restPositions?: number[];
  /** Per-vertex bone weights: [boneIdx, weight, ...][] */
  boneWeights?: (number[] | null)[];
  /** Per-bone transforms: { R: number[], t: number[] }[] */
  boneTransforms?: Array<{ R: number[]; t: number[] } | null>;
  /** Adjacency data */
  adjacency?: {
    edgeA: number[];
    edgeB: number[];
    edgeRestLen: number[];
    nEdges: number;
    nVerts: number;
  };
  /** Joint positions map */
  joints?: { [name: string]: [number, number, number] };
  /** FK joint positions (current frame) */
  fkJoints?: { [name: string]: [number, number, number] };
}

export interface KeyframeExport {
  runId: string;
  /** Full JointPosData from FBX extraction */
  keyframeData: unknown;
}

/**
 * Create and configure the Express app.
 * Exported separately so tests can use it without starting a listener.
 */
export function createBridgeApp(cageWalkDir: string) {
  const app = express();

  // Parse JSON bodies up to 100MB (deformed positions can be large)
  app.use(express.json({ limit: '100mb' }));

  // Health check
  app.get('/api/health', (_req, res) => {
    res.json({ status: 'ok', cageWalkDir });
  });

  // Export runtime data from browser
  app.post('/api/export-runtime', async (req, res) => {
    try {
      const data = req.body as RuntimeExport;
      if (!data.runId) {
        res.status(400).json({ error: 'runId required' });
        return;
      }

      const saved: string[] = [];

      if (data.deformedPositions) {
        const f32 = new Float32Array(data.deformedPositions);
        const p = await saveFloat32(data.runId, 'deformed_pos', f32);
        saved.push(p);
      }

      if (data.restPositions) {
        const f32 = new Float32Array(data.restPositions);
        const p = await saveFloat32(data.runId, 'rest_pos', f32);
        saved.push(p);
      }

      if (data.boneWeights) {
        const p = await saveJSON(data.runId, 'bone_weights', data.boneWeights);
        saved.push(p);
      }

      if (data.boneTransforms) {
        const p = await saveJSON(data.runId, 'bone_transforms', data.boneTransforms);
        saved.push(p);
      }

      if (data.adjacency) {
        const p = await saveJSON(data.runId, 'adjacency', data.adjacency);
        saved.push(p);
      }

      if (data.joints) {
        const p = await saveJSON(data.runId, 'joints', data.joints);
        saved.push(p);
      }

      if (data.fkJoints) {
        const p = await saveJSON(data.runId, 'fk_joints', data.fkJoints);
        saved.push(p);
      }

      log.info({ runId: data.runId, artifactCount: saved.length }, 'Runtime data exported');
      res.json({ ok: true, saved });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.error({ err: msg }, 'Export runtime failed');
      res.status(500).json({ error: msg });
    }
  });

  // Export keyframe data from FBX extraction
  app.post('/api/export-keyframes', async (req, res) => {
    try {
      const data = req.body as KeyframeExport;
      if (!data.runId) {
        res.status(400).json({ error: 'runId required' });
        return;
      }

      const p = await saveJSON(data.runId, 'keyframes', data.keyframeData);
      log.info({ runId: data.runId }, 'Keyframes exported');
      res.json({ ok: true, saved: [p] });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.error({ err: msg }, 'Export keyframes failed');
      res.status(500).json({ error: msg });
    }
  });

  // Serve Cage Walk directory as static files (must be last — catch-all)
  app.use(express.static(cageWalkDir, {
    index: 'index.html',
    extensions: ['html', 'json', 'glb'],
  }));

  return app;
}

/**
 * Start the bridge server.
 */
export async function startBridgeServer(opts: BridgeServerOptions = {}) {
  const env = getEnv();
  const port = opts.port ?? env.BRIDGE_PORT;
  const cageWalkDir = opts.cageWalkDir ?? env.CAGE_WALK_DIR;

  if (!cageWalkDir) {
    throw new Error('CAGE_WALK_DIR environment variable or cageWalkDir option required');
  }

  // Verify directory exists
  try {
    await fs.access(cageWalkDir);
  } catch {
    throw new Error(`CAGE_WALK_DIR does not exist: ${cageWalkDir}`);
  }

  const resolvedDir = path.resolve(cageWalkDir);
  const app = createBridgeApp(resolvedDir);

  return new Promise<ReturnType<typeof app.listen>>((resolve) => {
    const server = app.listen(port, () => {
      log.info({ port, cageWalkDir: resolvedDir }, 'Bridge server started');
      resolve(server);
    });
  });
}

// Auto-start: always start the server unless BRIDGE_SERVER_NO_AUTO is set.
// PM2 wraps scripts in ProcessContainerFork.js so process.argv[1] check doesn't work.
if (!process.env.BRIDGE_SERVER_NO_AUTO) {
  startBridgeServer().catch((err) => {
    console.error('Failed to start bridge server:', err.message);
    process.exit(1);
  });
}
