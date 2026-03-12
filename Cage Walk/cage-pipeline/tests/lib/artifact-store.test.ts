import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import fs from 'node:fs/promises';
import path from 'node:path';
import { saveFloat32, loadFloat32, saveJSON, loadJSON, deleteRunArtifacts } from '../../src/lib/artifact-store.js';

// Set env for artifact dir before import
const TEST_ARTIFACT_DIR = path.join(import.meta.dirname, '..', '.test-artifacts');
process.env.ARTIFACT_DIR = TEST_ARTIFACT_DIR;
process.env.SUPABASE_URL = 'https://test.supabase.co';
process.env.SUPABASE_SERVICE_KEY = 'test-key';
process.env.REDIS_URL = 'redis://localhost:6379';

const TEST_RUN_ID = 'test-run-artifact-001';

describe('artifact-store', () => {
  beforeAll(async () => {
    await fs.mkdir(TEST_ARTIFACT_DIR, { recursive: true });
  });

  afterAll(async () => {
    await deleteRunArtifacts(TEST_RUN_ID);
    await fs.rm(TEST_ARTIFACT_DIR, { recursive: true, force: true });
  });

  it('round-trips Float32Array', async () => {
    const original = new Float32Array([1.5, -2.3, 0, 100.123, -0.001]);
    const filePath = await saveFloat32(TEST_RUN_ID, 'test_positions', original);

    expect(filePath.endsWith('.f32')).toBe(true);

    const loaded = await loadFloat32(filePath);
    expect(loaded.length).toBe(original.length);

    for (let i = 0; i < original.length; i++) {
      expect(loaded[i]).toBeCloseTo(original[i], 5);
    }
  });

  it('round-trips large Float32Array (mesh-sized)', async () => {
    // Simulate a 5000-vertex mesh (15000 floats)
    const original = new Float32Array(15000);
    for (let i = 0; i < 15000; i++) {
      original[i] = (Math.random() - 0.5) * 2000;
    }

    const filePath = await saveFloat32(TEST_RUN_ID, 'mesh_positions', original);
    const loaded = await loadFloat32(filePath);

    expect(loaded.length).toBe(15000);
    for (let i = 0; i < 100; i++) { // spot check first 100
      expect(loaded[i]).toBeCloseTo(original[i], 5);
    }
  });

  it('round-trips JSON', async () => {
    const original = {
      hips: [0, 1.0, 0.02],
      l_knee: [-0.1, 0.5, 0.01],
      metadata: { source: 'test', count: 42 },
    };

    const filePath = await saveJSON(TEST_RUN_ID, 'test_joints', original);
    expect(filePath.endsWith('.json')).toBe(true);

    const loaded = await loadJSON<typeof original>(filePath);
    expect(loaded.hips).toEqual(original.hips);
    expect(loaded.l_knee).toEqual(original.l_knee);
    expect(loaded.metadata.count).toBe(42);
  });

  it('handles empty Float32Array', async () => {
    const original = new Float32Array(0);
    const filePath = await saveFloat32(TEST_RUN_ID, 'empty', original);
    const loaded = await loadFloat32(filePath);
    expect(loaded.length).toBe(0);
  });
});
