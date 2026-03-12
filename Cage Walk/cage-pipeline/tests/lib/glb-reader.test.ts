import { describe, it, expect } from 'vitest';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { readGLB } from '../../src/lib/glb-reader.js';

/**
 * Create a minimal valid GLB file in memory.
 * Contains one triangle: (0,0,0), (1,0,0), (0,1,0)
 */
function createTestGLB(): Buffer {
  // Build a minimal glTF JSON
  const gltf = {
    asset: { version: '2.0' },
    buffers: [{ byteLength: 48 }], // 3 verts * 12 bytes + 3 indices * 4 bytes = 48
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36 },     // positions: 3 verts * 3 floats * 4 bytes
      { buffer: 0, byteOffset: 36, byteLength: 12 },     // indices: 3 * 4 bytes
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', max: [1, 1, 0], min: [0, 0, 0] }, // positions
      { bufferView: 1, componentType: 5125, count: 3, type: 'SCALAR' }, // indices (UNSIGNED_INT)
    ],
    meshes: [{
      primitives: [{
        attributes: { POSITION: 0 },
        indices: 1,
      }],
    }],
  };

  const jsonStr = JSON.stringify(gltf);
  // Pad JSON to 4-byte boundary
  const jsonPadded = jsonStr + ' '.repeat((4 - (jsonStr.length % 4)) % 4);
  const jsonBuf = Buffer.from(jsonPadded, 'utf-8');

  // Binary data: 3 positions + 3 indices
  const binData = Buffer.alloc(48);
  // Position 0: (0, 0, 0)
  binData.writeFloatLE(0, 0); binData.writeFloatLE(0, 4); binData.writeFloatLE(0, 8);
  // Position 1: (1, 0, 0)
  binData.writeFloatLE(1, 12); binData.writeFloatLE(0, 16); binData.writeFloatLE(0, 20);
  // Position 2: (0, 1, 0)
  binData.writeFloatLE(0, 24); binData.writeFloatLE(1, 28); binData.writeFloatLE(0, 32);
  // Indices: 0, 1, 2
  binData.writeUInt32LE(0, 36); binData.writeUInt32LE(1, 40); binData.writeUInt32LE(2, 44);

  // GLB header
  const totalLen = 12 + 8 + jsonBuf.length + 8 + binData.length;
  const header = Buffer.alloc(12);
  header.writeUInt32LE(0x46546C67, 0); // magic 'glTF'
  header.writeUInt32LE(2, 4);          // version
  header.writeUInt32LE(totalLen, 8);    // total length

  // JSON chunk header
  const jsonChunkHeader = Buffer.alloc(8);
  jsonChunkHeader.writeUInt32LE(jsonBuf.length, 0);
  jsonChunkHeader.writeUInt32LE(0x4E4F534A, 4); // 'JSON'

  // BIN chunk header
  const binChunkHeader = Buffer.alloc(8);
  binChunkHeader.writeUInt32LE(binData.length, 0);
  binChunkHeader.writeUInt32LE(0x004E4942, 4); // 'BIN\0'

  return Buffer.concat([header, jsonChunkHeader, jsonBuf, binChunkHeader, binData]);
}

describe('readGLB', () => {
  let tmpFile: string;

  it('parses a minimal GLB and extracts positions/indices', async () => {
    tmpFile = path.join(os.tmpdir(), `test-${Date.now()}.glb`);
    await fs.writeFile(tmpFile, createTestGLB());

    const mesh = await readGLB(tmpFile);

    expect(mesh.nVerts).toBe(3);
    expect(mesh.nTris).toBe(1);
    expect(mesh.positions.length).toBe(9);  // 3 verts * 3 components
    expect(mesh.indices.length).toBe(3);

    // Check actual vertex data
    expect(mesh.positions[0]).toBeCloseTo(0);  // v0.x
    expect(mesh.positions[3]).toBeCloseTo(1);  // v1.x
    expect(mesh.positions[7]).toBeCloseTo(1);  // v2.y

    // Check indices
    expect(mesh.indices[0]).toBe(0);
    expect(mesh.indices[1]).toBe(1);
    expect(mesh.indices[2]).toBe(2);

    await fs.unlink(tmpFile);
  });

  it('rejects non-GLB files', async () => {
    tmpFile = path.join(os.tmpdir(), `test-bad-${Date.now()}.glb`);
    await fs.writeFile(tmpFile, Buffer.from('not a glb file'));
    await expect(readGLB(tmpFile)).rejects.toThrow(/Not a GLB file/);
    await fs.unlink(tmpFile);
  });

  it('rejects files too short for header', async () => {
    tmpFile = path.join(os.tmpdir(), `test-short-${Date.now()}.glb`);
    await fs.writeFile(tmpFile, Buffer.alloc(8));
    await expect(readGLB(tmpFile)).rejects.toThrow(/too short/);
    await fs.unlink(tmpFile);
  });
});
