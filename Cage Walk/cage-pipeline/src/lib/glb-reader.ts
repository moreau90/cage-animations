/**
 * Minimal GLB parser — extracts positions + indices from a .glb file.
 * No external dependencies, pure binary parsing.
 *
 * GLB format:
 *   12-byte header: magic(4) + version(4) + length(4)
 *   JSON chunk: chunkLen(4) + chunkType(4) + JSON data
 *   BIN chunk:  chunkLen(4) + chunkType(4) + binary data
 */
import fs from 'node:fs/promises';

const GLB_MAGIC = 0x46546C67;    // 'glTF'
const CHUNK_JSON = 0x4E4F534A;   // 'JSON'
const CHUNK_BIN = 0x004E4942;    // 'BIN\0'

// glTF component type → byte size
const COMP_SIZES: Record<number, number> = {
  5120: 1,  // BYTE
  5121: 1,  // UNSIGNED_BYTE
  5122: 2,  // SHORT
  5123: 2,  // UNSIGNED_SHORT
  5125: 4,  // UNSIGNED_INT
  5126: 4,  // FLOAT
};

// glTF type → component count
const TYPE_COUNTS: Record<string, number> = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
  MAT2: 4,
  MAT3: 9,
  MAT4: 16,
};

export interface GLBMesh {
  positions: Float32Array;
  indices: Uint32Array;
  nVerts: number;
  nTris: number;
}

interface GLTFAccessor {
  bufferView: number;
  byteOffset?: number;
  componentType: number;
  count: number;
  type: string;
}

interface GLTFBufferView {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
}

/**
 * Read a typed array from a glTF accessor.
 */
function readAccessor(
  accessor: GLTFAccessor,
  bufferViews: GLTFBufferView[],
  binData: Buffer,
): ArrayBuffer {
  const bv = bufferViews[accessor.bufferView];
  const compSize = COMP_SIZES[accessor.componentType];
  const compCount = TYPE_COUNTS[accessor.type];
  if (!compSize || !compCount) {
    throw new Error(`Unsupported accessor: componentType=${accessor.componentType}, type=${accessor.type}`);
  }

  const byteOffset = (bv.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const elementBytes = compSize * compCount;
  const stride = bv.byteStride ?? elementBytes;
  const totalElements = accessor.count * compCount;

  // Fast path: no stride or stride matches element size — contiguous copy
  if (stride === elementBytes) {
    const src = new Uint8Array(binData.buffer, binData.byteOffset + byteOffset, accessor.count * elementBytes);
    const out = new ArrayBuffer(src.byteLength);
    new Uint8Array(out).set(src);
    return out;
  }

  // Strided: copy element by element
  const out = new ArrayBuffer(accessor.count * elementBytes);
  const outView = new Uint8Array(out);
  const srcView = new Uint8Array(binData.buffer, binData.byteOffset);
  for (let i = 0; i < accessor.count; i++) {
    const srcOff = byteOffset + i * stride;
    const dstOff = i * elementBytes;
    for (let b = 0; b < elementBytes; b++) {
      outView[dstOff + b] = srcView[srcOff + b];
    }
  }
  return out;
}

/**
 * Parse a GLB file and extract the first mesh's positions and indices.
 *
 * @param filePath - Path to .glb file
 * @returns GLBMesh with positions (Float32Array) and indices (Uint32Array)
 */
export async function readGLB(filePath: string): Promise<GLBMesh> {
  const buf = await fs.readFile(filePath);

  // --- Header ---
  if (buf.length < 12) throw new Error('GLB too short for header');
  const magic = buf.readUInt32LE(0);
  if (magic !== GLB_MAGIC) throw new Error(`Not a GLB file (magic=0x${magic.toString(16)})`);
  const version = buf.readUInt32LE(4);
  if (version !== 2) throw new Error(`Unsupported GLB version ${version}`);

  // --- Chunks ---
  let offset = 12;
  let jsonData: string | null = null;
  let binData: Buffer | null = null;

  while (offset < buf.length) {
    if (offset + 8 > buf.length) break;
    const chunkLen = buf.readUInt32LE(offset);
    const chunkType = buf.readUInt32LE(offset + 4);
    offset += 8;

    if (chunkType === CHUNK_JSON) {
      jsonData = buf.toString('utf-8', offset, offset + chunkLen);
    } else if (chunkType === CHUNK_BIN) {
      binData = buf.subarray(offset, offset + chunkLen);
    }
    offset += chunkLen;
  }

  if (!jsonData) throw new Error('No JSON chunk in GLB');
  if (!binData) throw new Error('No BIN chunk in GLB');

  const gltf = JSON.parse(jsonData);
  const accessors: GLTFAccessor[] = gltf.accessors;
  const bufferViews: GLTFBufferView[] = gltf.bufferViews;

  // Find the first mesh primitive
  const meshes = gltf.meshes;
  if (!meshes || meshes.length === 0) throw new Error('No meshes in GLB');
  const primitive = meshes[0].primitives[0];
  if (!primitive) throw new Error('No primitives in first mesh');

  // Position accessor
  const posAccessorIdx = primitive.attributes?.POSITION;
  if (posAccessorIdx === undefined) throw new Error('No POSITION attribute');
  const posAccessor = accessors[posAccessorIdx];
  const posBuf = readAccessor(posAccessor, bufferViews, binData);
  const positions = new Float32Array(posBuf);

  // Index accessor
  const idxAccessorIdx = primitive.indices;
  if (idxAccessorIdx === undefined) throw new Error('No indices in primitive');
  const idxAccessor = accessors[idxAccessorIdx];
  const idxBuf = readAccessor(idxAccessor, bufferViews, binData);

  // Convert indices to Uint32Array (may come as Uint16)
  let indices: Uint32Array;
  if (idxAccessor.componentType === 5123) {
    // UNSIGNED_SHORT → convert to Uint32
    const u16 = new Uint16Array(idxBuf);
    indices = new Uint32Array(u16.length);
    for (let i = 0; i < u16.length; i++) indices[i] = u16[i];
  } else if (idxAccessor.componentType === 5125) {
    indices = new Uint32Array(idxBuf);
  } else {
    throw new Error(`Unsupported index componentType: ${idxAccessor.componentType}`);
  }

  const nVerts = posAccessor.count;
  const nTris = indices.length / 3;

  return { positions, indices, nVerts, nTris };
}

/**
 * Quick CLI check: `node src/lib/glb-reader.ts <path>`
 */
if (process.argv[1]?.endsWith('glb-reader.ts') && process.argv[2]) {
  readGLB(process.argv[2]).then(m => {
    console.log(`GLB: ${m.nVerts} verts, ${m.nTris} tris`);
  }).catch(console.error);
}
