/**
 * Build mesh edge adjacency data from triangle indices + rest positions.
 * Ported from index.html buildMeshAdjacency().
 */
import type { MeshAdjacency } from '../qa/deformation-qa.js';

export interface FullMeshAdjacency extends MeshAdjacency {
  /** Per-vertex neighbor list (CSR format) */
  adjList: Uint32Array;
  /** Start index into adjList for vertex i */
  adjStart: Uint32Array;
  /** Number of neighbors for vertex i */
  adjCount: Uint16Array;
}

/**
 * Build unique-edge adjacency from triangle indices and rest positions.
 *
 * @param meshRestPos - Interleaved rest positions [x,y,z, ...]
 * @param indices - Triangle index buffer (every 3 = one tri)
 * @returns MeshAdjacency with edge arrays + per-vertex neighbor lists
 */
export function buildMeshAdjacency(
  meshRestPos: Float32Array,
  indices: Uint32Array,
): FullMeshAdjacency {
  const nTri = indices.length / 3;
  const nMesh = meshRestPos.length / 3;

  // Build unique edges from triangle index buffer
  const edgeSet = new Set<number>();
  const edgeAArr: number[] = [];
  const edgeBArr: number[] = [];
  const edgeRestLenArr: number[] = [];

  for (let t = 0; t < nTri; t++) {
    const a = indices[t * 3], b = indices[t * 3 + 1], c = indices[t * 3 + 2];
    const pairs: [number, number][] = [[a, b], [b, c], [a, c]];
    for (const [v0, v1] of pairs) {
      const lo = Math.min(v0, v1), hi = Math.max(v0, v1);
      const key = lo * nMesh + hi;
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
        edgeAArr.push(lo);
        edgeBArr.push(hi);
        const dx = meshRestPos[hi * 3] - meshRestPos[lo * 3];
        const dy = meshRestPos[hi * 3 + 1] - meshRestPos[lo * 3 + 1];
        const dz = meshRestPos[hi * 3 + 2] - meshRestPos[lo * 3 + 2];
        edgeRestLenArr.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
      }
    }
  }

  const nEdges = edgeAArr.length;
  const edgeA = new Uint32Array(edgeAArr);
  const edgeB = new Uint32Array(edgeBArr);
  const edgeRestLen = new Float32Array(edgeRestLenArr);

  // Build per-vertex adjacency (neighbor lists) in CSR format
  const adjCount = new Uint16Array(nMesh);
  for (let i = 0; i < nEdges; i++) {
    adjCount[edgeA[i]]++;
    adjCount[edgeB[i]]++;
  }
  const adjStart = new Uint32Array(nMesh);
  for (let i = 1; i < nMesh; i++) adjStart[i] = adjStart[i - 1] + adjCount[i - 1];
  const totalAdj = nMesh > 0 ? adjStart[nMesh - 1] + adjCount[nMesh - 1] : 0;
  const adjList = new Uint32Array(totalAdj);
  const adjFill = new Uint16Array(nMesh);
  for (let i = 0; i < nEdges; i++) {
    const a = edgeA[i], b = edgeB[i];
    adjList[adjStart[a] + adjFill[a]++] = b;
    adjList[adjStart[b] + adjFill[b]++] = a;
  }

  return {
    edgeA,
    edgeB,
    edgeRestLen,
    nEdges,
    nVerts: nMesh,
    adjList,
    adjStart,
    adjCount,
  };
}
