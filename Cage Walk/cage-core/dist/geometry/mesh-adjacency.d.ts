import { MeshAdjacency } from '../qa/deformation-qa.js';
import '../types/index.js';
import '../weights/per-bone-matrices.js';

/**
 * Build mesh edge adjacency data from triangle indices + rest positions.
 * Ported from index.html buildMeshAdjacency().
 */

interface FullMeshAdjacency extends MeshAdjacency {
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
declare function buildMeshAdjacency(meshRestPos: Float32Array, indices: Uint32Array): FullMeshAdjacency;

export { type FullMeshAdjacency, buildMeshAdjacency };
