/**
 * Barycentric weight transfer from a source mesh (e.g. FBX) to a target mesh.
 * For each target vertex, finds the closest point on the source mesh surface,
 * computes barycentric coordinates, and interpolates per-bone skin weights.
 */
/** Flat interleaved bone weights: [boneIdx0, w0, boneIdx1, w1, ...] */
type FlatBoneWeights = number[] | null;
/** Source mesh skin data for weight transfer */
interface WeightTransferSource {
    /** Interleaved positions [x,y,z,...] */
    positions: Float32Array;
    /** Per-vertex bone weights: [boneIdx, weight, ...] flat arrays */
    boneWeights: (number[] | null)[];
    /** Triangle index buffer (if null, sequential implicit indexing is used) */
    index: Uint32Array | null;
    /** Number of vertices */
    nVerts: number;
}
/** Result of barycentric weight transfer */
interface WeightTransferResult {
    /** Per-vertex weights for target mesh: [boneIdx0, w0, boneIdx1, w1, ...] (top 4) */
    weights: FlatBoneWeights[];
    /** Per-vertex: which source triangle was matched (-1 = none) */
    triMapTri: Int32Array;
    /** Per-vertex barycentric coords (3 floats per vertex) */
    triMapBary: Float32Array;
    /** Average distance from target vert to source surface (meters) */
    avgDist: number;
    /** Max distance from target vert to source surface (meters) */
    maxDist: number;
    /** Number of fallback verts (nearest-vert instead of barycentric) */
    fallbackCount: number;
}
/**
 * Transfer bone weights from source mesh to target mesh via barycentric interpolation.
 *
 * For each target vertex, finds the closest triangle on the source mesh using a spatial
 * grid, then interpolates the source bone weights using barycentric coordinates.
 * Falls back to nearest-vertex weights for distant vertices (>50mm).
 * Returns top-4 bones per vertex, normalized.
 *
 * @param targetRestPos - Target mesh positions [x,y,z,...] (Float32Array)
 * @param source - Source mesh skin data (positions, bone weights, index, nVerts)
 * @param fallbackThreshold - Distance threshold for nearest-vertex fallback (default 0.05 = 50mm)
 */
declare function computeBarycentricWeightTransfer(targetRestPos: Float32Array, source: WeightTransferSource, fallbackThreshold?: number): WeightTransferResult | null;

export { type FlatBoneWeights, type WeightTransferResult, type WeightTransferSource, computeBarycentricWeightTransfer };
