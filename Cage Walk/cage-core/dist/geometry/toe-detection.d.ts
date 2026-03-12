import { Vec3 } from '../types/index.js';

/** Result of toe auto-detection for one foot */
interface ToeDetectionResult {
    toePos: Vec3;
    footLen: number;
    vertCount: number;
    totalFootVerts: number;
}
/**
 * Auto-detect toe position from mesh geometry.
 * Finds the distal end (furthest from ankle) of foot region vertices
 * and uses the centroid of the top 20% furthest as the toe position.
 *
 * @param meshRestPos - Interleaved rest positions [x,y,z, ...]
 * @param anklePos - Ankle joint position
 * @param footRegionId - Region ID for this foot (5=left, 6=right)
 * @param meshVertRegions - Per-vertex region assignments (sparse: entries[k]=regionId, entries[k+1]=weight)
 */
declare function autoDetectToe(meshRestPos: Float32Array, anklePos: Vec3, footRegionId: number, meshVertRegions: (number[] | null)[]): ToeDetectionResult | null;

export { type ToeDetectionResult, autoDetectToe };
