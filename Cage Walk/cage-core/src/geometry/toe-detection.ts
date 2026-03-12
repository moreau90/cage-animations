import type { Vec3 } from '../types/index.js';

/** Result of toe auto-detection for one foot */
export interface ToeDetectionResult {
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
export function autoDetectToe(
  meshRestPos: Float32Array,
  anklePos: Vec3,
  footRegionId: number,
  meshVertRegions: (number[] | null)[],
): ToeDetectionResult | null {
  const nMesh = meshRestPos.length / 3;

  // Collect foot region vertices with distance from ankle
  const footVerts: { x: number; y: number; z: number; dist: number }[] = [];
  for (let i = 0; i < nMesh; i++) {
    const entries = meshVertRegions[i];
    if (!entries) continue;
    let footW = 0;
    for (let k = 0; k < entries.length; k += 2) {
      if (entries[k] === footRegionId) { footW = entries[k + 1]; break; }
    }
    if (footW < 0.5) continue;
    const vx = meshRestPos[i * 3], vy = meshRestPos[i * 3 + 1], vz = meshRestPos[i * 3 + 2];
    const dx = vx - anklePos[0], dy = vy - anklePos[1], dz = vz - anklePos[2];
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    footVerts.push({ x: vx, y: vy, z: vz, dist });
  }

  if (footVerts.length < 10) return null;

  // Sort by distance from ankle, take top 20% as "toe area"
  footVerts.sort((a, b) => b.dist - a.dist);
  const topN = Math.max(5, Math.floor(footVerts.length * 0.2));
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < topN; i++) {
    cx += footVerts[i].x;
    cy += footVerts[i].y;
    cz += footVerts[i].z;
  }
  const toePos: Vec3 = [cx / topN, cy / topN, cz / topN];
  const dx = toePos[0] - anklePos[0];
  const dy = toePos[1] - anklePos[1];
  const dz = toePos[2] - anklePos[2];
  const footLen = Math.sqrt(dx * dx + dy * dy + dz * dz);

  return { toePos, footLen, vertCount: topN, totalFootVerts: footVerts.length };
}
