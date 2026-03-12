import type { TriRef, SlicePoint } from '../types/index.js';

/**
 * Slice mesh triangles at plane X=x0, returns array of {y, z} intersection points.
 * Uses exact triangle-plane intersection — no approximation.
 */
export function sliceMeshAtX(x0: number, triList: TriRef[], mPos: Float32Array): SlicePoint[] {
  const pts: SlicePoint[] = [];
  for (let ti = 0; ti < triList.length; ti++) {
    const t = triList[ti];
    const ax = mPos[t.a3], ay = mPos[t.a3 + 1], az = mPos[t.a3 + 2];
    const bx = mPos[t.b3], by = mPos[t.b3 + 1], bz = mPos[t.b3 + 2];
    const cx = mPos[t.c3], cy = mPos[t.c3 + 1], cz = mPos[t.c3 + 2];
    const sa = ax - x0, sb = bx - x0, sc = cx - x0;
    // Skip if all on same side
    if (sa > 0 && sb > 0 && sc > 0) continue;
    if (sa < 0 && sb < 0 && sc < 0) continue;
    // Find crossing edges and interpolate intersection points
    const edges: [number, number, number, number, number, number, number, number][] = [
      [sa, sb, ax, ay, az, bx, by, bz],
      [sb, sc, bx, by, bz, cx, cy, cz],
      [sa, sc, ax, ay, az, cx, cy, cz],
    ];
    for (let ei = 0; ei < 3; ei++) {
      const e = edges[ei];
      const d0 = e[0], d1 = e[1];
      if ((d0 > 0 && d1 > 0) || (d0 < 0 && d1 < 0)) continue;
      if (d0 === 0 && d1 === 0) continue;
      const t_ = d0 / (d0 - d1);
      pts.push({
        y: e[3] + t_ * (e[6] - e[3]),
        z: e[4] + t_ * (e[7] - e[4]),
      });
    }
  }
  return pts;
}

/**
 * Pre-filter mesh triangles within a hand region bounding box.
 * Returns TriRef[] for use with sliceMeshAtX.
 */
export function filterHandTriangles(
  meshRestPos: Float32Array,
  meshIndex: Uint32Array,
  knuckleX: number,
  knuckleY: number,
  knZMin: number,
  knZMax: number,
  handSign: number,
): TriRef[] {
  const handTris: TriRef[] = [];
  const nTri = meshIndex.length / 3;
  const handSpanZ = knZMax - knZMin;
  const handZLo = knZMin - handSpanZ * 0.15;
  const handZHi = knZMax + handSpanZ * 0.15;
  for (let t = 0; t < nTri; t++) {
    const ai = meshIndex[t * 3], bi = meshIndex[t * 3 + 1], ci = meshIndex[t * 3 + 2];
    const a3 = ai * 3, b3 = bi * 3, c3 = ci * 3;
    const ax = meshRestPos[a3], bx = meshRestPos[b3], cx = meshRestPos[c3];
    const ay = meshRestPos[a3 + 1], by = meshRestPos[b3 + 1], cy = meshRestPos[c3 + 1];
    const az = meshRestPos[a3 + 2], bz = meshRestPos[b3 + 2], cz = meshRestPos[c3 + 2];
    // At least one vert past knuckleX in finger direction
    const dxa = (ax - knuckleX) * handSign;
    const dxb = (bx - knuckleX) * handSign;
    const dxc = (cx - knuckleX) * handSign;
    if (dxa < -0.005 && dxb < -0.005 && dxc < -0.005) continue;
    // At least one vert within Y range of hand
    if (Math.abs(ay - knuckleY) > 0.040 &&
        Math.abs(by - knuckleY) > 0.040 &&
        Math.abs(cy - knuckleY) > 0.040) continue;
    // At least one vert within Z range of hand
    const zMin3 = Math.min(az, bz, cz), zMax3 = Math.max(az, bz, cz);
    if (zMax3 < handZLo || zMin3 > handZHi) continue;
    handTris.push({ a3, b3, c3 });
  }
  return handTris;
}
