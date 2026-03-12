/**
 * Barycentric weight transfer from a source mesh (e.g. FBX) to a target mesh.
 * For each target vertex, finds the closest point on the source mesh surface,
 * computes barycentric coordinates, and interpolates per-bone skin weights.
 */

/** Flat interleaved bone weights: [boneIdx0, w0, boneIdx1, w1, ...] */
export type FlatBoneWeights = number[] | null;

/** Source mesh skin data for weight transfer */
export interface WeightTransferSource {
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
export interface WeightTransferResult {
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
 * Closest point on triangle using Voronoi region method.
 * Returns barycentric coords [u,v,w] and squared distance.
 */
function closestPointOnTri(
  px: number, py: number, pz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number
): { bary: [number, number, number]; dist2: number } {
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) {
    const dx = px - ax, dy = py - ay, dz = pz - az;
    return { bary: [1, 0, 0], dist2: dx * dx + dy * dy + dz * dz };
  }
  const bpx = px - bx, bpy = py - by, bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) {
    const dx = px - bx, dy = py - by, dz = pz - bz;
    return { bary: [0, 1, 0], dist2: dx * dx + dy * dy + dz * dz };
  }
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    const qx = ax + v * abx, qy = ay + v * aby, qz = az + v * abz;
    const dx = px - qx, dy = py - qy, dz = pz - qz;
    return { bary: [1 - v, v, 0], dist2: dx * dx + dy * dy + dz * dz };
  }
  const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) {
    const dx = px - cx, dy = py - cy, dz = pz - cz;
    return { bary: [0, 0, 1], dist2: dx * dx + dy * dy + dz * dz };
  }
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    const qx = ax + w * acx, qy = ay + w * acy, qz = az + w * acz;
    const dx = px - qx, dy = py - qy, dz = pz - qz;
    return { bary: [1 - w, 0, w], dist2: dx * dx + dy * dy + dz * dz };
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    const qx = bx + w * (cx - bx), qy = by + w * (cy - by), qz = bz + w * (cz - bz);
    const dx = px - qx, dy = py - qy, dz = pz - qz;
    return { bary: [0, 1 - w, w], dist2: dx * dx + dy * dy + dz * dz };
  }
  const denom = 1 / (va + vb + vc);
  const v = vb * denom, w = vc * denom, u = 1 - v - w;
  const qx = u * ax + v * bx + w * cx;
  const qy = u * ay + v * by + w * cy;
  const qz = u * az + v * bz + w * cz;
  const dx = px - qx, dy = py - qy, dz = pz - qz;
  return { bary: [u, v, w], dist2: dx * dx + dy * dy + dz * dz };
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
export function computeBarycentricWeightTransfer(
  targetRestPos: Float32Array,
  source: WeightTransferSource,
  fallbackThreshold = 0.05
): WeightTransferResult | null {
  if (!source || !targetRestPos || !source.boneWeights) return null;

  const nTarget = targetRestPos.length / 3;
  const srcPos = source.positions;
  const srcBW = source.boneWeights;
  const nSrc = source.nVerts;

  // Handle non-indexed geometry: generate implicit sequential index
  let srcIdx = source.index;
  if (!srcIdx) {
    srcIdx = new Uint32Array(nSrc);
    for (let i = 0; i < nSrc; i++) srcIdx[i] = i;
  }
  const nTri = srcIdx.length / 3;

  // Build spatial grid over source triangle centroids
  const triCentroids = new Float32Array(nTri * 3);
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let t = 0; t < nTri; t++) {
    const i0 = srcIdx[t * 3], i1 = srcIdx[t * 3 + 1], i2 = srcIdx[t * 3 + 2];
    const cx = (srcPos[i0 * 3] + srcPos[i1 * 3] + srcPos[i2 * 3]) / 3;
    const cy = (srcPos[i0 * 3 + 1] + srcPos[i1 * 3 + 1] + srcPos[i2 * 3 + 1]) / 3;
    const cz = (srcPos[i0 * 3 + 2] + srcPos[i1 * 3 + 2] + srcPos[i2 * 3 + 2]) / 3;
    triCentroids[t * 3] = cx;
    triCentroids[t * 3 + 1] = cy;
    triCentroids[t * 3 + 2] = cz;
    if (cx < minX) minX = cx; if (cx > maxX) maxX = cx;
    if (cy < minY) minY = cy; if (cy > maxY) maxY = cy;
    if (cz < minZ) minZ = cz; if (cz > maxZ) maxZ = cz;
  }

  const cellSize = Math.max(0.005, Math.cbrt(
    (maxX - minX) * (maxY - minY) * (maxZ - minZ) * 20 / Math.max(nTri, 1)
  ));
  const invCell = 1 / cellSize;
  const nX = Math.ceil((maxX - minX) * invCell) + 1;
  const nY = Math.ceil((maxY - minY) * invCell) + 1;

  const triGrid = new Map<number, number[]>();
  for (let t = 0; t < nTri; t++) {
    const gx = Math.floor((triCentroids[t * 3] - minX) * invCell);
    const gy = Math.floor((triCentroids[t * 3 + 1] - minY) * invCell);
    const gz = Math.floor((triCentroids[t * 3 + 2] - minZ) * invCell);
    const key = gx + gy * nX + gz * nX * nY;
    let cell = triGrid.get(key);
    if (!cell) { cell = []; triGrid.set(key, cell); }
    cell.push(t);
  }

  // For each target vertex, find closest source triangle and transfer weights
  const result: FlatBoneWeights[] = new Array(nTarget);
  const triMapTri = new Int32Array(nTarget).fill(-1);
  const triMapBary = new Float32Array(nTarget * 3);
  let totalDist = 0, maxDist = 0, fallbackCount = 0;

  for (let i = 0; i < nTarget; i++) {
    const vx = targetRestPos[i * 3], vy = targetRestPos[i * 3 + 1], vz = targetRestPos[i * 3 + 2];
    const gx = Math.floor((vx - minX) * invCell);
    const gy = Math.floor((vy - minY) * invCell);
    const gz = Math.floor((vz - minZ) * invCell);

    let bestDist2 = Infinity, bestBary: [number, number, number] | null = null, bestTri = -1;

    // Expanding shell search over grid cells
    for (let shell = 0; shell <= 5; shell++) {
      for (let dx = -shell; dx <= shell; dx++) {
        for (let dy = -shell; dy <= shell; dy++) {
          for (let dz = -shell; dz <= shell; dz++) {
            if (shell > 0 && Math.abs(dx) < shell && Math.abs(dy) < shell && Math.abs(dz) < shell) continue;
            const key = (gx + dx) + (gy + dy) * nX + (gz + dz) * nX * nY;
            const cell = triGrid.get(key);
            if (!cell) continue;
            for (let ci = 0; ci < cell.length; ci++) {
              const t = cell[ci];
              const i0 = srcIdx![t * 3], i1 = srcIdx![t * 3 + 1], i2 = srcIdx![t * 3 + 2];
              const res = closestPointOnTri(vx, vy, vz,
                srcPos[i0 * 3], srcPos[i0 * 3 + 1], srcPos[i0 * 3 + 2],
                srcPos[i1 * 3], srcPos[i1 * 3 + 1], srcPos[i1 * 3 + 2],
                srcPos[i2 * 3], srcPos[i2 * 3 + 1], srcPos[i2 * 3 + 2]);
              if (res.dist2 < bestDist2) {
                bestDist2 = res.dist2;
                bestBary = res.bary;
                bestTri = t;
              }
            }
          }
        }
      }
      if (bestTri >= 0 && bestDist2 < (cellSize * (shell + 1)) ** 2) break;
    }

    const dist = Math.sqrt(bestDist2);
    totalDist += dist;
    if (dist > maxDist) maxDist = dist;

    // Store triangle mapping
    if (bestTri >= 0 && bestBary) {
      triMapTri[i] = bestTri;
      triMapBary[i * 3] = bestBary[0];
      triMapBary[i * 3 + 1] = bestBary[1];
      triMapBary[i * 3 + 2] = bestBary[2];
    }

    // Accumulate bone weights using barycentric coordinates
    const weightMap = new Map<number, number>();
    if (bestTri >= 0) {
      const triVerts = [srcIdx![bestTri * 3], srcIdx![bestTri * 3 + 1], srcIdx![bestTri * 3 + 2]];

      const useFallback = dist > fallbackThreshold;
      if (useFallback) fallbackCount++;

      if (useFallback) {
        // Find nearest of the 3 triangle vertices
        let nearestVert = triVerts[0], nearestD2 = Infinity;
        for (let vi = 0; vi < 3; vi++) {
          const fvi = triVerts[vi];
          const dx2 = vx - srcPos[fvi * 3], dy2 = vy - srcPos[fvi * 3 + 1], dz2 = vz - srcPos[fvi * 3 + 2];
          const d2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
          if (d2 < nearestD2) { nearestD2 = d2; nearestVert = fvi; }
        }
        const vw = srcBW[nearestVert];
        if (vw) {
          for (let e = 0; e < vw.length; e += 2) {
            weightMap.set(vw[e], (weightMap.get(vw[e]) || 0) + vw[e + 1]);
          }
        }
      } else {
        for (let vi = 0; vi < 3; vi++) {
          const vw = srcBW[triVerts[vi]];
          if (!vw) continue;
          const b = bestBary![vi];
          for (let e = 0; e < vw.length; e += 2) {
            weightMap.set(vw[e], (weightMap.get(vw[e]) || 0) + b * vw[e + 1]);
          }
        }
      }
    }

    // Normalize, drop < 0.005, keep top-4
    if (weightMap.size > 0) {
      let entries = Array.from(weightMap.entries());
      entries.sort((a, b) => b[1] - a[1]);
      if (entries.length > 4) entries = entries.slice(0, 4);
      let totalW = 0;
      for (const e of entries) totalW += e[1];
      const flat: number[] = [];
      if (totalW > 0) {
        for (const e of entries) {
          const nw = e[1] / totalW;
          if (nw >= 0.005) flat.push(e[0], nw);
        }
        // Re-normalize after dropping small weights
        let totalW2 = 0;
        for (let e = 1; e < flat.length; e += 2) totalW2 += flat[e];
        if (totalW2 > 0 && Math.abs(totalW2 - 1) > 0.001) {
          for (let e = 1; e < flat.length; e += 2) flat[e] /= totalW2;
        }
      }
      result[i] = flat.length > 0 ? flat : null;
    } else {
      result[i] = null;
    }
  }

  return {
    weights: result,
    triMapTri,
    triMapBary,
    avgDist: totalDist / nTarget,
    maxDist,
    fallbackCount,
  };
}
