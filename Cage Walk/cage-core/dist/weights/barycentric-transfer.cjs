"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/weights/barycentric-transfer.ts
var barycentric_transfer_exports = {};
__export(barycentric_transfer_exports, {
  computeBarycentricWeightTransfer: () => computeBarycentricWeightTransfer
});
module.exports = __toCommonJS(barycentric_transfer_exports);
function closestPointOnTri(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz) {
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) {
    const dx2 = px - ax, dy2 = py - ay, dz2 = pz - az;
    return { bary: [1, 0, 0], dist2: dx2 * dx2 + dy2 * dy2 + dz2 * dz2 };
  }
  const bpx = px - bx, bpy = py - by, bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) {
    const dx2 = px - bx, dy2 = py - by, dz2 = pz - bz;
    return { bary: [0, 1, 0], dist2: dx2 * dx2 + dy2 * dy2 + dz2 * dz2 };
  }
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v2 = d1 / (d1 - d3);
    const qx2 = ax + v2 * abx, qy2 = ay + v2 * aby, qz2 = az + v2 * abz;
    const dx2 = px - qx2, dy2 = py - qy2, dz2 = pz - qz2;
    return { bary: [1 - v2, v2, 0], dist2: dx2 * dx2 + dy2 * dy2 + dz2 * dz2 };
  }
  const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) {
    const dx2 = px - cx, dy2 = py - cy, dz2 = pz - cz;
    return { bary: [0, 0, 1], dist2: dx2 * dx2 + dy2 * dy2 + dz2 * dz2 };
  }
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w2 = d2 / (d2 - d6);
    const qx2 = ax + w2 * acx, qy2 = ay + w2 * acy, qz2 = az + w2 * acz;
    const dx2 = px - qx2, dy2 = py - qy2, dz2 = pz - qz2;
    return { bary: [1 - w2, 0, w2], dist2: dx2 * dx2 + dy2 * dy2 + dz2 * dz2 };
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w2 = (d4 - d3) / (d4 - d3 + (d5 - d6));
    const qx2 = bx + w2 * (cx - bx), qy2 = by + w2 * (cy - by), qz2 = bz + w2 * (cz - bz);
    const dx2 = px - qx2, dy2 = py - qy2, dz2 = pz - qz2;
    return { bary: [0, 1 - w2, w2], dist2: dx2 * dx2 + dy2 * dy2 + dz2 * dz2 };
  }
  const denom = 1 / (va + vb + vc);
  const v = vb * denom, w = vc * denom, u = 1 - v - w;
  const qx = u * ax + v * bx + w * cx;
  const qy = u * ay + v * by + w * cy;
  const qz = u * az + v * bz + w * cz;
  const dx = px - qx, dy = py - qy, dz = pz - qz;
  return { bary: [u, v, w], dist2: dx * dx + dy * dy + dz * dz };
}
function computeBarycentricWeightTransfer(targetRestPos, source, fallbackThreshold = 0.05) {
  if (!source || !targetRestPos || !source.boneWeights) return null;
  const nTarget = targetRestPos.length / 3;
  const srcPos = source.positions;
  const srcBW = source.boneWeights;
  const nSrc = source.nVerts;
  let srcIdx = source.index;
  if (!srcIdx) {
    srcIdx = new Uint32Array(nSrc);
    for (let i = 0; i < nSrc; i++) srcIdx[i] = i;
  }
  const nTri = srcIdx.length / 3;
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
    if (cx < minX) minX = cx;
    if (cx > maxX) maxX = cx;
    if (cy < minY) minY = cy;
    if (cy > maxY) maxY = cy;
    if (cz < minZ) minZ = cz;
    if (cz > maxZ) maxZ = cz;
  }
  const cellSize = Math.max(5e-3, Math.cbrt(
    (maxX - minX) * (maxY - minY) * (maxZ - minZ) * 20 / Math.max(nTri, 1)
  ));
  const invCell = 1 / cellSize;
  const nX = Math.ceil((maxX - minX) * invCell) + 1;
  const nY = Math.ceil((maxY - minY) * invCell) + 1;
  const triGrid = /* @__PURE__ */ new Map();
  for (let t = 0; t < nTri; t++) {
    const gx = Math.floor((triCentroids[t * 3] - minX) * invCell);
    const gy = Math.floor((triCentroids[t * 3 + 1] - minY) * invCell);
    const gz = Math.floor((triCentroids[t * 3 + 2] - minZ) * invCell);
    const key = gx + gy * nX + gz * nX * nY;
    let cell = triGrid.get(key);
    if (!cell) {
      cell = [];
      triGrid.set(key, cell);
    }
    cell.push(t);
  }
  const result = new Array(nTarget);
  const triMapTri = new Int32Array(nTarget).fill(-1);
  const triMapBary = new Float32Array(nTarget * 3);
  let totalDist = 0, maxDist = 0, fallbackCount = 0;
  for (let i = 0; i < nTarget; i++) {
    const vx = targetRestPos[i * 3], vy = targetRestPos[i * 3 + 1], vz = targetRestPos[i * 3 + 2];
    const gx = Math.floor((vx - minX) * invCell);
    const gy = Math.floor((vy - minY) * invCell);
    const gz = Math.floor((vz - minZ) * invCell);
    let bestDist2 = Infinity, bestBary = null, bestTri = -1;
    for (let shell = 0; shell <= 5; shell++) {
      for (let dx = -shell; dx <= shell; dx++) {
        for (let dy = -shell; dy <= shell; dy++) {
          for (let dz = -shell; dz <= shell; dz++) {
            if (shell > 0 && Math.abs(dx) < shell && Math.abs(dy) < shell && Math.abs(dz) < shell) continue;
            const key = gx + dx + (gy + dy) * nX + (gz + dz) * nX * nY;
            const cell = triGrid.get(key);
            if (!cell) continue;
            for (let ci = 0; ci < cell.length; ci++) {
              const t = cell[ci];
              const i0 = srcIdx[t * 3], i1 = srcIdx[t * 3 + 1], i2 = srcIdx[t * 3 + 2];
              const res = closestPointOnTri(
                vx,
                vy,
                vz,
                srcPos[i0 * 3],
                srcPos[i0 * 3 + 1],
                srcPos[i0 * 3 + 2],
                srcPos[i1 * 3],
                srcPos[i1 * 3 + 1],
                srcPos[i1 * 3 + 2],
                srcPos[i2 * 3],
                srcPos[i2 * 3 + 1],
                srcPos[i2 * 3 + 2]
              );
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
    if (bestTri >= 0 && bestBary) {
      triMapTri[i] = bestTri;
      triMapBary[i * 3] = bestBary[0];
      triMapBary[i * 3 + 1] = bestBary[1];
      triMapBary[i * 3 + 2] = bestBary[2];
    }
    const weightMap = /* @__PURE__ */ new Map();
    if (bestTri >= 0) {
      const triVerts = [srcIdx[bestTri * 3], srcIdx[bestTri * 3 + 1], srcIdx[bestTri * 3 + 2]];
      const useFallback = dist > fallbackThreshold;
      if (useFallback) fallbackCount++;
      if (useFallback) {
        let nearestVert = triVerts[0], nearestD2 = Infinity;
        for (let vi = 0; vi < 3; vi++) {
          const fvi = triVerts[vi];
          const dx2 = vx - srcPos[fvi * 3], dy2 = vy - srcPos[fvi * 3 + 1], dz2 = vz - srcPos[fvi * 3 + 2];
          const d2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
          if (d2 < nearestD2) {
            nearestD2 = d2;
            nearestVert = fvi;
          }
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
          const b = bestBary[vi];
          for (let e = 0; e < vw.length; e += 2) {
            weightMap.set(vw[e], (weightMap.get(vw[e]) || 0) + b * vw[e + 1]);
          }
        }
      }
    }
    if (weightMap.size > 0) {
      let entries = Array.from(weightMap.entries());
      entries.sort((a, b) => b[1] - a[1]);
      if (entries.length > 4) entries = entries.slice(0, 4);
      let totalW = 0;
      for (const e of entries) totalW += e[1];
      const flat = [];
      if (totalW > 0) {
        for (const e of entries) {
          const nw = e[1] / totalW;
          if (nw >= 5e-3) flat.push(e[0], nw);
        }
        let totalW2 = 0;
        for (let e = 1; e < flat.length; e += 2) totalW2 += flat[e];
        if (totalW2 > 0 && Math.abs(totalW2 - 1) > 1e-3) {
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
    fallbackCount
  };
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  computeBarycentricWeightTransfer
});
