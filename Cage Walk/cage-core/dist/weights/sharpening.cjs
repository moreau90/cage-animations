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

// src/weights/sharpening.ts
var sharpening_exports = {};
__export(sharpening_exports, {
  SHARPEN_EXCLUSION_PAIRS: () => SHARPEN_EXCLUSION_PAIRS,
  sharpenMidSegmentWeights: () => sharpenMidSegmentWeights
});
module.exports = __toCommonJS(sharpening_exports);
var LOWER_BODY_JOINTS = /* @__PURE__ */ new Set([
  "hips",
  "l_hip",
  "r_hip",
  "l_knee",
  "r_knee",
  "l_ankle",
  "r_ankle",
  "l_toe",
  "r_toe"
]);
var SHARPEN_EXCLUSION_PAIRS = [
  ["hips", "l_hip"],
  ["hips", "r_hip"],
  ["l_hip", "l_knee"],
  ["r_hip", "r_knee"],
  ["l_knee", "l_ankle"],
  ["r_knee", "r_ankle"]
];
function sharpenMidSegmentWeights(weights, meshRestPos, joints, boneNameToIdx, jointPrimaryChild, boneSegments, strengthUpper, strengthLower) {
  if (!weights || !joints || !boneNameToIdx) return null;
  if (strengthUpper <= 0 && strengthLower <= 0) return null;
  const jointToBoneIdx = {};
  for (const [biStr, jn] of Object.entries(boneNameToIdx)) {
    jointToBoneIdx[jn] = +biStr;
  }
  const segForBone = {};
  for (const [biStr, jn] of Object.entries(boneNameToIdx)) {
    const bi = +biStr;
    const child = jointPrimaryChild[jn];
    let par, ch;
    if (child && joints[jn] && joints[child]) {
      par = jn;
      ch = child;
    } else {
      for (const [p, c] of boneSegments) {
        if (c === jn && joints[p] && joints[jn]) {
          par = p;
          ch = jn;
          break;
        }
      }
    }
    if (!par || !ch) continue;
    const pP = joints[par], cP = joints[ch];
    const dx = cP[0] - pP[0], dy = cP[1] - pP[1], dz = cP[2] - pP[2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (len < 1e-6) continue;
    segForBone[bi] = {
      pP,
      cP,
      segLen: len,
      ax: dx / len,
      ay: dy / len,
      az: dz / len,
      jn,
      isLower: LOWER_BODY_JOINTS.has(jn)
    };
  }
  const exclusionPairBIs = SHARPEN_EXCLUSION_PAIRS.map(([a, b]) => [jointToBoneIdx[a], jointToBoneIdx[b]]).filter(([a, b]) => a !== void 0 && b !== void 0);
  const nMesh = weights.length;
  let sharpenedUpper = 0, sharpenedLower = 0, excluded = 0;
  for (let i = 0; i < nMesh; i++) {
    const bw = weights[i];
    if (!bw || bw.length < 4) continue;
    let domIdx = 0, domW = bw[1];
    let secIdx = -1, secW = 0;
    for (let e = 2; e < bw.length; e += 2) {
      if (bw[e + 1] > domW) {
        secIdx = domIdx;
        secW = domW;
        domIdx = e;
        domW = bw[e + 1];
      } else if (bw[e + 1] > secW) {
        secIdx = e;
        secW = bw[e + 1];
      }
    }
    if (domW < 0.6 || domW >= 1) continue;
    const domBi = bw[domIdx];
    const seg = segForBone[domBi];
    if (!seg) continue;
    if (secIdx >= 0) {
      const secBi = bw[secIdx];
      let isExcluded = false;
      for (const [a, b] of exclusionPairBIs) {
        if (domBi === a && secBi === b || domBi === b && secBi === a) {
          isExcluded = true;
          break;
        }
      }
      if (isExcluded) {
        excluded++;
        continue;
      }
    }
    const strength = seg.isLower ? strengthLower : strengthUpper;
    if (strength <= 0) continue;
    const rx = meshRestPos[i * 3], ry = meshRestPos[i * 3 + 1], rz = meshRestPos[i * 3 + 2];
    const vdx = rx - seg.pP[0], vdy = ry - seg.pP[1], vdz = rz - seg.pP[2];
    const tRaw = (vdx * seg.ax + vdy * seg.ay + vdz * seg.az) / seg.segLen;
    const tInner0 = seg.isLower ? 0.3 : 0.2;
    const tInner1 = seg.isLower ? 0.7 : 0.8;
    const tOuter0 = tInner0 - 0.1;
    const tOuter1 = tInner1 + 0.1;
    if (tRaw < tOuter0 || tRaw > tOuter1) continue;
    let tFactor;
    if (tRaw < tInner0) tFactor = (tRaw - tOuter0) / 0.1;
    else if (tRaw > tInner1) tFactor = (tOuter1 - tRaw) / 0.1;
    else tFactor = 1;
    tFactor = tFactor * tFactor * (3 - 2 * tFactor);
    const effectiveStrength = strength * tFactor;
    if (effectiveStrength < 1e-3) continue;
    const newDomW = domW + (1 - domW) * effectiveStrength;
    const otherScale = (1 - newDomW) / (1 - domW);
    bw[domIdx + 1] = newDomW;
    for (let e = 0; e < bw.length; e += 2) {
      if (e === domIdx) continue;
      bw[e + 1] *= otherScale;
    }
    let sum = 0;
    for (let e = 0; e < bw.length; e += 2) sum += bw[e + 1];
    if (sum > 0) {
      for (let e = 0; e < bw.length; e += 2) bw[e + 1] /= sum;
    }
    if (seg.isLower) sharpenedLower++;
    else sharpenedUpper++;
  }
  return { sharpenedUpper, sharpenedLower, excluded, total: nMesh };
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  SHARPEN_EXCLUSION_PAIRS,
  sharpenMidSegmentWeights
});
