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

// src/geometry/toe-detection.ts
var toe_detection_exports = {};
__export(toe_detection_exports, {
  autoDetectToe: () => autoDetectToe
});
module.exports = __toCommonJS(toe_detection_exports);
function autoDetectToe(meshRestPos, anklePos, footRegionId, meshVertRegions) {
  const nMesh = meshRestPos.length / 3;
  const footVerts = [];
  for (let i = 0; i < nMesh; i++) {
    const entries = meshVertRegions[i];
    if (!entries) continue;
    let footW = 0;
    for (let k = 0; k < entries.length; k += 2) {
      if (entries[k] === footRegionId) {
        footW = entries[k + 1];
        break;
      }
    }
    if (footW < 0.5) continue;
    const vx = meshRestPos[i * 3], vy = meshRestPos[i * 3 + 1], vz = meshRestPos[i * 3 + 2];
    const dx2 = vx - anklePos[0], dy2 = vy - anklePos[1], dz2 = vz - anklePos[2];
    const dist = Math.sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);
    footVerts.push({ x: vx, y: vy, z: vz, dist });
  }
  if (footVerts.length < 10) return null;
  footVerts.sort((a, b) => b.dist - a.dist);
  const topN = Math.max(5, Math.floor(footVerts.length * 0.2));
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < topN; i++) {
    cx += footVerts[i].x;
    cy += footVerts[i].y;
    cz += footVerts[i].z;
  }
  const toePos = [cx / topN, cy / topN, cz / topN];
  const dx = toePos[0] - anklePos[0];
  const dy = toePos[1] - anklePos[1];
  const dz = toePos[2] - anklePos[2];
  const footLen = Math.sqrt(dx * dx + dy * dy + dz * dz);
  return { toePos, footLen, vertCount: topN, totalFootVerts: footVerts.length };
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  autoDetectToe
});
