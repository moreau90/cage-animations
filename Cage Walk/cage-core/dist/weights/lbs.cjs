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

// src/weights/lbs.ts
var lbs_exports = {};
__export(lbs_exports, {
  applyPerBoneLBS: () => applyPerBoneLBS
});
module.exports = __toCommonJS(lbs_exports);
function applyPerBoneLBS(restPos, outPos, boneWeights, boneTransforms, alpha) {
  const nVerts = restPos.length / 3;
  for (let i = 0; i < nVerts; i++) {
    const rx = restPos[i * 3], ry = restPos[i * 3 + 1], rz = restPos[i * 3 + 2];
    const bw = boneWeights[i];
    let px = 0, py = 0, pz = 0;
    if (bw) {
      for (let e = 0; e < bw.length; e += 2) {
        const bi = bw[e], w = bw[e + 1];
        const tf = boneTransforms[bi];
        if (!tf) {
          px += w * rx;
          py += w * ry;
          pz += w * rz;
          continue;
        }
        const R = tf.R, t = tf.t;
        px += w * (R[0] * rx + R[1] * ry + R[2] * rz + t[0]);
        py += w * (R[3] * rx + R[4] * ry + R[5] * rz + t[1]);
        pz += w * (R[6] * rx + R[7] * ry + R[8] * rz + t[2]);
      }
    } else {
      px = rx;
      py = ry;
      pz = rz;
    }
    if (!isFinite(px) || !isFinite(py) || !isFinite(pz)) {
      px = rx;
      py = ry;
      pz = rz;
    }
    outPos[i * 3] = rx + alpha * (px - rx);
    outPos[i * 3 + 1] = ry + alpha * (py - ry);
    outPos[i * 3 + 2] = rz + alpha * (pz - rz);
  }
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  applyPerBoneLBS
});
