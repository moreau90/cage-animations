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

// src/weights/per-bone-matrices.ts
var per_bone_matrices_exports = {};
__export(per_bone_matrices_exports, {
  computePerBoneMatrices: () => computePerBoneMatrices
});
module.exports = __toCommonJS(per_bone_matrices_exports);

// src/math/mat3.ts
function mat3_mulVec3(M, x, y, z) {
  return [
    M[0] * x + M[1] * y + M[2] * z,
    M[3] * x + M[4] * y + M[5] * z,
    M[6] * x + M[7] * y + M[8] * z
  ];
}

// src/weights/per-bone-matrices.ts
function computePerBoneMatrices(boneNameToJoint, jointRotMats, restJoints, fkJoints, groundCorrection, boneCount) {
  const boneTransforms = new Array(boneCount).fill(null);
  for (let bi = 0; bi < boneCount; bi++) {
    const jn = boneNameToJoint[bi];
    if (!jn) continue;
    const R = jointRotMats[jn];
    const jRest = restJoints[jn];
    const jFK = fkJoints[jn];
    if (!R || !jRest || !jFK) continue;
    const jCurX = jFK[0];
    const jCurY = jFK[1] + groundCorrection;
    const jCurZ = jFK[2];
    const Rj = mat3_mulVec3(R, jRest[0], jRest[1], jRest[2]);
    boneTransforms[bi] = {
      R,
      t: [jCurX - Rj[0], jCurY - Rj[1], jCurZ - Rj[2]]
    };
  }
  return boneTransforms;
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  computePerBoneMatrices
});
