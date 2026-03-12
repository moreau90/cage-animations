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
export {
  computePerBoneMatrices
};
