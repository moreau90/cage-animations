/**
 * Per-bone Linear Blend Skinning (LBS).
 * Standard weighted linear combination of per-bone rigid transforms.
 */

import type { BoneTransform } from './per-bone-matrices.js';

/**
 * Apply per-bone Linear Blend Skinning to a mesh.
 *
 * For each vertex, computes the weighted sum of bone transforms applied to the rest position,
 * then blends with rest position by alpha.
 *
 * @param restPos - Rest-pose positions [x,y,z,...] (Float32Array)
 * @param outPos - Output positions [x,y,z,...] (Float32Array, written in place)
 * @param boneWeights - Per-vertex flat weights: [boneIdx0, w0, boneIdx1, w1, ...] (or null)
 * @param boneTransforms - Per-bone transforms (indexed by bone index)
 * @param alpha - Blend factor (0 = rest pose, 1 = fully deformed)
 */
export function applyPerBoneLBS(
  restPos: Float32Array,
  outPos: Float32Array,
  boneWeights: (number[] | null)[],
  boneTransforms: (BoneTransform | null)[],
  alpha: number
): void {
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
          px += w * rx; py += w * ry; pz += w * rz;
          continue;
        }
        const R = tf.R, t = tf.t;
        px += w * (R[0] * rx + R[1] * ry + R[2] * rz + t[0]);
        py += w * (R[3] * rx + R[4] * ry + R[5] * rz + t[1]);
        pz += w * (R[6] * rx + R[7] * ry + R[8] * rz + t[2]);
      }
    } else {
      px = rx; py = ry; pz = rz;
    }

    if (!isFinite(px) || !isFinite(py) || !isFinite(pz)) {
      px = rx; py = ry; pz = rz;
    }

    outPos[i * 3] = rx + alpha * (px - rx);
    outPos[i * 3 + 1] = ry + alpha * (py - ry);
    outPos[i * 3 + 2] = rz + alpha * (pz - rz);
  }
}
