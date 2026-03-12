/**
 * Per-bone Dual Quaternion Skinning (DQS).
 * Blends dual quaternions with hemisphere correction, then applies to rest positions.
 */

import type { Quat, DualQuat } from '../types/index.js';
import type { BoneTransform } from './per-bone-matrices.js';
import { mat3_to_quat } from '../math/quat.js';
import { rigid_to_dq, dq_apply } from '../math/dualquat.js';

/**
 * Convert bone transforms to dual quaternions.
 *
 * @param boneTransforms - Per-bone {R, t} transforms
 * @returns Array of dual quaternions (indexed by bone index)
 */
export function boneTransformsToDualQuats(
  boneTransforms: (BoneTransform | null)[]
): (DualQuat | null)[] {
  const boneDQs: (DualQuat | null)[] = new Array(boneTransforms.length).fill(null);
  for (let bi = 0; bi < boneTransforms.length; bi++) {
    const tf = boneTransforms[bi];
    if (!tf) continue;
    const q = mat3_to_quat(tf.R) as Quat;
    boneDQs[bi] = rigid_to_dq(q, tf.t);
  }
  return boneDQs;
}

/**
 * Apply per-bone Dual Quaternion Skinning to a mesh.
 *
 * For each vertex, blends dual quaternions weighted by bone weights (with hemisphere
 * correction), normalizes, then applies to rest position. Blends with rest by alpha.
 *
 * @param restPos - Rest-pose positions [x,y,z,...] (Float32Array)
 * @param outPos - Output positions [x,y,z,...] (Float32Array, written in place)
 * @param boneWeights - Per-vertex flat weights: [boneIdx0, w0, ...] (or null)
 * @param boneDQs - Per-bone dual quaternions (from boneTransformsToDualQuats)
 * @param alpha - Blend factor (0 = rest pose, 1 = fully deformed)
 */
export function applyPerBoneDQS(
  restPos: Float32Array,
  outPos: Float32Array,
  boneWeights: (number[] | null)[],
  boneDQs: (DualQuat | null)[],
  alpha: number
): void {
  const nVerts = restPos.length / 3;

  for (let i = 0; i < nVerts; i++) {
    const rx = restPos[i * 3], ry = restPos[i * 3 + 1], rz = restPos[i * 3 + 2];
    const bw = boneWeights[i];
    let px: number, py: number, pz: number;

    if (bw) {
      let bqr0 = 0, bqr1 = 0, bqr2 = 0, bqr3 = 0;
      let bqd0 = 0, bqd1 = 0, bqd2 = 0, bqd3 = 0;
      let firstQr: Quat | null = null;

      for (let e = 0; e < bw.length; e += 2) {
        const bi = bw[e], w = bw[e + 1];
        const dq = boneDQs[bi];
        if (!dq) {
          // No transform — contribute identity DQ
          bqr3 += w;
          continue;
        }
        let qr = dq.qr, qd = dq.qd;
        // Hemisphere correction: flip to same hemisphere as first bone
        if (!firstQr) {
          firstQr = qr;
        } else if (firstQr[0] * qr[0] + firstQr[1] * qr[1] + firstQr[2] * qr[2] + firstQr[3] * qr[3] < 0) {
          qr = [-qr[0], -qr[1], -qr[2], -qr[3]];
          qd = [-qd[0], -qd[1], -qd[2], -qd[3]];
        }
        bqr0 += w * qr[0]; bqr1 += w * qr[1]; bqr2 += w * qr[2]; bqr3 += w * qr[3];
        bqd0 += w * qd[0]; bqd1 += w * qd[1]; bqd2 += w * qd[2]; bqd3 += w * qd[3];
      }

      const len = Math.sqrt(bqr0 * bqr0 + bqr1 * bqr1 + bqr2 * bqr2 + bqr3 * bqr3);
      if (len < 1e-8) {
        px = rx; py = ry; pz = rz;
      } else {
        const inv = 1.0 / len;
        const nqr: Quat = [bqr0 * inv, bqr1 * inv, bqr2 * inv, bqr3 * inv];
        const nqd: Quat = [bqd0 * inv, bqd1 * inv, bqd2 * inv, bqd3 * inv];
        const p = dq_apply(nqr, nqd, rx, ry, rz);
        px = p[0]; py = p[1]; pz = p[2];
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
