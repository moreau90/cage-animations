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

// src/weights/dqs.ts
var dqs_exports = {};
__export(dqs_exports, {
  applyPerBoneDQS: () => applyPerBoneDQS,
  boneTransformsToDualQuats: () => boneTransformsToDualQuats
});
module.exports = __toCommonJS(dqs_exports);

// src/math/quat.ts
function mat3_to_quat(M) {
  const m00 = M[0], m01 = M[1], m02 = M[2];
  const m10 = M[3], m11 = M[4], m12 = M[5];
  const m20 = M[6], m21 = M[7], m22 = M[8];
  const tr = m00 + m11 + m22;
  let x, y, z, w;
  if (tr > 0) {
    const s = 0.5 / Math.sqrt(tr + 1);
    w = 0.25 / s;
    x = (m21 - m12) * s;
    y = (m02 - m20) * s;
    z = (m10 - m01) * s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    w = (m21 - m12) / s;
    x = 0.25 * s;
    y = (m01 + m10) / s;
    z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    w = (m02 - m20) / s;
    x = (m01 + m10) / s;
    y = 0.25 * s;
    z = (m12 + m21) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    w = (m10 - m01) / s;
    x = (m02 + m20) / s;
    y = (m12 + m21) / s;
    z = 0.25 * s;
  }
  const len = Math.hypot(x, y, z, w) || 1;
  return [x / len, y / len, z / len, w / len];
}

// src/math/dualquat.ts
function rigid_to_dq(q, t) {
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  const tx = t[0], ty = t[1], tz = t[2];
  return {
    qr: [qx, qy, qz, qw],
    qd: [
      0.5 * (tx * qw + ty * qz - tz * qy),
      0.5 * (-tx * qz + ty * qw + tz * qx),
      0.5 * (tx * qy - ty * qx + tz * qw),
      0.5 * (-tx * qx - ty * qy - tz * qz)
    ]
  };
}
function dq_apply(qr, qd, vx, vy, vz) {
  const ax = qr[0], ay = qr[1], az = qr[2], aw = qr[3];
  const tx2 = 2 * (ay * vz - az * vy);
  const ty2 = 2 * (az * vx - ax * vz);
  const tz2 = 2 * (ax * vy - ay * vx);
  const rx = vx + aw * tx2 + (ay * tz2 - az * ty2);
  const ry = vy + aw * ty2 + (az * tx2 - ax * tz2);
  const rz = vz + aw * tz2 + (ax * ty2 - ay * tx2);
  const ttx = 2 * (qd[0] * aw - qd[3] * ax + qd[2] * ay - qd[1] * az);
  const tty = 2 * (qd[1] * aw - qd[3] * ay + qd[0] * az - qd[2] * ax);
  const ttz = 2 * (qd[2] * aw - qd[3] * az + qd[1] * ax - qd[0] * ay);
  return [rx + ttx, ry + tty, rz + ttz];
}

// src/weights/dqs.ts
function boneTransformsToDualQuats(boneTransforms) {
  const boneDQs = new Array(boneTransforms.length).fill(null);
  for (let bi = 0; bi < boneTransforms.length; bi++) {
    const tf = boneTransforms[bi];
    if (!tf) continue;
    const q = mat3_to_quat(tf.R);
    boneDQs[bi] = rigid_to_dq(q, tf.t);
  }
  return boneDQs;
}
function applyPerBoneDQS(restPos, outPos, boneWeights, boneDQs, alpha) {
  const nVerts = restPos.length / 3;
  for (let i = 0; i < nVerts; i++) {
    const rx = restPos[i * 3], ry = restPos[i * 3 + 1], rz = restPos[i * 3 + 2];
    const bw = boneWeights[i];
    let px, py, pz;
    if (bw) {
      let bqr0 = 0, bqr1 = 0, bqr2 = 0, bqr3 = 0;
      let bqd0 = 0, bqd1 = 0, bqd2 = 0, bqd3 = 0;
      let firstQr = null;
      for (let e = 0; e < bw.length; e += 2) {
        const bi = bw[e], w = bw[e + 1];
        const dq = boneDQs[bi];
        if (!dq) {
          bqr3 += w;
          continue;
        }
        let qr = dq.qr, qd = dq.qd;
        if (!firstQr) {
          firstQr = qr;
        } else if (firstQr[0] * qr[0] + firstQr[1] * qr[1] + firstQr[2] * qr[2] + firstQr[3] * qr[3] < 0) {
          qr = [-qr[0], -qr[1], -qr[2], -qr[3]];
          qd = [-qd[0], -qd[1], -qd[2], -qd[3]];
        }
        bqr0 += w * qr[0];
        bqr1 += w * qr[1];
        bqr2 += w * qr[2];
        bqr3 += w * qr[3];
        bqd0 += w * qd[0];
        bqd1 += w * qd[1];
        bqd2 += w * qd[2];
        bqd3 += w * qd[3];
      }
      const len = Math.sqrt(bqr0 * bqr0 + bqr1 * bqr1 + bqr2 * bqr2 + bqr3 * bqr3);
      if (len < 1e-8) {
        px = rx;
        py = ry;
        pz = rz;
      } else {
        const inv = 1 / len;
        const nqr = [bqr0 * inv, bqr1 * inv, bqr2 * inv, bqr3 * inv];
        const nqd = [bqd0 * inv, bqd1 * inv, bqd2 * inv, bqd3 * inv];
        const p = dq_apply(nqr, nqd, rx, ry, rz);
        px = p[0];
        py = p[1];
        pz = p[2];
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
  applyPerBoneDQS,
  boneTransformsToDualQuats
});
