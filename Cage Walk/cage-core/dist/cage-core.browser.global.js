"use strict";
var CageCore = (() => {
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

  // src/browser.ts
  var browser_exports = {};
  __export(browser_exports, {
    BONE_SEGMENTS: () => BONE_SEGMENTS,
    CIRCUMFERENCE_SEGMENTS: () => CIRCUMFERENCE_SEGMENTS,
    FBX_BONE_MAP: () => FBX_BONE_MAP,
    FK_CHILDREN: () => FK_CHILDREN,
    JOINT_PRIMARY_CHILD: () => JOINT_PRIMARY_CHILD,
    SHARPEN_EXCLUSION_PAIRS: () => SHARPEN_EXCLUSION_PAIRS,
    applyDeltaRotation: () => applyDeltaRotation,
    applyPerBoneDQS: () => applyPerBoneDQS,
    applyPerBoneLBS: () => applyPerBoneLBS,
    autoDetectToe: () => autoDetectToe,
    boneTransformsToDualQuats: () => boneTransformsToDualQuats,
    buildMeshAdjacency: () => buildMeshAdjacency,
    canonicalBendRef: () => canonicalBendRef,
    catmullRomVec3: () => catmullRomVec3,
    checkSymmetry: () => checkSymmetry,
    clamp: () => clamp,
    compareSkeletons: () => compareSkeletons,
    computeBarycentricWeightTransfer: () => computeBarycentricWeightTransfer,
    computeCircumference: () => computeCircumference,
    computeFKChain: () => computeFKChain,
    computeGroundCorrection: () => computeGroundCorrection,
    computeIKRotMatrix: () => computeIKRotMatrix,
    computeJointRotationMatrices: () => computeJointRotationMatrices,
    computePerBoneMatrices: () => computePerBoneMatrices,
    computeRigidTransformKabsch: () => computeRigidTransformKabsch,
    computeRigidityStats: () => computeRigidityStats,
    computeStrain: () => computeStrain,
    detectHipGeometry: () => detectHipGeometry,
    dq_apply: () => dq_apply,
    evalBoundaryLine: () => evalBoundaryLine,
    extractBoneTwist: () => extractBoneTwist,
    extractPositionTwist: () => extractPositionTwist,
    extractQuatTwist: () => extractQuatTwist,
    extrapolateBoundaries: () => extrapolateBoundaries,
    filterHandTriangles: () => filterHandTriangles,
    findZRegions: () => findZRegions,
    fitLineZD: () => fitLineZD,
    getJointPositionAtTime: () => getJointPositionAtTime,
    getJointQuaternionAtTime: () => getJointQuaternionAtTime,
    getJointTwistAtTime: () => getJointTwistAtTime,
    getRootPositionAtTime: () => getRootPositionAtTime,
    mat3_create: () => mat3_create,
    mat3_det: () => mat3_det,
    mat3_identity: () => mat3_identity,
    mat3_mul: () => mat3_mul,
    mat3_mulVec3: () => mat3_mulVec3,
    mat3_orthonormalize: () => mat3_orthonormalize,
    mat3_rotAxis: () => mat3_rotAxis,
    mat3_rotX: () => mat3_rotX,
    mat3_rotY: () => mat3_rotY,
    mat3_rotZ: () => mat3_rotZ,
    mat3_to_quat: () => mat3_to_quat,
    mat3_transpose: () => mat3_transpose,
    matchFBXBone: () => matchFBXBone,
    matchFBXBoneToRegion: () => matchFBXBoneToRegion,
    matchRegionsToFingers: () => matchRegionsToFingers,
    quat_conjugate: () => quat_conjugate,
    quat_mul: () => quat_mul,
    quat_rotateVec3: () => quat_rotateVec3,
    quat_rotate_vec: () => quat_rotate_vec,
    quat_slerp: () => quat_slerp,
    quat_to_mat3: () => quat_to_mat3,
    rigid_to_dq: () => rigid_to_dq,
    rotateAroundAxis: () => rotateAroundAxis,
    sharpenMidSegmentWeights: () => sharpenMidSegmentWeights,
    shortest_arc_quat: () => shortest_arc_quat,
    sliceMeshAtPlane: () => sliceMeshAtPlane,
    sliceMeshAtX: () => sliceMeshAtX,
    smoothArray: () => smoothArray,
    svd3x3: () => svd3x3,
    swingTwistDecompose: () => swingTwistDecompose,
    symmetricEigen3x3: () => symmetricEigen3x3,
    vadd: () => vadd,
    vcross: () => vcross,
    vdot: () => vdot,
    vlen: () => vlen,
    vnorm: () => vnorm,
    vscale: () => vscale,
    vsub: () => vsub
  });

  // src/math/vec3.ts
  function clamp(x, a, b) {
    return Math.max(a, Math.min(b, x));
  }
  function vsub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  }
  function vadd(a, b) {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  }
  function vdot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }
  function vlen(a) {
    return Math.sqrt(vdot(a, a));
  }
  function vscale(a, s) {
    return [a[0] * s, a[1] * s, a[2] * s];
  }
  function vcross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]
    ];
  }
  function vnorm(a) {
    const L = vlen(a);
    return L > 1e-12 ? [a[0] / L, a[1] / L, a[2] / L] : [1, 0, 0];
  }

  // src/math/mat3.ts
  function mat3_create() {
    return new Float64Array(9);
  }
  function mat3_identity() {
    const m = mat3_create();
    m[0] = m[4] = m[8] = 1;
    return m;
  }
  function mat3_rotX(angle) {
    const c = Math.cos(angle), s = Math.sin(angle), m = mat3_create();
    m[0] = 1;
    m[4] = c;
    m[5] = -s;
    m[7] = s;
    m[8] = c;
    return m;
  }
  function mat3_rotY(angle) {
    const c = Math.cos(angle), s = Math.sin(angle), m = mat3_create();
    m[0] = c;
    m[2] = s;
    m[4] = 1;
    m[6] = -s;
    m[8] = c;
    return m;
  }
  function mat3_rotZ(angle) {
    const c = Math.cos(angle), s = Math.sin(angle), m = mat3_create();
    m[0] = c;
    m[1] = -s;
    m[3] = s;
    m[4] = c;
    m[8] = 1;
    return m;
  }
  function mat3_rotAxis(ax, ay, az, angle) {
    const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
    const len = Math.hypot(ax, ay, az) || 1;
    ax /= len;
    ay /= len;
    az /= len;
    return [
      t * ax * ax + c,
      t * ax * ay - s * az,
      t * ax * az + s * ay,
      t * ax * ay + s * az,
      t * ay * ay + c,
      t * ay * az - s * ax,
      t * ax * az - s * ay,
      t * ay * az + s * ax,
      t * az * az + c
    ];
  }
  function mat3_mul(A, B) {
    const C = mat3_create();
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
    return C;
  }
  function mat3_transpose(A) {
    const T = mat3_create();
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        T[i * 3 + j] = A[j * 3 + i];
    return T;
  }
  function mat3_det(A) {
    return A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
  }
  function mat3_mulVec3(M, x, y, z) {
    return [
      M[0] * x + M[1] * y + M[2] * z,
      M[3] * x + M[4] * y + M[5] * z,
      M[6] * x + M[7] * y + M[8] * z
    ];
  }
  function mat3_orthonormalize(M) {
    let c0 = [M[0], M[3], M[6]];
    let c1 = [M[1], M[4], M[7]];
    let len = Math.hypot(c0[0], c0[1], c0[2]);
    if (len < 1e-12) {
      c0 = [1, 0, 0];
      len = 1;
    }
    c0 = [c0[0] / len, c0[1] / len, c0[2] / len];
    let dot = c1[0] * c0[0] + c1[1] * c0[1] + c1[2] * c0[2];
    c1 = [c1[0] - dot * c0[0], c1[1] - dot * c0[1], c1[2] - dot * c0[2]];
    len = Math.hypot(c1[0], c1[1], c1[2]);
    if (len < 1e-12) {
      c1 = Math.abs(c0[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
      dot = c1[0] * c0[0] + c1[1] * c0[1] + c1[2] * c0[2];
      c1 = [c1[0] - dot * c0[0], c1[1] - dot * c0[1], c1[2] - dot * c0[2]];
      len = Math.hypot(c1[0], c1[1], c1[2]);
    }
    c1 = [c1[0] / len, c1[1] / len, c1[2] / len];
    const c2 = [
      c0[1] * c1[2] - c0[2] * c1[1],
      c0[2] * c1[0] - c0[0] * c1[2],
      c0[0] * c1[1] - c0[1] * c1[0]
    ];
    const R = new Float64Array(9);
    R[0] = c0[0];
    R[1] = c1[0];
    R[2] = c2[0];
    R[3] = c0[1];
    R[4] = c1[1];
    R[5] = c2[1];
    R[6] = c0[2];
    R[7] = c1[2];
    R[8] = c2[2];
    return R;
  }

  // src/math/quat.ts
  function quat_mul(a, b) {
    return [
      a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
      a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
      a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
      a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
    ];
  }
  function quat_conjugate(q) {
    return [-q[0], -q[1], -q[2], q[3]];
  }
  function quat_slerp(a, b, t) {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    const b2 = dot < 0 ? [-b[0], -b[1], -b[2], -b[3]] : b;
    dot = Math.abs(dot);
    if (dot > 0.9995) {
      const r = [
        a[0] + (b2[0] - a[0]) * t,
        a[1] + (b2[1] - a[1]) * t,
        a[2] + (b2[2] - a[2]) * t,
        a[3] + (b2[3] - a[3]) * t
      ];
      const inv = 1 / Math.hypot(r[0], r[1], r[2], r[3]);
      return [r[0] * inv, r[1] * inv, r[2] * inv, r[3] * inv];
    }
    const theta = Math.acos(dot);
    const sinTheta = Math.sin(theta);
    const wa = Math.sin((1 - t) * theta) / sinTheta;
    const wb = Math.sin(t * theta) / sinTheta;
    return [
      wa * a[0] + wb * b2[0],
      wa * a[1] + wb * b2[1],
      wa * a[2] + wb * b2[2],
      wa * a[3] + wb * b2[3]
    ];
  }
  function swingTwistDecompose(qDelta, twistAxis) {
    const proj = qDelta[0] * twistAxis[0] + qDelta[1] * twistAxis[1] + qDelta[2] * twistAxis[2];
    let tx = proj * twistAxis[0], ty = proj * twistAxis[1], tz = proj * twistAxis[2];
    const tw = qDelta[3];
    const len = Math.hypot(tx, ty, tz, tw);
    if (len < 1e-10) return 0;
    tx /= len;
    ty /= len;
    tz /= len;
    const twn = tw / len;
    const sinHalf = Math.hypot(tx, ty, tz);
    let angle = 2 * Math.atan2(sinHalf, Math.abs(twn));
    if (proj < 0) angle = -angle;
    if (twn < 0) angle = angle > 0 ? angle - 2 * Math.PI : angle + 2 * Math.PI;
    return angle;
  }
  function quat_rotate_vec(q, v) {
    const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    const vx = v[0], vy = v[1], vz = v[2];
    const cx = qy * vz - qz * vy, cy = qz * vx - qx * vz, cz = qx * vy - qy * vx;
    const c2x = qy * cz - qz * cy, c2y = qz * cx - qx * cz, c2z = qx * cy - qy * cx;
    return [vx + 2 * (qw * cx + c2x), vy + 2 * (qw * cy + c2y), vz + 2 * (qw * cz + c2z)];
  }
  function shortest_arc_quat(a, b) {
    const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    if (d > 0.999999) return [0, 0, 0, 1];
    if (d < -0.999999) {
      let perp = vcross(a, [1, 0, 0]);
      if (vlen(perp) < 1e-6) perp = vcross(a, [0, 1, 0]);
      perp = vnorm(perp);
      return [perp[0], perp[1], perp[2], 0];
    }
    const c = vcross(a, b);
    const w = 1 + d;
    const inv = 1 / Math.hypot(c[0], c[1], c[2], w);
    return [c[0] * inv, c[1] * inv, c[2] * inv, w * inv];
  }
  function extractQuatTwist(restQ, curQ, boneDir) {
    let qDelta = quat_mul(curQ, quat_conjugate(restQ));
    if (qDelta[3] < 0) qDelta = [-qDelta[0], -qDelta[1], -qDelta[2], -qDelta[3]];
    const d_cur = quat_rotate_vec(qDelta, boneDir);
    const q_swing = shortest_arc_quat(boneDir, d_cur);
    const q_residual = quat_mul(quat_conjugate(q_swing), qDelta);
    return swingTwistDecompose(q_residual, boneDir);
  }
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
  function quat_to_mat3(q) {
    const x = q[0], y = q[1], z = q[2], w = q[3];
    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, xy = x * y2, xz = x * z2;
    const yy = y * y2, yz = y * z2, zz = z * z2;
    const wx = w * x2, wy = w * y2, wz = w * z2;
    return new Float64Array([
      1 - (yy + zz),
      xy - wz,
      xz + wy,
      xy + wz,
      1 - (xx + zz),
      yz - wx,
      xz - wy,
      yz + wx,
      1 - (xx + yy)
    ]);
  }

  // src/math/svd.ts
  function symmetricEigen3x3(S) {
    const A = Float64Array.from(S);
    const V = mat3_identity();
    for (let iter = 0; iter < 30; iter++) {
      let maxVal = 0, p = 0, q = 1;
      for (let i = 0; i < 3; i++)
        for (let j = i + 1; j < 3; j++)
          if (Math.abs(A[i * 3 + j]) > maxVal) {
            maxVal = Math.abs(A[i * 3 + j]);
            p = i;
            q = j;
          }
      if (maxVal < 1e-12) break;
      const app = A[p * 3 + p], aqq = A[q * 3 + q], apq = A[p * 3 + q];
      const theta = Math.abs(app - aqq) < 1e-14 ? Math.PI / 4 : 0.5 * Math.atan2(2 * apq, app - aqq);
      const c = Math.cos(theta), s = Math.sin(theta);
      const B = Float64Array.from(A);
      for (let i = 0; i < 3; i++) {
        B[i * 3 + p] = c * A[i * 3 + p] + s * A[i * 3 + q];
        B[i * 3 + q] = -s * A[i * 3 + p] + c * A[i * 3 + q];
      }
      for (let j = 0; j < 3; j++) {
        A[p * 3 + j] = c * B[p * 3 + j] + s * B[q * 3 + j];
        A[q * 3 + j] = -s * B[p * 3 + j] + c * B[q * 3 + j];
      }
      for (let i = 0; i < 3; i++) {
        const vip = V[i * 3 + p], viq = V[i * 3 + q];
        V[i * 3 + p] = c * vip + s * viq;
        V[i * 3 + q] = -s * vip + c * viq;
      }
    }
    return { eigenvalues: [A[0], A[4], A[8]], V };
  }
  function svd3x3(H) {
    const Ht = mat3_transpose(H);
    const HtH = mat3_mul(Ht, H);
    const { eigenvalues, V } = symmetricEigen3x3(HtH);
    const sigma = [
      Math.sqrt(Math.max(0, eigenvalues[0])),
      Math.sqrt(Math.max(0, eigenvalues[1])),
      Math.sqrt(Math.max(0, eigenvalues[2]))
    ];
    const HV = mat3_mul(H, V);
    const U = mat3_create();
    for (let j = 0; j < 3; j++) {
      const invS = sigma[j] > 1e-8 ? 1 / sigma[j] : 0;
      for (let i = 0; i < 3; i++) {
        U[i * 3 + j] = HV[i * 3 + j] * invS;
      }
    }
    return { U, S: sigma, V };
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
  function quat_rotateVec3(q, vx, vy, vz) {
    const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    const tx = 2 * (qy * vz - qz * vy);
    const ty = 2 * (qz * vx - qx * vz);
    const tz = 2 * (qx * vy - qy * vx);
    return [
      vx + qw * tx + (qy * tz - qz * ty),
      vy + qw * ty + (qz * tx - qx * tz),
      vz + qw * tz + (qx * ty - qy * tx)
    ];
  }

  // src/math/kabsch.ts
  function computeRigidTransformKabsch(indices, restPositions, currentPositions) {
    const n = indices.length;
    let c0x = 0, c0y = 0, c0z = 0, ctx = 0, cty = 0, ctz = 0;
    for (const i of indices) {
      c0x += restPositions[i][0];
      c0y += restPositions[i][1];
      c0z += restPositions[i][2];
      ctx += currentPositions[i * 3];
      cty += currentPositions[i * 3 + 1];
      ctz += currentPositions[i * 3 + 2];
    }
    c0x /= n;
    c0y /= n;
    c0z /= n;
    ctx /= n;
    cty /= n;
    ctz /= n;
    const H = mat3_create();
    for (const i of indices) {
      const dx0 = restPositions[i][0] - c0x, dy0 = restPositions[i][1] - c0y, dz0 = restPositions[i][2] - c0z;
      const dxt = currentPositions[i * 3] - ctx, dyt = currentPositions[i * 3 + 1] - cty, dzt = currentPositions[i * 3 + 2] - ctz;
      H[0] += dx0 * dxt;
      H[1] += dx0 * dyt;
      H[2] += dx0 * dzt;
      H[3] += dy0 * dxt;
      H[4] += dy0 * dyt;
      H[5] += dy0 * dzt;
      H[6] += dz0 * dxt;
      H[7] += dz0 * dyt;
      H[8] += dz0 * dzt;
    }
    const { U, S, V } = svd3x3(H);
    const sMax = Math.max(S[0], S[1], S[2]);
    if (sMax > 1e-10) {
      let minI = 0;
      if (S[1] < S[minI]) minI = 1;
      if (S[2] < S[minI]) minI = 2;
      if (S[minI] < sMax * 1e-3) {
        const a = (minI + 1) % 3, b = (minI + 2) % 3;
        U[0 * 3 + minI] = U[1 * 3 + a] * U[2 * 3 + b] - U[2 * 3 + a] * U[1 * 3 + b];
        U[1 * 3 + minI] = U[2 * 3 + a] * U[0 * 3 + b] - U[0 * 3 + a] * U[2 * 3 + b];
        U[2 * 3 + minI] = U[0 * 3 + a] * U[1 * 3 + b] - U[1 * 3 + a] * U[0 * 3 + b];
      }
    }
    const Ut = mat3_transpose(U);
    let R = mat3_mul(V, Ut);
    if (mat3_det(R) < 0) {
      const Vfix = Float64Array.from(V);
      let minI = 0;
      if (S[1] < S[minI]) minI = 1;
      if (S[2] < S[minI]) minI = 2;
      Vfix[0 * 3 + minI] *= -1;
      Vfix[1 * 3 + minI] *= -1;
      Vfix[2 * 3 + minI] *= -1;
      R = mat3_mul(Vfix, Ut);
    }
    R = mat3_orthonormalize(R);
    const Rc0 = mat3_mulVec3(R, c0x, c0y, c0z);
    const t = [ctx - Rc0[0], cty - Rc0[1], ctz - Rc0[2]];
    return { R, t, cRest: [c0x, c0y, c0z], cCurr: [ctx, cty, ctz] };
  }

  // src/math/rotation.ts
  function rotateAroundAxis(px, py, pz, ox, oy, oz, ax, ay, az, angle) {
    const x = px - ox, y = py - oy, z = pz - oz;
    const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
    const len = Math.hypot(ax, ay, az) || 1;
    ax /= len;
    ay /= len;
    az /= len;
    const nx = (t * ax * ax + c) * x + (t * ax * ay - s * az) * y + (t * ax * az + s * ay) * z;
    const ny = (t * ax * ay + s * az) * x + (t * ay * ay + c) * y + (t * ay * az - s * ax) * z;
    const nz = (t * ax * az - s * ay) * x + (t * ay * az + s * ax) * y + (t * az * az + c) * z;
    return [nx + ox, ny + oy, nz + oz];
  }
  function applyDeltaRotation(fromDir, toDir, applyToDir) {
    let ax = vcross(fromDir, toDir);
    const axLen = vlen(ax);
    const d = Math.max(-1, Math.min(1, vdot(fromDir, toDir)));
    const ang = Math.acos(d);
    if (Math.abs(ang) < 1e-4) return applyToDir;
    if (axLen < 1e-6) {
      ax = Math.abs(fromDir[1]) < 0.9 ? vcross(fromDir, [0, 1, 0]) : vcross(fromDir, [1, 0, 0]);
      if (vlen(ax) < 1e-6) return applyToDir;
    }
    ax = vnorm(ax);
    const r = rotateAroundAxis(
      applyToDir[0],
      applyToDir[1],
      applyToDir[2],
      0,
      0,
      0,
      ax[0],
      ax[1],
      ax[2],
      ang
    );
    return vnorm(r);
  }
  function computeIKRotMatrix(restDir, targetDir) {
    let ax = vcross(restDir, targetDir);
    const axLen = vlen(ax);
    const d = Math.max(-1, Math.min(1, vdot(restDir, targetDir)));
    const ang = Math.acos(d);
    if (Math.abs(ang) < 1e-4) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
    if (axLen < 1e-6) {
      ax = Math.abs(restDir[1]) < 0.9 ? vcross(restDir, [0, 1, 0]) : vcross(restDir, [1, 0, 0]);
      if (vlen(ax) < 1e-6) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
    }
    ax = vnorm(ax);
    return mat3_rotAxis(ax[0], ax[1], ax[2], ang);
  }

  // src/math/interpolation.ts
  function smoothArray(arr, passes = 3) {
    const n = arr.length;
    let result = arr.slice();
    for (let p = 0; p < passes; p++) {
      const tmp = result.slice();
      for (let i = 0; i < n; i++) {
        result[i] = 0.25 * tmp[(i - 1 + n) % n] + 0.5 * tmp[i] + 0.25 * tmp[(i + 1) % n];
      }
    }
    return result;
  }
  function catmullRomVec3(p0, p1, p2, p3, t) {
    const t2 = t * t, t3 = t2 * t;
    const out = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      out[i] = 0.5 * (2 * p1[i] + (-p0[i] + p2[i]) * t + (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * t2 + (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * t3);
    }
    return out;
  }

  // src/skeleton/constants.ts
  var FBX_BONE_MAP = [
    // Spine / head
    { suffix: "Hips", name: "hips" },
    { suffix: "Spine", name: "spine_joint" },
    { suffix: "Spine1", name: "spine1_joint" },
    { suffix: "Spine2", name: "spine2_joint" },
    { suffix: "Neck", name: "neck" },
    { suffix: "Head", name: "head" },
    // Collars
    { suffix: "LeftShoulder", name: "l_collar" },
    { suffix: "RightShoulder", name: "r_collar" },
    // Left arm
    { suffix: "LeftArm", name: "l_shoulder" },
    { suffix: "LeftForeArm", name: "l_elbow" },
    { suffix: "LeftHand", name: "l_wrist" },
    // Left fingers
    { suffix: "LeftHandThumb1", name: "l_thumb1" },
    { suffix: "LeftHandThumb2", name: "l_thumb2" },
    { suffix: "LeftHandThumb3", name: "l_thumb3" },
    { suffix: "LeftHandThumb4", name: "l_thumb4" },
    { suffix: "LeftHandIndex1", name: "l_index1" },
    { suffix: "LeftHandIndex2", name: "l_index2" },
    { suffix: "LeftHandIndex3", name: "l_index3" },
    { suffix: "LeftHandIndex4", name: "l_index4" },
    { suffix: "LeftHandMiddle1", name: "l_mid_knuckle" },
    { suffix: "LeftHandMiddle2", name: "l_middle2" },
    { suffix: "LeftHandMiddle3", name: "l_middle3" },
    { suffix: "LeftHandMiddle4", name: "l_middle4" },
    { suffix: "LeftHandRing1", name: "l_ring1" },
    { suffix: "LeftHandRing2", name: "l_ring2" },
    { suffix: "LeftHandRing3", name: "l_ring3" },
    { suffix: "LeftHandRing4", name: "l_ring4" },
    { suffix: "LeftHandPinky1", name: "l_pinky1" },
    { suffix: "LeftHandPinky2", name: "l_pinky2" },
    { suffix: "LeftHandPinky3", name: "l_pinky3" },
    { suffix: "LeftHandPinky4", name: "l_pinky4" },
    // Left leg
    { suffix: "LeftUpLeg", name: "l_hip" },
    { suffix: "LeftLeg", name: "l_knee" },
    { suffix: "LeftFoot", name: "l_ankle" },
    { suffix: "LeftToeBase", name: "l_toe" },
    // Right arm
    { suffix: "RightArm", name: "r_shoulder" },
    { suffix: "RightForeArm", name: "r_elbow" },
    { suffix: "RightHand", name: "r_wrist" },
    // Right fingers
    { suffix: "RightHandThumb1", name: "r_thumb1" },
    { suffix: "RightHandThumb2", name: "r_thumb2" },
    { suffix: "RightHandThumb3", name: "r_thumb3" },
    { suffix: "RightHandThumb4", name: "r_thumb4" },
    { suffix: "RightHandIndex1", name: "r_index1" },
    { suffix: "RightHandIndex2", name: "r_index2" },
    { suffix: "RightHandIndex3", name: "r_index3" },
    { suffix: "RightHandIndex4", name: "r_index4" },
    { suffix: "RightHandMiddle1", name: "r_mid_knuckle" },
    { suffix: "RightHandMiddle2", name: "r_middle2" },
    { suffix: "RightHandMiddle3", name: "r_middle3" },
    { suffix: "RightHandMiddle4", name: "r_middle4" },
    { suffix: "RightHandRing1", name: "r_ring1" },
    { suffix: "RightHandRing2", name: "r_ring2" },
    { suffix: "RightHandRing3", name: "r_ring3" },
    { suffix: "RightHandRing4", name: "r_ring4" },
    { suffix: "RightHandPinky1", name: "r_pinky1" },
    { suffix: "RightHandPinky2", name: "r_pinky2" },
    { suffix: "RightHandPinky3", name: "r_pinky3" },
    { suffix: "RightHandPinky4", name: "r_pinky4" },
    // Right leg
    { suffix: "RightUpLeg", name: "r_hip" },
    { suffix: "RightLeg", name: "r_knee" },
    { suffix: "RightFoot", name: "r_ankle" },
    { suffix: "RightToeBase", name: "r_toe" }
  ];
  var FK_CHILDREN = {
    "hips": ["spine_joint", "l_hip", "r_hip"],
    "spine_joint": ["spine1_joint"],
    "spine1_joint": ["spine2_joint"],
    "spine2_joint": ["neck", "l_collar", "r_collar"],
    "neck": ["head"],
    "head": [],
    "l_collar": ["l_shoulder"],
    "r_collar": ["r_shoulder"],
    "l_hip": ["l_knee"],
    "r_hip": ["r_knee"],
    "l_knee": ["l_ankle"],
    "r_knee": ["r_ankle"],
    "l_ankle": ["l_toe"],
    "r_ankle": ["r_toe"],
    "l_toe": [],
    "r_toe": [],
    "l_shoulder": ["l_elbow"],
    "r_shoulder": ["r_elbow"],
    "l_elbow": ["l_wrist"],
    "r_elbow": ["r_wrist"],
    "l_wrist": ["l_mid_knuckle", "l_thumb1", "l_index1", "l_ring1", "l_pinky1"],
    "r_wrist": ["r_mid_knuckle", "r_thumb1", "r_index1", "r_ring1", "r_pinky1"],
    "l_thumb1": ["l_thumb2"],
    "l_thumb2": ["l_thumb3"],
    "l_thumb3": ["l_thumb4"],
    "l_thumb4": [],
    "l_index1": ["l_index2"],
    "l_index2": ["l_index3"],
    "l_index3": ["l_index4"],
    "l_index4": [],
    "l_mid_knuckle": ["l_middle2"],
    "l_middle2": ["l_middle3"],
    "l_middle3": ["l_middle4"],
    "l_middle4": [],
    "l_ring1": ["l_ring2"],
    "l_ring2": ["l_ring3"],
    "l_ring3": ["l_ring4"],
    "l_ring4": [],
    "l_pinky1": ["l_pinky2"],
    "l_pinky2": ["l_pinky3"],
    "l_pinky3": ["l_pinky4"],
    "l_pinky4": [],
    "r_thumb1": ["r_thumb2"],
    "r_thumb2": ["r_thumb3"],
    "r_thumb3": ["r_thumb4"],
    "r_thumb4": [],
    "r_index1": ["r_index2"],
    "r_index2": ["r_index3"],
    "r_index3": ["r_index4"],
    "r_index4": [],
    "r_mid_knuckle": ["r_middle2"],
    "r_middle2": ["r_middle3"],
    "r_middle3": ["r_middle4"],
    "r_middle4": [],
    "r_ring1": ["r_ring2"],
    "r_ring2": ["r_ring3"],
    "r_ring3": ["r_ring4"],
    "r_ring4": [],
    "r_pinky1": ["r_pinky2"],
    "r_pinky2": ["r_pinky3"],
    "r_pinky3": ["r_pinky4"],
    "r_pinky4": []
  };
  var BONE_SEGMENTS = [
    ["hips", "l_hip"],
    ["hips", "r_hip"],
    ["l_hip", "l_knee"],
    ["r_hip", "r_knee"],
    ["l_knee", "l_ankle"],
    ["r_knee", "r_ankle"],
    ["l_ankle", "l_toe"],
    ["r_ankle", "r_toe"],
    ["hips", "spine_joint"],
    ["spine_joint", "spine1_joint"],
    ["spine1_joint", "spine2_joint"],
    ["spine2_joint", "neck"],
    ["neck", "head"],
    ["l_shoulder", "l_elbow"],
    ["r_shoulder", "r_elbow"],
    ["l_elbow", "l_wrist"],
    ["r_elbow", "r_wrist"]
  ];
  var JOINT_PRIMARY_CHILD = {
    "hips": "spine_joint",
    "spine_joint": "spine1_joint",
    "spine1_joint": "spine2_joint",
    "spine2_joint": "neck",
    "neck": "head",
    "l_collar": "l_shoulder",
    "r_collar": "r_shoulder",
    "l_hip": "l_knee",
    "r_hip": "r_knee",
    "l_knee": "l_ankle",
    "r_knee": "r_ankle",
    "l_ankle": "l_toe",
    "r_ankle": "r_toe",
    "l_shoulder": "l_elbow",
    "r_shoulder": "r_elbow",
    "l_elbow": "l_wrist",
    "r_elbow": "r_wrist"
  };

  // src/skeleton/bone-matching.ts
  function matchFBXBone(boneName, boneMap = FBX_BONE_MAP) {
    const lower = boneName.toLowerCase();
    const sorted = boneMap.slice().sort((a, b) => b.suffix.length - a.suffix.length);
    for (const entry of sorted) {
      if (lower.endsWith(entry.suffix.toLowerCase())) return entry.name;
      if (entry.alt && lower.endsWith(entry.alt.toLowerCase())) return entry.name;
    }
    return null;
  }
  function matchFBXBoneToRegion(boneName, boneMap = FBX_BONE_MAP) {
    const mapped = matchFBXBone(boneName, boneMap);
    if (mapped) return mapped;
    const lower = boneName.toLowerCase();
    if (lower.includes("lefthand")) return "l_wrist";
    if (lower.includes("righthand")) return "r_wrist";
    return null;
  }

  // src/skeleton/fk.ts
  function computeFKChain(rootPos, restJoints, jointRMat, fkChildren) {
    const fkPos = { hips: rootPos };
    const queue = ["hips"];
    while (queue.length > 0) {
      const parent = queue.shift();
      const children = fkChildren[parent];
      if (!children) continue;
      const R_p = jointRMat[parent];
      if (!R_p || !restJoints[parent]) continue;
      for (const child of children) {
        if (!restJoints[child]) continue;
        const pRest = restJoints[parent];
        const cRest = restJoints[child];
        const bx = cRest[0] - pRest[0], by = cRest[1] - pRest[1], bz = cRest[2] - pRest[2];
        const rd = mat3_mulVec3(R_p, bx, by, bz);
        const pPos = fkPos[parent];
        fkPos[child] = [pPos[0] + rd[0], pPos[1] + rd[1], pPos[2] + rd[2]];
        queue.push(child);
      }
    }
    return fkPos;
  }
  function computeJointRotationMatrices(restQuats, currentQuats) {
    const jointRMat = {};
    for (const jn of Object.keys(restQuats)) {
      const curQ = currentQuats[jn];
      const restQ = restQuats[jn];
      if (!curQ || !restQ) continue;
      let qd = quat_mul(curQ, quat_conjugate(restQ));
      if (qd[3] < 0) qd = [-qd[0], -qd[1], -qd[2], -qd[3]];
      jointRMat[jn] = quat_to_mat3(qd);
    }
    return jointRMat;
  }
  function computeGroundCorrection(fkPos, floorY) {
    let minY = Infinity;
    for (const jn of Object.keys(fkPos)) {
      const pos = fkPos[jn];
      if (pos) minY = Math.min(minY, pos[1]);
    }
    return Math.max(0, floorY - minY);
  }

  // src/skeleton/twist.ts
  function extractBoneTwist(restBoneDir, curBoneDir, restChildDir, curChildDir) {
    const restCross = vcross(restBoneDir, restChildDir);
    const curCross = vcross(curBoneDir, curChildDir);
    if (vlen(restCross) < 2e-3 || vlen(curCross) < 0.05) return 0;
    const restBendN = vnorm(restCross);
    const curBendN = vnorm(curCross);
    const expectedBendN = applyDeltaRotation(restBoneDir, curBoneDir, restBendN);
    const projE = vsub(expectedBendN, vscale(curBoneDir, vdot(expectedBendN, curBoneDir)));
    const projC = vsub(curBendN, vscale(curBoneDir, vdot(curBendN, curBoneDir)));
    if (vlen(projE) < 1e-6 || vlen(projC) < 1e-6) return 0;
    const pE = vnorm(projE), pC = vnorm(projC);
    const dotTw = Math.max(-1, Math.min(1, vdot(pE, pC)));
    let twist = Math.acos(dotTw);
    const crossTw = vcross(pE, pC);
    if (vdot(crossTw, curBoneDir) < 0) twist = -twist;
    return twist;
  }
  function extractPositionTwist(swingR, targetBoneDir, refPerp, actualBendAxis, bendAngle) {
    if (Math.abs(bendAngle) < 0.1) return 0;
    const expected = vnorm(mat3_mulVec3(swingR, refPerp[0], refPerp[1], refPerp[2]));
    const projE = vsub(expected, vscale(targetBoneDir, vdot(expected, targetBoneDir)));
    const projA = vsub(actualBendAxis, vscale(targetBoneDir, vdot(actualBendAxis, targetBoneDir)));
    if (vlen(projE) < 1e-6 || vlen(projA) < 1e-6) return 0;
    const eN = vnorm(projE), aN = vnorm(projA);
    const cosT = Math.max(-1, Math.min(1, vdot(eN, aN)));
    const sinT = vdot(vcross(eN, aN), targetBoneDir);
    return Math.atan2(sinT, cosT);
  }
  function canonicalBendRef(rawRef, restBoneDir) {
    const d = vdot(rawRef, restBoneDir);
    const orth = vsub(rawRef, vscale(restBoneDir, d));
    return vlen(orth) > 1e-6 ? vnorm(orth) : rawRef;
  }

  // src/animation/keyframe.ts
  function getJointPositionAtTime(data, jointName, t) {
    if (!data || !data.joints[jointName]) return null;
    const jd = data.joints[jointName];
    const times = jd.times;
    const positions = jd.positions;
    const n = positions.length;
    if (n === 0) return null;
    if (n === 1) return positions[0].slice();
    const duration = data.duration;
    const cycleT = t % duration;
    let i = 0;
    while (i < n - 1 && times[i + 1] < cycleT) i++;
    const i0 = (i - 1 + n) % n;
    const i1 = i;
    const i2 = (i + 1) % n;
    const i3 = (i + 2) % n;
    const t0 = times[i1];
    const t1 = i2 === 0 ? duration : times[i2];
    let frac;
    if (cycleT < t0) {
      const tLast = times[n - 1];
      frac = (cycleT + duration - tLast) / (times[0] + duration - tLast);
      frac = Math.max(0, Math.min(1, frac));
      const p0 = positions[(n - 2 + n) % n];
      const p1 = positions[n - 1];
      const p2 = positions[0];
      const p3 = positions[Math.min(1, n - 1)];
      return catmullRomVec3(p0, p1, p2, p3, frac);
    }
    frac = t1 > t0 ? (cycleT - t0) / (t1 - t0) : 0;
    frac = Math.max(0, Math.min(1, frac));
    return catmullRomVec3(positions[i0], positions[i1], positions[i2], positions[i3], frac);
  }
  function getJointQuaternionAtTime(data, jointName, t) {
    if (!data || !data.joints[jointName]) return null;
    const jd = data.joints[jointName];
    const quats = jd.quaternions;
    if (!quats || quats.length === 0) return null;
    if (quats.length === 1) return quats[0].slice();
    const times = jd.times;
    const n = quats.length;
    const duration = data.duration;
    const cycleT = t % duration;
    let i = 0;
    while (i < n - 1 && times[i + 1] < cycleT) i++;
    const i1 = i;
    const i2 = (i + 1) % n;
    const t0 = times[i1];
    const t1 = i2 === 0 ? duration : times[i2];
    let frac;
    if (cycleT < t0) {
      const tLast = times[n - 1];
      frac = (cycleT + duration - tLast) / (times[0] + duration - tLast);
      frac = Math.max(0, Math.min(1, frac));
      return quat_slerp(quats[n - 1], quats[0], frac);
    }
    frac = t1 > t0 ? (cycleT - t0) / (t1 - t0) : 0;
    frac = Math.max(0, Math.min(1, frac));
    return quat_slerp(quats[i1], quats[i2], frac);
  }
  function getRootPositionAtTime(data, t) {
    if (!data || !data.root_positions) return null;
    const rp = data.root_positions;
    const n = rp.length;
    if (n === 0) return null;
    if (n === 1) return rp[0].slice();
    const duration = data.duration;
    const cycleT = t % duration;
    const fps = data.fps;
    const fExact = cycleT * fps;
    const i1 = Math.min(Math.floor(fExact), n - 1);
    const i2 = Math.min(i1 + 1, n - 1);
    const frac = fExact - Math.floor(fExact);
    return [
      rp[i1][0] + (rp[i2][0] - rp[i1][0]) * frac,
      rp[i1][1] + (rp[i2][1] - rp[i1][1]) * frac,
      rp[i1][2] + (rp[i2][2] - rp[i1][2]) * frac
    ];
  }
  function getJointTwistAtTime(data, boneName, childName, t) {
    const curQ = getJointQuaternionAtTime(data, boneName, t);
    if (!curQ || !data.rest_quats[boneName]) return 0;
    const restQ = data.rest_quats[boneName];
    const rp = data.rest_pose;
    if (!rp[boneName] || !rp[childName]) return 0;
    const boneDir = vnorm(vsub(rp[childName], rp[boneName]));
    return extractQuatTwist(restQ, curQ, boneDir);
  }

  // src/geometry/cross-section.ts
  function sliceMeshAtX(x0, triList, mPos) {
    const pts = [];
    for (let ti = 0; ti < triList.length; ti++) {
      const t = triList[ti];
      const ax = mPos[t.a3], ay = mPos[t.a3 + 1], az = mPos[t.a3 + 2];
      const bx = mPos[t.b3], by = mPos[t.b3 + 1], bz = mPos[t.b3 + 2];
      const cx = mPos[t.c3], cy = mPos[t.c3 + 1], cz = mPos[t.c3 + 2];
      const sa = ax - x0, sb = bx - x0, sc = cx - x0;
      if (sa > 0 && sb > 0 && sc > 0) continue;
      if (sa < 0 && sb < 0 && sc < 0) continue;
      const edges = [
        [sa, sb, ax, ay, az, bx, by, bz],
        [sb, sc, bx, by, bz, cx, cy, cz],
        [sa, sc, ax, ay, az, cx, cy, cz]
      ];
      for (let ei = 0; ei < 3; ei++) {
        const e = edges[ei];
        const d0 = e[0], d1 = e[1];
        if (d0 > 0 && d1 > 0 || d0 < 0 && d1 < 0) continue;
        if (d0 === 0 && d1 === 0) continue;
        const t_ = d0 / (d0 - d1);
        pts.push({
          y: e[3] + t_ * (e[6] - e[3]),
          z: e[4] + t_ * (e[7] - e[4])
        });
      }
    }
    return pts;
  }
  function filterHandTriangles(meshRestPos, meshIndex, knuckleX, knuckleY, knZMin, knZMax, handSign) {
    const handTris = [];
    const nTri = meshIndex.length / 3;
    const handSpanZ = knZMax - knZMin;
    const handZLo = knZMin - handSpanZ * 0.15;
    const handZHi = knZMax + handSpanZ * 0.15;
    for (let t = 0; t < nTri; t++) {
      const ai = meshIndex[t * 3], bi = meshIndex[t * 3 + 1], ci = meshIndex[t * 3 + 2];
      const a3 = ai * 3, b3 = bi * 3, c3 = ci * 3;
      const ax = meshRestPos[a3], bx = meshRestPos[b3], cx = meshRestPos[c3];
      const ay = meshRestPos[a3 + 1], by = meshRestPos[b3 + 1], cy = meshRestPos[c3 + 1];
      const az = meshRestPos[a3 + 2], bz = meshRestPos[b3 + 2], cz = meshRestPos[c3 + 2];
      const dxa = (ax - knuckleX) * handSign;
      const dxb = (bx - knuckleX) * handSign;
      const dxc = (cx - knuckleX) * handSign;
      if (dxa < -5e-3 && dxb < -5e-3 && dxc < -5e-3) continue;
      if (Math.abs(ay - knuckleY) > 0.04 && Math.abs(by - knuckleY) > 0.04 && Math.abs(cy - knuckleY) > 0.04) continue;
      const zMin3 = Math.min(az, bz, cz), zMax3 = Math.max(az, bz, cz);
      if (zMax3 < handZLo || zMin3 > handZHi) continue;
      handTris.push({ a3, b3, c3 });
    }
    return handTris;
  }

  // src/geometry/regression.ts
  function fitLineZD(pts) {
    const n = pts.length;
    if (n < 2) return null;
    let sD = 0, sZ = 0, sDD = 0, sDZ = 0;
    for (let i = 0; i < n; i++) {
      sD += pts[i].d;
      sZ += pts[i].z;
      sDD += pts[i].d * pts[i].d;
      sDZ += pts[i].d * pts[i].z;
    }
    const den = n * sDD - sD * sD;
    if (Math.abs(den) < 1e-12) return { slope: 0, intercept: sZ / n };
    const slope = (n * sDZ - sD * sZ) / den;
    const intercept = (sZ - slope * sD) / n;
    return { slope, intercept };
  }
  function evalBoundaryLine(line, d) {
    return line.slope * d + line.intercept;
  }

  // src/geometry/boundary-extrapolation.ts
  function findZRegions(dorsalPts, binW = 1e-3) {
    let zMin = Infinity, zMax = -Infinity;
    for (let pi = 0; pi < dorsalPts.length; pi++) {
      if (dorsalPts[pi].z < zMin) zMin = dorsalPts[pi].z;
      if (dorsalPts[pi].z > zMax) zMax = dorsalPts[pi].z;
    }
    const nBins = Math.ceil((zMax - zMin) / binW) + 1;
    if (nBins < 2 || nBins > 500) return [];
    const occupied = new Uint8Array(nBins);
    for (let pi = 0; pi < dorsalPts.length; pi++) {
      const bi = Math.min(nBins - 1, Math.floor((dorsalPts[pi].z - zMin) / binW));
      occupied[bi] = 1;
    }
    const regions = [];
    let regStart = -1;
    for (let bi = 0; bi <= nBins; bi++) {
      if (bi < nBins && occupied[bi]) {
        if (regStart < 0) regStart = bi;
      } else {
        if (regStart >= 0) {
          regions.push({
            zLo: zMin + regStart * binW,
            zHi: zMin + bi * binW
          });
          regStart = -1;
        }
      }
    }
    return regions;
  }
  function matchRegionsToFingers(regions, seedZ) {
    const fingerRI = [-1, -1, -1, -1];
    const fingerRD = [Infinity, Infinity, Infinity, Infinity];
    for (let ri = 0; ri < regions.length; ri++) {
      const rc = (regions[ri].zLo + regions[ri].zHi) / 2;
      let bestFi = 0, bestD = Math.abs(rc - seedZ[0]);
      for (let fi = 1; fi < 4; fi++) {
        const dd = Math.abs(rc - seedZ[fi]);
        if (dd < bestD) {
          bestD = dd;
          bestFi = fi;
        }
      }
      if (bestD < fingerRD[bestFi]) {
        fingerRD[bestFi] = bestD;
        fingerRI[bestFi] = ri;
      }
    }
    return fingerRI;
  }
  function extrapolateBoundaries(handTris, meshRestPos, knuckleX, knuckleY, handSign, seedZ, fingerLen) {
    const scanStep = 2e-3;
    const maxScanDist = fingerLen * 1.15;
    const scanStartDist = 5e-3;
    const binW = 1e-3;
    const bndPts = [[], [], [], [], []];
    let fiByZ = null;
    for (let d = scanStartDist; d <= maxScanDist; d += scanStep) {
      const xPos = knuckleX + d * handSign;
      const pts = sliceMeshAtX(xPos, handTris, meshRestPos);
      const rawPts = [];
      for (let pi = 0; pi < pts.length; pi++) {
        if (Math.abs(pts[pi].y - knuckleY) <= 0.035) rawPts.push(pts[pi]);
      }
      if (rawPts.length < 4) continue;
      rawPts.sort((a, b) => a.y - b.y);
      const medianY = rawPts[Math.floor(rawPts.length / 2)].y;
      const dorsalPts = [];
      for (let pi = 0; pi < rawPts.length; pi++) {
        if (rawPts[pi].y >= medianY) dorsalPts.push(rawPts[pi]);
      }
      if (dorsalPts.length < 2) continue;
      const regions = findZRegions(dorsalPts, binW);
      if (regions.length < 4) continue;
      const fingerRI = matchRegionsToFingers(regions, seedZ);
      if (fingerRI[0] < 0 || fingerRI[1] < 0 || fingerRI[2] < 0 || fingerRI[3] < 0) continue;
      const curOrder = [0, 1, 2, 3].sort((a, b) => regions[fingerRI[a]].zLo + regions[fingerRI[a]].zHi - (regions[fingerRI[b]].zLo + regions[fingerRI[b]].zHi));
      if (!fiByZ) fiByZ = curOrder.slice();
      bndPts[0].push({ d, z: regions[fingerRI[fiByZ[0]]].zLo });
      bndPts[4].push({ d, z: regions[fingerRI[fiByZ[3]]].zHi });
      for (let gi = 0; gi < 3; gi++) {
        const gapZ = (regions[fingerRI[fiByZ[gi]]].zHi + regions[fingerRI[fiByZ[gi + 1]]].zLo) / 2;
        bndPts[gi + 1].push({ d, z: gapZ });
      }
    }
    const bndLines = bndPts.map(fitLineZD);
    const ok = fiByZ !== null && bndLines.every((l) => l !== null);
    const fingerBndLo = [0, 0, 0, 0];
    const fingerBndHi = [0, 0, 0, 0];
    if (fiByZ) {
      for (let k = 0; k < 4; k++) {
        fingerBndLo[fiByZ[k]] = k;
        fingerBndHi[fiByZ[k]] = k + 1;
      }
    }
    return {
      boundaryLines: bndLines,
      fingerOrderByZ: fiByZ,
      fingerBndLo,
      fingerBndHi,
      ok
    };
  }

  // src/geometry/hip-detection.ts
  function detectHipGeometry(roughY, meshH, meshRestPos) {
    if (!meshRestPos) {
      const hipsCenterY2 = roughY + meshH * 0.096;
      const hipJointY2 = hipsCenterY2 - meshH * 0.0316;
      return {
        hipsCenterY: hipsCenterY2,
        hipJointY: hipJointY2,
        hipSpreadHalf: meshH * 0.0445,
        crotchY: roughY,
        hipShelfY: hipsCenterY2 + meshH * 0.014
      };
    }
    const nMesh = meshRestPos.length / 3;
    const scanStep = meshH * 3e-3;
    const crotchScanMin = roughY - meshH * 0.1;
    const crotchScanMax = roughY + meshH * 0.05;
    let maxCenterDensity = 0;
    let crotchY = roughY;
    for (let y = crotchScanMin; y < crotchScanMax; y += scanStep) {
      let centerCount = 0;
      for (let i = 0; i < nMesh; i++) {
        if (Math.abs(meshRestPos[i * 3 + 1] - y) < scanStep && Math.abs(meshRestPos[i * 3]) < 0.015) {
          centerCount++;
        }
      }
      if (centerCount > maxCenterDensity) {
        maxCenterDensity = centerCount;
        crotchY = y;
      }
    }
    let maxWidth = 0, hipShelfY = crotchY;
    for (let y = crotchY + meshH * 0.02; y < crotchY + meshH * 0.15; y += scanStep) {
      let minX = Infinity, maxX = -Infinity;
      for (let i = 0; i < nMesh; i++) {
        if (Math.abs(meshRestPos[i * 3 + 1] - y) < scanStep) {
          minX = Math.min(minX, meshRestPos[i * 3]);
          maxX = Math.max(maxX, meshRestPos[i * 3]);
        }
      }
      const w = maxX - minX;
      if (w > maxWidth) {
        maxWidth = w;
        hipShelfY = y;
      }
    }
    const hipsCenterY = hipShelfY - meshH * 0.014;
    const hipJointY = hipsCenterY - meshH * 0.0316;
    const hipSpreadHalf = meshH * 0.0445;
    return { hipsCenterY, hipJointY, hipSpreadHalf, crotchY, hipShelfY };
  }

  // src/geometry/toe-detection.ts
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

  // src/weights/barycentric-transfer.ts
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

  // src/weights/sharpening.ts
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

  // src/weights/lbs.ts
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

  // src/geometry/plane-slice.ts
  function buildPlaneFrame(normal) {
    const [nx, ny, nz] = normal;
    let tx, ty, tz;
    if (Math.abs(nx) < 0.9) {
      tx = 0;
      ty = nz;
      tz = -ny;
    } else {
      tx = -nz;
      ty = 0;
      tz = nx;
    }
    const tLen = Math.sqrt(tx * tx + ty * ty + tz * tz);
    tx /= tLen;
    ty /= tLen;
    tz /= tLen;
    const bx = ny * tz - nz * ty;
    const by = nz * tx - nx * tz;
    const bz = nx * ty - ny * tx;
    return [[tx, ty, tz], [bx, by, bz]];
  }
  function sliceMeshAtPlane(planePoint, planeNormal, meshPos, indices, boneAxisOrOpts) {
    let boneAxis;
    let maxRadius = Infinity;
    if (boneAxisOrOpts) {
      if (Array.isArray(boneAxisOrOpts)) {
        boneAxis = boneAxisOrOpts;
      } else {
        boneAxis = boneAxisOrOpts.boneAxis;
        if (boneAxisOrOpts.maxRadius !== void 0) {
          maxRadius = boneAxisOrOpts.maxRadius;
        }
      }
    }
    const [px, py, pz] = planePoint;
    const [nx, ny, nz] = planeNormal;
    const maxRadiusSq = maxRadius * maxRadius;
    const [tangent, bitangent] = buildPlaneFrame(planeNormal);
    const [tx, ty, tz] = tangent;
    const [bx, by, bz] = bitangent;
    const points = [];
    const nTri = indices.length / 3;
    for (let ti = 0; ti < nTri; ti++) {
      const ai = indices[ti * 3], bi2 = indices[ti * 3 + 1], ci = indices[ti * 3 + 2];
      const a3 = ai * 3, b3 = bi2 * 3, c3 = ci * 3;
      const ax = meshPos[a3], ay = meshPos[a3 + 1], az = meshPos[a3 + 2];
      const bvx = meshPos[b3], bvy = meshPos[b3 + 1], bvz = meshPos[b3 + 2];
      const cx = meshPos[c3], cy = meshPos[c3 + 1], cz = meshPos[c3 + 2];
      if (maxRadius < Infinity) {
        const distSqA = (ax - px) ** 2 + (ay - py) ** 2 + (az - pz) ** 2;
        const distSqB = (bvx - px) ** 2 + (bvy - py) ** 2 + (bvz - pz) ** 2;
        const distSqC = (cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2;
        if (distSqA > maxRadiusSq && distSqB > maxRadiusSq && distSqC > maxRadiusSq) continue;
      }
      const da = (ax - px) * nx + (ay - py) * ny + (az - pz) * nz;
      const db = (bvx - px) * nx + (bvy - py) * ny + (bvz - pz) * nz;
      const dc = (cx - px) * nx + (cy - py) * ny + (cz - pz) * nz;
      if (da > 0 && db > 0 && dc > 0) continue;
      if (da < 0 && db < 0 && dc < 0) continue;
      const edges = [
        [da, db, ax, ay, az, bvx, bvy, bvz],
        [db, dc, bvx, bvy, bvz, cx, cy, cz],
        [da, dc, ax, ay, az, cx, cy, cz]
      ];
      for (const e of edges) {
        const d0 = e[0], d1 = e[1];
        if (d0 > 0 && d1 > 0 || d0 < 0 && d1 < 0) continue;
        if (d0 === 0 && d1 === 0) continue;
        const t = d0 / (d0 - d1);
        const wx = e[2] + t * (e[5] - e[2]);
        const wy = e[3] + t * (e[6] - e[3]);
        const wz = e[4] + t * (e[7] - e[4]);
        const dx = wx - px, dy = wy - py, dz = wz - pz;
        const u = dx * tx + dy * ty + dz * tz;
        const v = dx * bx + dy * by + dz * bz;
        points.push({ wx, wy, wz, u, v });
      }
    }
    let sumU = 0, sumV = 0, sumWx = 0, sumWy = 0, sumWz = 0;
    for (const pt of points) {
      sumU += pt.u;
      sumV += pt.v;
      sumWx += pt.wx;
      sumWy += pt.wy;
      sumWz += pt.wz;
    }
    const n = points.length || 1;
    const centroid2D = { u: sumU / n, v: sumV / n };
    const centroid3D = [sumWx / n, sumWy / n, sumWz / n];
    const offsetWorld = [
      centroid3D[0] - px,
      centroid3D[1] - py,
      centroid3D[2] - pz
    ];
    const axis = boneAxis ?? planeNormal;
    const [anx, any_, anz] = axis;
    const [at, ab] = buildPlaneFrame(axis);
    const alongBone = offsetWorld[0] * anx + offsetWorld[1] * any_ + offsetWorld[2] * anz;
    const lateral = offsetWorld[0] * at[0] + offsetWorld[1] * at[1] + offsetWorld[2] * at[2];
    const depth = offsetWorld[0] * ab[0] + offsetWorld[1] * ab[1] + offsetWorld[2] * ab[2];
    const offsetLocal = [alongBone, lateral, depth];
    return {
      points,
      centroid2D,
      centroid3D,
      offsetWorld,
      offsetLocal
    };
  }

  // src/geometry/mesh-adjacency.ts
  function buildMeshAdjacency(meshRestPos, indices) {
    const nTri = indices.length / 3;
    const nMesh = meshRestPos.length / 3;
    const edgeSet = /* @__PURE__ */ new Set();
    const edgeAArr = [];
    const edgeBArr = [];
    const edgeRestLenArr = [];
    for (let t = 0; t < nTri; t++) {
      const a = indices[t * 3], b = indices[t * 3 + 1], c = indices[t * 3 + 2];
      const pairs = [[a, b], [b, c], [a, c]];
      for (const [v0, v1] of pairs) {
        const lo = Math.min(v0, v1), hi = Math.max(v0, v1);
        const key = lo * nMesh + hi;
        if (!edgeSet.has(key)) {
          edgeSet.add(key);
          edgeAArr.push(lo);
          edgeBArr.push(hi);
          const dx = meshRestPos[hi * 3] - meshRestPos[lo * 3];
          const dy = meshRestPos[hi * 3 + 1] - meshRestPos[lo * 3 + 1];
          const dz = meshRestPos[hi * 3 + 2] - meshRestPos[lo * 3 + 2];
          edgeRestLenArr.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
        }
      }
    }
    const nEdges = edgeAArr.length;
    const edgeA = new Uint32Array(edgeAArr);
    const edgeB = new Uint32Array(edgeBArr);
    const edgeRestLen = new Float32Array(edgeRestLenArr);
    const adjCount = new Uint16Array(nMesh);
    for (let i = 0; i < nEdges; i++) {
      adjCount[edgeA[i]]++;
      adjCount[edgeB[i]]++;
    }
    const adjStart = new Uint32Array(nMesh);
    for (let i = 1; i < nMesh; i++) adjStart[i] = adjStart[i - 1] + adjCount[i - 1];
    const totalAdj = nMesh > 0 ? adjStart[nMesh - 1] + adjCount[nMesh - 1] : 0;
    const adjList = new Uint32Array(totalAdj);
    const adjFill = new Uint16Array(nMesh);
    for (let i = 0; i < nEdges; i++) {
      const a = edgeA[i], b = edgeB[i];
      adjList[adjStart[a] + adjFill[a]++] = b;
      adjList[adjStart[b] + adjFill[b]++] = a;
    }
    return {
      edgeA,
      edgeB,
      edgeRestLen,
      nEdges,
      nVerts: nMesh,
      adjList,
      adjStart,
      adjCount
    };
  }

  // src/qa/skeleton-qa.ts
  var STANDARD_JOINTS = [
    "hips",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
    "l_toe",
    "r_toe",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_mid_knuckle",
    "r_mid_knuckle",
    "neck",
    "head",
    "chest",
    "spine_joint",
    "spine1_joint",
    "spine2_joint",
    "l_collar",
    "r_collar"
  ];
  var BONE_DEFS = [
    ["hips", "l_hip", "L hip offset"],
    ["hips", "r_hip", "R hip offset"],
    ["l_hip", "l_knee", "L thigh"],
    ["r_hip", "r_knee", "R thigh"],
    ["l_knee", "l_ankle", "L shin"],
    ["r_knee", "r_ankle", "R shin"],
    ["l_ankle", "l_toe", "L foot"],
    ["r_ankle", "r_toe", "R foot"],
    ["l_shoulder", "l_elbow", "L upper arm"],
    ["r_shoulder", "r_elbow", "R upper arm"],
    ["l_elbow", "l_wrist", "L forearm"],
    ["r_elbow", "r_wrist", "R forearm"],
    ["l_wrist", "l_mid_knuckle", "L hand"],
    ["r_wrist", "r_mid_knuckle", "R hand"],
    ["hips", "spine_joint", "Pelvis\u2192spine"],
    ["spine_joint", "spine1_joint", "Spine 0\u21921"],
    ["spine1_joint", "spine2_joint", "Spine 1\u21922"],
    ["spine2_joint", "neck", "Spine 2\u2192neck"],
    ["neck", "head", "Neck\u2192head"],
    ["l_collar", "l_shoulder", "L collar"],
    ["r_collar", "r_shoulder", "R collar"]
  ];
  function boneLen(src, a, b) {
    const pa = src[a], pb = src[b];
    if (!pa || !pb) return NaN;
    return Math.sqrt((pb[0] - pa[0]) ** 2 + (pb[1] - pa[1]) ** 2 + (pb[2] - pa[2]) ** 2) * 1e3;
  }
  function compareSkeletons(ourJoints, refRestPose, anchor, origJoints) {
    const refWorld = {};
    if (refRestPose) {
      refWorld["hips"] = anchor;
      for (const jn of STANDARD_JOINTS) {
        if (jn === "hips") continue;
        const rp = refRestPose[jn];
        if (rp) {
          refWorld[jn] = [rp[0] + anchor[0], rp[1] + anchor[1], rp[2] + anchor[2]];
        }
      }
    }
    const jointComparisons = [];
    let maxDiff = 0, maxDiffName = "";
    for (const jn of STANDARD_JOINTS) {
      const pj = ourJoints[jn] || null;
      const rpj = refWorld[jn] || null;
      let diffMm = 0;
      let status = "OK";
      if (!pj && !rpj) continue;
      if (!pj) {
        status = "MISSING_OURS";
      } else if (!rpj) {
        status = "MISSING_REF";
      } else {
        diffMm = Math.sqrt(
          (pj[0] - rpj[0]) ** 2 + (pj[1] - rpj[1]) ** 2 + (pj[2] - rpj[2]) ** 2
        ) * 1e3;
        if (diffMm > 1) status = "MISMATCH";
        if (diffMm > maxDiff) {
          maxDiff = diffMm;
          maxDiffName = jn;
        }
      }
      jointComparisons.push({ jointName: jn, ourPos: pj, refPos: rpj, diffMm, status });
    }
    const boneLengthComparisons = [];
    for (const [a, b, label] of BONE_DEFS) {
      const pLen = boneLen(ourJoints, a, b);
      const fLen = boneLen(refWorld, a, b);
      const oLen = origJoints ? boneLen(origJoints, a, b) : NaN;
      const ratio = !isNaN(oLen) && !isNaN(fLen) && fLen > 0 ? oLen / fLen : NaN;
      const isMismatch = !isNaN(ratio) && (ratio < 0.9 || ratio > 1.1);
      boneLengthComparisons.push({
        label,
        parent: a,
        child: b,
        ourLength: pLen,
        refLength: fLen,
        origLength: oLen,
        ratio,
        isMismatch
      });
    }
    const totalLegOurs = boneLen(ourJoints, "l_hip", "l_knee") + boneLen(ourJoints, "l_knee", "l_ankle");
    const totalLegRef = boneLen(refWorld, "l_hip", "l_knee") + boneLen(refWorld, "l_knee", "l_ankle");
    const totalArmOurs = boneLen(ourJoints, "l_shoulder", "l_elbow") + boneLen(ourJoints, "l_elbow", "l_wrist");
    const totalArmRef = boneLen(refWorld, "l_shoulder", "l_elbow") + boneLen(refWorld, "l_elbow", "l_wrist");
    return {
      jointComparisons,
      boneLengthComparisons,
      maxJointDiffMm: maxDiff,
      maxJointDiffName: maxDiffName,
      totalLegOurs,
      totalLegRef,
      totalArmOurs,
      totalArmRef
    };
  }
  function checkSymmetry(joints) {
    const PAIRS = [
      ["l_hip", "l_knee", "r_hip", "r_knee", "Thigh"],
      ["l_knee", "l_ankle", "r_knee", "r_ankle", "Shin"],
      ["l_ankle", "l_toe", "r_ankle", "r_toe", "Foot"],
      ["l_shoulder", "l_elbow", "r_shoulder", "r_elbow", "Upper arm"],
      ["l_elbow", "l_wrist", "r_elbow", "r_wrist", "Forearm"]
    ];
    return PAIRS.map(([lp, lc, rp, rc, label]) => {
      const leftMm = boneLen(joints, lp, lc);
      const rightMm = boneLen(joints, rp, rc);
      const ratio = !isNaN(leftMm) && !isNaN(rightMm) && rightMm > 0 ? leftMm / rightMm : NaN;
      const isMismatch = !isNaN(ratio) && (ratio < 0.95 || ratio > 1.05);
      return { label, leftMm, rightMm, ratio, isMismatch };
    });
  }

  // src/qa/deformation-qa.ts
  function computeRigidityStats(boneTransforms, deformedPos, meshRestPos, meshBoneWeights, joints, boneNameToIdx, jointPrimaryChild, boneSegments, alpha) {
    const nMesh = meshRestPos.length / 3;
    const boneSegInfo = {};
    for (const [biStr, jn] of Object.entries(boneNameToIdx)) {
      const bi = +biStr;
      const child = jointPrimaryChild[jn];
      if (child && joints[jn] && joints[child]) {
        const dx = joints[child][0] - joints[jn][0];
        const dy = joints[child][1] - joints[jn][1];
        const dz = joints[child][2] - joints[jn][2];
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        boneSegInfo[bi] = { segKey: jn + ">" + child, segLen: len || 0.1, parent: jn, child };
      } else {
        for (const [par, ch] of boneSegments) {
          if (ch === jn && joints[par] && joints[jn]) {
            const dx = joints[jn][0] - joints[par][0];
            const dy = joints[jn][1] - joints[par][1];
            const dz = joints[jn][2] - joints[par][2];
            const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
            boneSegInfo[bi] = { segKey: par + ">" + jn, segLen: len || 0.1, parent: par, child: jn };
            break;
          }
        }
      }
    }
    const segAccum = {};
    for (const bi of Object.keys(boneSegInfo)) {
      const si = boneSegInfo[+bi];
      if (!segAccum[si.segKey]) {
        segAccum[si.segKey] = {
          sum: 0,
          sumSq: 0,
          count: 0,
          max: 0,
          blendSum: 0,
          highErrCount: 0,
          sumDist: 0,
          maxDist: 0,
          label: si.parent + "\u2192" + si.child,
          parent: si.parent,
          child: si.child
        };
      }
    }
    for (let i = 0; i < nMesh; i++) {
      const bw = meshBoneWeights[i];
      if (!bw || bw.length < 2) continue;
      let domBi = bw[0], domW = bw[1];
      for (let e = 2; e < bw.length; e += 2) {
        if (bw[e + 1] > domW) {
          domBi = bw[e];
          domW = bw[e + 1];
        }
      }
      const blendFactor = 1 - domW;
      const tf = boneTransforms[domBi];
      if (!tf) continue;
      const rx = meshRestPos[i * 3], ry = meshRestPos[i * 3 + 1], rz = meshRestPos[i * 3 + 2];
      const R = tf.R, t = tf.t;
      const rpx = R[0] * rx + R[1] * ry + R[2] * rz + t[0];
      const rpy = R[3] * rx + R[4] * ry + R[5] * rz + t[1];
      const rpz = R[6] * rx + R[7] * ry + R[8] * rz + t[2];
      const rigidX = rx + alpha * (rpx - rx);
      const rigidY = ry + alpha * (rpy - ry);
      const rigidZ = rz + alpha * (rpz - rz);
      const lbsX = deformedPos[i * 3], lbsY = deformedPos[i * 3 + 1], lbsZ = deformedPos[i * 3 + 2];
      const dx = lbsX - rigidX, dy = lbsY - rigidY, dz = lbsZ - rigidZ;
      const errDist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const si = boneSegInfo[domBi];
      const segLen = si ? si.segLen : 0.1;
      const errNorm = errDist / segLen;
      if (si && domW >= 0.8) {
        const st = segAccum[si.segKey];
        st.sum += errNorm;
        st.sumSq += errNorm * errNorm;
        st.count++;
        if (errNorm > st.max) st.max = errNorm;
        st.blendSum += blendFactor;
        if (errNorm > 0.02) st.highErrCount++;
        st.sumDist += errDist;
        if (errDist > st.maxDist) st.maxDist = errDist;
      }
    }
    const results = [];
    for (const key of Object.keys(segAccum).sort()) {
      const st = segAccum[key];
      if (st.count === 0) continue;
      const meanErr = st.sum / st.count;
      const meanBlend = st.blendSum / st.count;
      let tag;
      if (meanErr < 5e-3 && meanBlend < 0.2) tag = "GOOD";
      else if (meanErr > 0.02 && meanBlend < 0.2) tag = "TRANSFORM?";
      else if (meanErr > 0.02 && meanBlend > 0.4) tag = "WEIGHTS";
      else if (meanErr < 0.01 && meanBlend > 0.4) tag = "SHIFT";
      else tag = "OK";
      results.push({
        segKey: key,
        label: st.label,
        parent: st.parent,
        child: st.child,
        meanError: meanErr,
        maxError: st.max,
        meanBlend,
        highErrCount: st.highErrCount,
        count: st.count,
        meanDistMm: st.sumDist / st.count * 1e3,
        maxDistMm: st.maxDist * 1e3,
        tag
      });
    }
    return results;
  }
  function computeStrain(deformedPos, adjacency) {
    const { edgeA, edgeB, edgeRestLen, nEdges, nVerts } = adjacency;
    const worstStrain = new Float32Array(nVerts).fill(1);
    for (let i = 0; i < nEdges; i++) {
      const a = edgeA[i], b = edgeB[i];
      const dx = deformedPos[b * 3] - deformedPos[a * 3];
      const dy = deformedPos[b * 3 + 1] - deformedPos[a * 3 + 1];
      const dz = deformedPos[b * 3 + 2] - deformedPos[a * 3 + 2];
      const curLen = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const rest = edgeRestLen[i];
      if (rest < 1e-8) continue;
      const ratio = curLen / rest;
      const deviation = Math.abs(ratio - 1);
      if (deviation > Math.abs(worstStrain[a] - 1)) worstStrain[a] = ratio;
      if (deviation > Math.abs(worstStrain[b] - 1)) worstStrain[b] = ratio;
    }
    let highStrainCount = 0, totalDeviation = 0;
    for (let i = 0; i < nVerts; i++) {
      const dev = Math.abs(worstStrain[i] - 1);
      totalDeviation += dev;
      if (dev > 0.3) highStrainCount++;
    }
    return {
      worstStrain,
      highStrainCount,
      avgDeviation: totalDeviation / nVerts,
      nVerts,
      nEdges
    };
  }
  var CIRCUMFERENCE_SEGMENTS = [
    { parent: "l_hip", child: "l_knee", label: "L thigh" },
    { parent: "r_hip", child: "r_knee", label: "R thigh" },
    { parent: "l_knee", child: "l_ankle", label: "L shin" },
    { parent: "r_knee", child: "r_ankle", label: "R shin" },
    { parent: "l_shoulder", child: "l_elbow", label: "L upper arm" },
    { parent: "r_shoulder", child: "r_elbow", label: "R upper arm" },
    { parent: "l_elbow", child: "l_wrist", label: "L forearm" },
    { parent: "r_elbow", child: "r_wrist", label: "R forearm" }
  ];
  function computeCircumference(deformedPos, meshRestPos, meshBoneWeights, restJoints, fkJoints, boneNameToIdx, groundCorrection, segments = CIRCUMFERENCE_SEGMENTS) {
    const nMesh = meshRestPos.length / 3;
    const results = [];
    const jointToBoneIdx = {};
    for (const [biStr, jn] of Object.entries(boneNameToIdx)) {
      jointToBoneIdx[jn] = +biStr;
    }
    for (const seg of segments) {
      const pRest = restJoints[seg.parent], cRest = restJoints[seg.child];
      const pDef = fkJoints[seg.parent], cDef = fkJoints[seg.child];
      if (!pRest || !cRest || !pDef || !cDef) continue;
      const rax = cRest[0] - pRest[0], ray = cRest[1] - pRest[1], raz = cRest[2] - pRest[2];
      const raLen = Math.sqrt(rax * rax + ray * ray + raz * raz);
      if (raLen < 1e-6) continue;
      const rnx = rax / raLen, rny = ray / raLen, rnz = raz / raLen;
      const dax = cDef[0] - pDef[0], day = cDef[1] - pDef[1], daz = cDef[2] - pDef[2];
      const daLen = Math.sqrt(dax * dax + day * day + daz * daz);
      if (daLen < 1e-6) continue;
      const dnx = dax / daLen, dny = day / daLen, dnz = daz / daLen;
      const dpx = pDef[0], dpy = pDef[1] + groundCorrection, dpz = pDef[2];
      const ratios = [];
      const vertIndices = [];
      for (let i = 0; i < nMesh; i++) {
        const bw = meshBoneWeights[i];
        if (!bw || bw.length < 2) continue;
        let domBi = bw[0], domW = bw[1];
        for (let e = 2; e < bw.length; e += 2) {
          if (bw[e + 1] > domW) {
            domBi = bw[e];
            domW = bw[e + 1];
          }
        }
        const domJn = boneNameToIdx[domBi];
        if (domJn !== seg.parent) continue;
        if (domW < 0.5) continue;
        const rx = meshRestPos[i * 3], ry = meshRestPos[i * 3 + 1], rz = meshRestPos[i * 3 + 2];
        const vx = rx - pRest[0], vy = ry - pRest[1], vz = rz - pRest[2];
        const t_rest = vx * rnx + vy * rny + vz * rnz;
        const t_norm = t_rest / raLen;
        if (t_norm < 0.1 || t_norm > 0.9) continue;
        const prx = vx - t_rest * rnx, pry = vy - t_rest * rny, prz = vz - t_rest * rnz;
        const r_rest = Math.sqrt(prx * prx + pry * pry + prz * prz);
        if (r_rest < 1e-5) continue;
        const dx2 = deformedPos[i * 3] - dpx, dy2 = deformedPos[i * 3 + 1] - dpy, dz2 = deformedPos[i * 3 + 2] - dpz;
        const t_def = dx2 * dnx + dy2 * dny + dz2 * dnz;
        const pdx = dx2 - t_def * dnx, pdy = dy2 - t_def * dny, pdz = dz2 - t_def * dnz;
        const r_def = Math.sqrt(pdx * pdx + pdy * pdy + pdz * pdz);
        ratios.push(r_def / r_rest);
        vertIndices.push(i);
      }
      if (ratios.length < 5) continue;
      const sorted = ratios.slice().sort((a, b) => a - b);
      const n = sorted.length;
      const p5 = sorted[Math.floor(n * 0.05)];
      const p50 = sorted[Math.floor(n * 0.5)];
      const p95 = sorted[Math.floor(n * 0.95)];
      const mean = ratios.reduce((a, b) => a + b, 0) / n;
      let sumR = 0, sumRSq = 0, sumR0 = 0, sumR0Sq = 0;
      for (let j = 0; j < ratios.length; j++) {
        const vi = vertIndices[j];
        const rx = meshRestPos[vi * 3], ry = meshRestPos[vi * 3 + 1], rz = meshRestPos[vi * 3 + 2];
        const vx = rx - pRest[0], vy = ry - pRest[1], vz = rz - pRest[2];
        const t_rest = vx * rnx + vy * rny + vz * rnz;
        const prx2 = vx - t_rest * rnx, pry2 = vy - t_rest * rny, prz2 = vz - t_rest * rnz;
        const r0 = Math.sqrt(prx2 * prx2 + pry2 * pry2 + prz2 * prz2);
        const dx3 = deformedPos[vi * 3] - dpx, dy3 = deformedPos[vi * 3 + 1] - dpy, dz3 = deformedPos[vi * 3 + 2] - dpz;
        const t_def2 = dx3 * dnx + dy3 * dny + dz3 * dnz;
        const pdx2 = dx3 - t_def2 * dnx, pdy2 = dy3 - t_def2 * dny, pdz2 = dz3 - t_def2 * dnz;
        const rD = Math.sqrt(pdx2 * pdx2 + pdy2 * pdy2 + pdz2 * pdz2);
        sumR0 += r0;
        sumR0Sq += r0 * r0;
        sumR += rD;
        sumRSq += rD * rD;
      }
      const meanR0 = sumR0 / n, meanR = sumR / n;
      const stdR0 = Math.sqrt(Math.max(0, sumR0Sq / n - meanR0 * meanR0));
      const stdR = Math.sqrt(Math.max(0, sumRSq / n - meanR * meanR));
      const cvRest = meanR0 > 1e-6 ? stdR0 / meanR0 : 0;
      const cvDef = meanR > 1e-6 ? stdR / meanR : 0;
      const flattenScore = cvDef - cvRest;
      const tag = p5 < 0.8 ? "COLLAPSE" : p5 < 0.9 ? "COMPRESS" : mean > 1.1 ? "BULGE" : "OK";
      results.push({
        label: seg.label,
        parent: seg.parent,
        child: seg.child,
        mean,
        p5,
        p50,
        p95,
        flattenScore,
        count: n,
        tag
      });
    }
    return results;
  }
  return __toCommonJS(browser_exports);
})();
