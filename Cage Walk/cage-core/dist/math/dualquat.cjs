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

// src/math/dualquat.ts
var dualquat_exports = {};
__export(dualquat_exports, {
  dq_apply: () => dq_apply,
  quat_rotateVec3: () => quat_rotateVec3,
  rigid_to_dq: () => rigid_to_dq
});
module.exports = __toCommonJS(dualquat_exports);
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
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  dq_apply,
  quat_rotateVec3,
  rigid_to_dq
});
