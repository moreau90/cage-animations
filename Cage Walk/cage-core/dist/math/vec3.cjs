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

// src/math/vec3.ts
var vec3_exports = {};
__export(vec3_exports, {
  clamp: () => clamp,
  vadd: () => vadd,
  vcross: () => vcross,
  vdot: () => vdot,
  vlen: () => vlen,
  vnorm: () => vnorm,
  vscale: () => vscale,
  vsub: () => vsub
});
module.exports = __toCommonJS(vec3_exports);
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
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  clamp,
  vadd,
  vcross,
  vdot,
  vlen,
  vnorm,
  vscale,
  vsub
});
