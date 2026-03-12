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

// src/math/interpolation.ts
var interpolation_exports = {};
__export(interpolation_exports, {
  catmullRomVec3: () => catmullRomVec3,
  smoothArray: () => smoothArray
});
module.exports = __toCommonJS(interpolation_exports);
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
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  catmullRomVec3,
  smoothArray
});
