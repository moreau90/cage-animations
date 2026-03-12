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

// src/geometry/regression.ts
var regression_exports = {};
__export(regression_exports, {
  evalBoundaryLine: () => evalBoundaryLine,
  fitLineZD: () => fitLineZD
});
module.exports = __toCommonJS(regression_exports);
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
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  evalBoundaryLine,
  fitLineZD
});
