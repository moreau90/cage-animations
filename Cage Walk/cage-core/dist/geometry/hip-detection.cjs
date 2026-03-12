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

// src/geometry/hip-detection.ts
var hip_detection_exports = {};
__export(hip_detection_exports, {
  detectHipGeometry: () => detectHipGeometry
});
module.exports = __toCommonJS(hip_detection_exports);
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
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  detectHipGeometry
});
