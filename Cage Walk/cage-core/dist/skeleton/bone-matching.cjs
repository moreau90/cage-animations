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

// src/skeleton/bone-matching.ts
var bone_matching_exports = {};
__export(bone_matching_exports, {
  matchFBXBone: () => matchFBXBone,
  matchFBXBoneToRegion: () => matchFBXBoneToRegion
});
module.exports = __toCommonJS(bone_matching_exports);

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
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  matchFBXBone,
  matchFBXBoneToRegion
});
