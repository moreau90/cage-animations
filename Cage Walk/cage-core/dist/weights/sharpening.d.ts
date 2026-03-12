import { Vec3, BoneSegment } from '../types/index.js';

/**
 * Mid-segment weight sharpening.
 * Boosts dominant bone weight in mid-segment areas to reduce blending artifacts.
 */

/** Joint-to-bone-index mapping */
interface JointBoneMap {
    [boneIdx: number]: string;
}
/** Joint pairs that should NEVER be sharpened (real anatomical blend zones) */
declare const SHARPEN_EXCLUSION_PAIRS: [string, string][];
/** Result statistics from sharpening */
interface SharpenStats {
    sharpenedUpper: number;
    sharpenedLower: number;
    excluded: number;
    total: number;
}
/**
 * Sharpen mid-segment bone weights by boosting the dominant bone and reducing others.
 *
 * For each vertex in the mid-segment band (avoiding joints), increases the dominant
 * bone's weight toward 1.0 using a smoothstep profile. Excludes anatomical blend zones
 * (e.g., hip-to-knee transitions) where blending is desired.
 *
 * @param weights - Per-vertex flat bone weights: [boneIdx0, w0, ...] (MUTATED in place)
 * @param meshRestPos - Mesh rest positions [x,y,z,...]
 * @param joints - Named joint positions
 * @param boneNameToIdx - Maps bone index → joint name
 * @param jointPrimaryChild - Maps joint name → its primary child joint
 * @param boneSegments - Array of [parent, child] bone segment definitions
 * @param strengthUpper - Sharpening strength for upper body [0..1]
 * @param strengthLower - Sharpening strength for lower body [0..1]
 * @returns Statistics about sharpening, or null if inputs missing
 */
declare function sharpenMidSegmentWeights(weights: (number[] | null)[], meshRestPos: Float32Array, joints: {
    [name: string]: Vec3 | undefined;
}, boneNameToIdx: JointBoneMap, jointPrimaryChild: {
    [name: string]: string;
}, boneSegments: BoneSegment[], strengthUpper: number, strengthLower: number): SharpenStats | null;

export { type JointBoneMap, SHARPEN_EXCLUSION_PAIRS, type SharpenStats, sharpenMidSegmentWeights };
