import { DualQuat } from '../types/index.js';
import { BoneTransform } from './per-bone-matrices.js';

/**
 * Per-bone Dual Quaternion Skinning (DQS).
 * Blends dual quaternions with hemisphere correction, then applies to rest positions.
 */

/**
 * Convert bone transforms to dual quaternions.
 *
 * @param boneTransforms - Per-bone {R, t} transforms
 * @returns Array of dual quaternions (indexed by bone index)
 */
declare function boneTransformsToDualQuats(boneTransforms: (BoneTransform | null)[]): (DualQuat | null)[];
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
declare function applyPerBoneDQS(restPos: Float32Array, outPos: Float32Array, boneWeights: (number[] | null)[], boneDQs: (DualQuat | null)[], alpha: number): void;

export { applyPerBoneDQS, boneTransformsToDualQuats };
