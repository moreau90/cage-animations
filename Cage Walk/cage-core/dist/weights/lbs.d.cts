import { BoneTransform } from './per-bone-matrices.cjs';
import '../types/index.cjs';

/**
 * Per-bone Linear Blend Skinning (LBS).
 * Standard weighted linear combination of per-bone rigid transforms.
 */

/**
 * Apply per-bone Linear Blend Skinning to a mesh.
 *
 * For each vertex, computes the weighted sum of bone transforms applied to the rest position,
 * then blends with rest position by alpha.
 *
 * @param restPos - Rest-pose positions [x,y,z,...] (Float32Array)
 * @param outPos - Output positions [x,y,z,...] (Float32Array, written in place)
 * @param boneWeights - Per-vertex flat weights: [boneIdx0, w0, boneIdx1, w1, ...] (or null)
 * @param boneTransforms - Per-bone transforms (indexed by bone index)
 * @param alpha - Blend factor (0 = rest pose, 1 = fully deformed)
 */
declare function applyPerBoneLBS(restPos: Float32Array, outPos: Float32Array, boneWeights: (number[] | null)[], boneTransforms: (BoneTransform | null)[], alpha: number): void;

export { applyPerBoneLBS };
