import { Vec3, Mat3 } from '../types/index.cjs';

/**
 * Extract twist around a bone axis by comparing actual vs expected bend plane normals.
 * Returns signed twist angle in radians.
 */
declare function extractBoneTwist(restBoneDir: Vec3, curBoneDir: Vec3, restChildDir: Vec3, curChildDir: Vec3): number;
/**
 * Position-based twist extraction from bend-axis comparison.
 * Compares actual joint bend axis with expected "no-twist" bend axis.
 * Returns signed twist angle in radians. Returns 0 if bend < ~6°.
 */
declare function extractPositionTwist(swingR: Mat3, targetBoneDir: Vec3, refPerp: Vec3, actualBendAxis: Vec3, bendAngle: number): number;
/**
 * Compute canonical reference perpendicular for a bone rest direction.
 * Orthogonalizes rawRef to restBoneDir to ensure perpendicularity.
 */
declare function canonicalBendRef(rawRef: Vec3, restBoneDir: Vec3): Vec3;

export { canonicalBendRef, extractBoneTwist, extractPositionTwist };
