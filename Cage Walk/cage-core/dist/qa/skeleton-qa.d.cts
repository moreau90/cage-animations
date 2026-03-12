import { Vec3 } from '../types/index.cjs';

/**
 * Skeleton QA: bone length comparison, joint position comparison, symmetry checks.
 * Pure functions for comparing skeleton proportions against reference data.
 */

/** Result of comparing a single bone length */
interface BoneLengthComparison {
    label: string;
    parent: string;
    child: string;
    ourLength: number;
    refLength: number;
    origLength: number;
    ratio: number;
    isMismatch: boolean;
}
/** Result of comparing a single joint position */
interface JointPositionComparison {
    jointName: string;
    ourPos: Vec3 | null;
    refPos: Vec3 | null;
    diffMm: number;
    status: 'OK' | 'MISMATCH' | 'MISSING_OURS' | 'MISSING_REF';
}
/** Full skeleton QA report */
interface SkeletonQAReport {
    jointComparisons: JointPositionComparison[];
    boneLengthComparisons: BoneLengthComparison[];
    maxJointDiffMm: number;
    maxJointDiffName: string;
    totalLegOurs: number;
    totalLegRef: number;
    totalArmOurs: number;
    totalArmRef: number;
}
/**
 * Compare skeleton joint positions and bone lengths against a reference skeleton.
 *
 * @param ourJoints - Our joint positions (world space, meters)
 * @param refRestPose - Reference rest pose (pelvis-relative offsets, meters). Can be null.
 * @param anchor - Pelvis/hips world position to anchor reference (meters)
 * @param origJoints - Pre-override joint positions (for ratio comparison). Can be null.
 * @returns Full QA report with joint and bone comparisons
 */
declare function compareSkeletons(ourJoints: {
    [name: string]: Vec3 | undefined;
}, refRestPose: {
    [name: string]: Vec3 | undefined;
} | null, anchor: Vec3, origJoints: {
    [name: string]: Vec3 | undefined;
} | null): SkeletonQAReport;
/**
 * Check bone length symmetry between left and right sides.
 *
 * @param joints - Joint positions
 * @returns Array of asymmetry results (label, leftMm, rightMm, ratio, isMismatch)
 */
declare function checkSymmetry(joints: {
    [name: string]: Vec3 | undefined;
}): {
    label: string;
    leftMm: number;
    rightMm: number;
    ratio: number;
    isMismatch: boolean;
}[];

export { type BoneLengthComparison, type JointPositionComparison, type SkeletonQAReport, checkSymmetry, compareSkeletons };
