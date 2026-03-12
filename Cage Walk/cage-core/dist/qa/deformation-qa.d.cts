import { Vec3, BoneSegment } from '../types/index.cjs';
import { BoneTransform } from '../weights/per-bone-matrices.cjs';

/**
 * Deformation QA: rigidity error, strain analysis, circumference diagnostics.
 * Pure functions for measuring deformation quality.
 */

/** Per-segment rigidity statistics */
interface RigiditySegStats {
    segKey: string;
    label: string;
    parent: string;
    child: string;
    meanError: number;
    maxError: number;
    meanBlend: number;
    highErrCount: number;
    count: number;
    meanDistMm: number;
    maxDistMm: number;
    tag: 'GOOD' | 'OK' | 'TRANSFORM?' | 'WEIGHTS' | 'SHIFT';
}
/**
 * Compute per-segment rigidity error: how far LBS result deviates from
 * the dominant bone's rigid transform.
 *
 * @param boneTransforms - Per-bone {R, t} transforms
 * @param deformedPos - Deformed mesh positions [x,y,z,...] (after LBS + alpha)
 * @param meshRestPos - Rest-pose positions [x,y,z,...]
 * @param meshBoneWeights - Per-vertex flat weights [boneIdx, w, ...]
 * @param joints - Rest-pose joint positions
 * @param boneNameToIdx - Maps bone index → joint name
 * @param jointPrimaryChild - Maps joint → its primary child
 * @param boneSegments - [parent, child] pairs
 * @param alpha - Alpha blend factor used in LBS
 * @returns Array of per-segment rigidity stats
 */
declare function computeRigidityStats(boneTransforms: (BoneTransform | null)[], deformedPos: Float32Array, meshRestPos: Float32Array, meshBoneWeights: (number[] | null)[], joints: {
    [name: string]: Vec3 | undefined;
}, boneNameToIdx: {
    [boneIdx: number]: string;
}, jointPrimaryChild: {
    [name: string]: string;
}, boneSegments: BoneSegment[], alpha: number): RigiditySegStats[];
/** Edge adjacency data for strain computation */
interface MeshAdjacency {
    edgeA: Uint32Array;
    edgeB: Uint32Array;
    edgeRestLen: Float32Array;
    nEdges: number;
    nVerts: number;
}
/** Per-vertex strain result */
interface StrainResult {
    /** Per-vertex worst stretch ratio (1.0 = perfect) */
    worstStrain: Float32Array;
    /** Number of vertices with >30% strain */
    highStrainCount: number;
    /** Average deviation from rest length (fraction) */
    avgDeviation: number;
    nVerts: number;
    nEdges: number;
}
/**
 * Compute per-vertex edge strain: how much each edge deviates from rest length.
 *
 * @param deformedPos - Deformed mesh positions [x,y,z,...]
 * @param adjacency - Mesh edge adjacency data
 * @returns Strain analysis result
 */
declare function computeStrain(deformedPos: Float32Array, adjacency: MeshAdjacency): StrainResult;
/** Segment definition for circumference analysis */
interface CircumferenceSegment {
    parent: string;
    child: string;
    label: string;
}
/** Standard limb segments for circumference analysis */
declare const CIRCUMFERENCE_SEGMENTS: CircumferenceSegment[];
/** Per-segment circumference result */
interface CircumferenceResult {
    label: string;
    parent: string;
    child: string;
    mean: number;
    p5: number;
    p50: number;
    p95: number;
    flattenScore: number;
    count: number;
    tag: 'OK' | 'COMPRESS' | 'COLLAPSE' | 'BULGE';
}
/**
 * Compute circumference (radial cross-section ratio r'/r) for limb segments.
 *
 * Measures how the radial distance from the bone axis changes under deformation.
 * Detects collapse (negative volume), compression, bulging, and flattening.
 *
 * @param deformedPos - Deformed mesh positions [x,y,z,...]
 * @param meshRestPos - Rest-pose positions [x,y,z,...]
 * @param meshBoneWeights - Per-vertex flat weights [boneIdx, w, ...]
 * @param restJoints - Rest-pose joint positions
 * @param fkJoints - Current FK joint positions
 * @param boneNameToIdx - Maps bone index → joint name
 * @param groundCorrection - Y offset for ground correction
 * @param segments - Segment definitions (default: CIRCUMFERENCE_SEGMENTS)
 * @returns Array of per-segment circumference results
 */
declare function computeCircumference(deformedPos: Float32Array, meshRestPos: Float32Array, meshBoneWeights: (number[] | null)[], restJoints: {
    [name: string]: Vec3 | undefined;
}, fkJoints: {
    [name: string]: Vec3 | undefined;
}, boneNameToIdx: {
    [boneIdx: number]: string;
}, groundCorrection: number, segments?: CircumferenceSegment[]): CircumferenceResult[];

export { CIRCUMFERENCE_SEGMENTS, type CircumferenceResult, type CircumferenceSegment, type MeshAdjacency, type RigiditySegStats, type StrainResult, computeCircumference, computeRigidityStats, computeStrain };
