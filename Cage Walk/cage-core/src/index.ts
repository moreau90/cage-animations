// Types
export type { Vec3, Quat, Mat3, JointMap, MeshData, BoundsData, TriRef, SlicePoint, BoundaryLine, RigidTransform, BoneWeightEntry, FBXSkinData, DualQuat } from './types/index.js';

// Math - Vec3
export { clamp, vsub, vadd, vdot, vlen, vscale, vcross, vnorm } from './math/vec3.js';

// Math - Mat3
export { mat3_create, mat3_identity, mat3_rotX, mat3_rotY, mat3_rotZ, mat3_rotAxis, mat3_mul, mat3_transpose, mat3_det, mat3_mulVec3, mat3_orthonormalize } from './math/mat3.js';

// Math - Quaternion
export { quat_mul, quat_conjugate, quat_slerp, swingTwistDecompose, quat_rotate_vec, shortest_arc_quat, extractQuatTwist, mat3_to_quat, quat_to_mat3 } from './math/quat.js';

// Math - SVD
export { symmetricEigen3x3, svd3x3 } from './math/svd.js';
export type { EigenResult, SVDResult } from './math/svd.js';

// Math - Dual Quaternion
export { rigid_to_dq, dq_apply, quat_rotateVec3 } from './math/dualquat.js';

// Math - Kabsch
export { computeRigidTransformKabsch } from './math/kabsch.js';

// Math - Rotation
export { rotateAroundAxis, applyDeltaRotation, computeIKRotMatrix } from './math/rotation.js';

// Math - Interpolation
export { smoothArray, catmullRomVec3 } from './math/interpolation.js';

// Types - new
export type { BoneMapEntry, JointKeyframeData, JointPosData, FKChildrenMap, BoneSegment } from './types/index.js';

// Skeleton - Constants
export { FBX_BONE_MAP, FK_CHILDREN, BONE_SEGMENTS, JOINT_PRIMARY_CHILD } from './skeleton/constants.js';

// Skeleton - Bone Matching
export { matchFBXBone, matchFBXBoneToRegion } from './skeleton/bone-matching.js';

// Skeleton - FK
export { computeFKChain, computeJointRotationMatrices, computeGroundCorrection } from './skeleton/fk.js';

// Skeleton - Twist
export { extractBoneTwist, extractPositionTwist, canonicalBendRef } from './skeleton/twist.js';

// Animation - Keyframe
export { getJointPositionAtTime, getJointQuaternionAtTime, getRootPositionAtTime, getJointTwistAtTime } from './animation/keyframe.js';

// Geometry - Cross Section
export { sliceMeshAtX, filterHandTriangles } from './geometry/cross-section.js';

// Geometry - Regression
export { fitLineZD, evalBoundaryLine } from './geometry/regression.js';
export type { DZPoint } from './geometry/regression.js';

// Geometry - Boundary Extrapolation
export { findZRegions, matchRegionsToFingers, extrapolateBoundaries } from './geometry/boundary-extrapolation.js';
export type { ZRegion, BoundaryExtrapolationResult } from './geometry/boundary-extrapolation.js';

// Geometry - Hip Detection
export { detectHipGeometry } from './geometry/hip-detection.js';
export type { HipGeometry } from './geometry/hip-detection.js';

// Geometry - Toe Detection
export { autoDetectToe } from './geometry/toe-detection.js';
export type { ToeDetectionResult } from './geometry/toe-detection.js';

// Weights - Barycentric Transfer
export { computeBarycentricWeightTransfer } from './weights/barycentric-transfer.js';
export type { WeightTransferSource, WeightTransferResult, FlatBoneWeights } from './weights/barycentric-transfer.js';

// Weights - Sharpening
export { sharpenMidSegmentWeights, SHARPEN_EXCLUSION_PAIRS } from './weights/sharpening.js';
export type { SharpenStats, JointBoneMap } from './weights/sharpening.js';

// Weights - Per-Bone Matrices
export { computePerBoneMatrices } from './weights/per-bone-matrices.js';
export type { BoneTransform } from './weights/per-bone-matrices.js';

// Weights - LBS
export { applyPerBoneLBS } from './weights/lbs.js';

// Weights - DQS
export { boneTransformsToDualQuats, applyPerBoneDQS } from './weights/dqs.js';

// Geometry - Plane Slice
export { sliceMeshAtPlane } from './geometry/plane-slice.js';
export type { PlaneSlicePoint, PlaneSliceResult, PlaneSliceOptions } from './geometry/plane-slice.js';

// Geometry - Mesh Adjacency
export { buildMeshAdjacency } from './geometry/mesh-adjacency.js';
export type { FullMeshAdjacency } from './geometry/mesh-adjacency.js';

// QA - Skeleton
export { compareSkeletons, checkSymmetry } from './qa/skeleton-qa.js';
export type { SkeletonQAReport, BoneLengthComparison, JointPositionComparison } from './qa/skeleton-qa.js';

// QA - Deformation
export { computeRigidityStats, computeStrain, computeCircumference, CIRCUMFERENCE_SEGMENTS } from './qa/deformation-qa.js';
export type { RigiditySegStats, StrainResult, MeshAdjacency, CircumferenceResult, CircumferenceSegment } from './qa/deformation-qa.js';
