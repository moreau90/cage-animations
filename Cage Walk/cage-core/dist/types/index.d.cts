/** 3D vector as [x, y, z] */
type Vec3 = [number, number, number];
/** Quaternion as [x, y, z, w] (Hamilton convention) */
type Quat = [number, number, number, number];
/** 3x3 matrix, row-major, as Float64Array(9) or number[] */
type Mat3 = Float64Array | number[];
/** Named joint positions map */
interface JointMap {
    [jointName: string]: Vec3 | undefined;
}
/** Mesh rest-pose data */
interface MeshData {
    /** Interleaved positions [x,y,z, x,y,z, ...] */
    restPositions: Float32Array;
    /** Triangle indices (every 3 = one tri) */
    index: Uint32Array;
    nVerts: number;
}
/** Axis-aligned bounding box */
interface BoundsData {
    min: Vec3;
    max: Vec3;
    height: number;
}
/** Triangle reference with pre-multiplied vertex offsets */
interface TriRef {
    a3: number;
    b3: number;
    c3: number;
}
/** Point on a cross-section slice */
interface SlicePoint {
    y: number;
    z: number;
}
/** Linear regression result: z = slope * d + intercept */
interface BoundaryLine {
    slope: number;
    intercept: number;
}
/** Rigid transform: rotation + translation */
interface RigidTransform {
    R: Float64Array;
    t: Vec3;
    cRest: Vec3;
    cCurr: Vec3;
}
/** Bone weight entry for one vertex */
interface BoneWeightEntry {
    boneIdx: number;
    weight: number;
}
/** FBX extracted skin data */
interface FBXSkinData {
    positions: Float32Array;
    boneWeights: BoneWeightEntry[][];
    boneNames: string[];
    boneCount: number;
    index?: Uint32Array;
    nVerts: number;
}
/** Dual quaternion */
interface DualQuat {
    qr: Quat;
    qd: Quat;
}
/** FBX bone map entry: maps FBX suffix to internal joint name */
interface BoneMapEntry {
    suffix: string;
    name: string;
    alt?: string;
}
/** Per-joint keyframe data */
interface JointKeyframeData {
    times: number[];
    positions: Vec3[];
    quaternions?: Quat[];
}
/** Full animation keyframe dataset extracted from FBX */
interface JointPosData {
    duration: number;
    fps: number;
    joints: {
        [jointName: string]: JointKeyframeData;
    };
    rest_pose: {
        [jointName: string]: Vec3;
    };
    rest_quats: {
        [jointName: string]: Quat;
    };
    root_positions: Vec3[];
}
/** FK children map: parent → children for forward kinematics traversal */
interface FKChildrenMap {
    [parent: string]: string[];
}
/** Bone segment: [parent, child] joint name pair */
type BoneSegment = [string, string];

export type { BoneMapEntry, BoneSegment, BoneWeightEntry, BoundaryLine, BoundsData, DualQuat, FBXSkinData, FKChildrenMap, JointKeyframeData, JointMap, JointPosData, Mat3, MeshData, Quat, RigidTransform, SlicePoint, TriRef, Vec3 };
