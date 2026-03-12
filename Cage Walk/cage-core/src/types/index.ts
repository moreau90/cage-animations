/** 3D vector as [x, y, z] */
export type Vec3 = [number, number, number];

/** Quaternion as [x, y, z, w] (Hamilton convention) */
export type Quat = [number, number, number, number];

/** 3x3 matrix, row-major, as Float64Array(9) or number[] */
export type Mat3 = Float64Array | number[];

/** Named joint positions map */
export interface JointMap {
  [jointName: string]: Vec3 | undefined;
}

/** Mesh rest-pose data */
export interface MeshData {
  /** Interleaved positions [x,y,z, x,y,z, ...] */
  restPositions: Float32Array;
  /** Triangle indices (every 3 = one tri) */
  index: Uint32Array;
  nVerts: number;
}

/** Axis-aligned bounding box */
export interface BoundsData {
  min: Vec3;
  max: Vec3;
  height: number;
}

/** Triangle reference with pre-multiplied vertex offsets */
export interface TriRef {
  a3: number;
  b3: number;
  c3: number;
}

/** Point on a cross-section slice */
export interface SlicePoint {
  y: number;
  z: number;
}

/** Linear regression result: z = slope * d + intercept */
export interface BoundaryLine {
  slope: number;
  intercept: number;
}

/** Rigid transform: rotation + translation */
export interface RigidTransform {
  R: Float64Array;
  t: Vec3;
  cRest: Vec3;
  cCurr: Vec3;
}

/** Bone weight entry for one vertex */
export interface BoneWeightEntry {
  boneIdx: number;
  weight: number;
}

/** FBX extracted skin data */
export interface FBXSkinData {
  positions: Float32Array;
  boneWeights: BoneWeightEntry[][];
  boneNames: string[];
  boneCount: number;
  index?: Uint32Array;
  nVerts: number;
}

/** Dual quaternion */
export interface DualQuat {
  qr: Quat;
  qd: Quat;
}

/** FBX bone map entry: maps FBX suffix to internal joint name */
export interface BoneMapEntry {
  suffix: string;
  name: string;
  alt?: string;
}

/** Per-joint keyframe data */
export interface JointKeyframeData {
  times: number[];
  positions: Vec3[];
  quaternions?: Quat[];
}

/** Full animation keyframe dataset extracted from FBX */
export interface JointPosData {
  duration: number;
  fps: number;
  joints: { [jointName: string]: JointKeyframeData };
  rest_pose: { [jointName: string]: Vec3 };
  rest_quats: { [jointName: string]: Quat };
  root_positions: Vec3[];
}

/** FK children map: parent → children for forward kinematics traversal */
export interface FKChildrenMap {
  [parent: string]: string[];
}

/** Bone segment: [parent, child] joint name pair */
export type BoneSegment = [string, string];
