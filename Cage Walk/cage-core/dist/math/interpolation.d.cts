import { Vec3 } from '../types/index.cjs';

/** Smooth a cyclic array with 3-point weighted average [0.25, 0.5, 0.25] */
declare function smoothArray(arr: number[], passes?: number): number[];
/** Catmull-Rom interpolation between 4 Vec3 control points at parameter t ∈ [0,1] */
declare function catmullRomVec3(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: number): Vec3;

export { catmullRomVec3, smoothArray };
