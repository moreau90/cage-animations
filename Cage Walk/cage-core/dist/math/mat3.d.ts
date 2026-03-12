import { Mat3, Vec3 } from '../types/index.js';

declare function mat3_create(): Float64Array;
declare function mat3_identity(): Float64Array;
declare function mat3_rotX(angle: number): Float64Array;
declare function mat3_rotY(angle: number): Float64Array;
declare function mat3_rotZ(angle: number): Float64Array;
/** Rotation matrix around an arbitrary axis (Rodrigues formula) */
declare function mat3_rotAxis(ax: number, ay: number, az: number, angle: number): number[];
declare function mat3_mul(A: Mat3, B: Mat3): Float64Array;
declare function mat3_transpose(A: Mat3): Float64Array;
declare function mat3_det(A: Mat3): number;
declare function mat3_mulVec3(M: Mat3, x: number, y: number, z: number): Vec3;
/** Gram-Schmidt orthonormalize a 3x3 matrix to ensure proper rotation */
declare function mat3_orthonormalize(M: Mat3): Float64Array;

export { mat3_create, mat3_det, mat3_identity, mat3_mul, mat3_mulVec3, mat3_orthonormalize, mat3_rotAxis, mat3_rotX, mat3_rotY, mat3_rotZ, mat3_transpose };
