import { Vec3 } from '../types/index.cjs';

declare function clamp(x: number, a: number, b: number): number;
declare function vsub(a: Vec3, b: Vec3): Vec3;
declare function vadd(a: Vec3, b: Vec3): Vec3;
declare function vdot(a: Vec3, b: Vec3): number;
declare function vlen(a: Vec3): number;
declare function vscale(a: Vec3, s: number): Vec3;
declare function vcross(a: Vec3, b: Vec3): Vec3;
declare function vnorm(a: Vec3): Vec3;

export { clamp, vadd, vcross, vdot, vlen, vnorm, vscale, vsub };
