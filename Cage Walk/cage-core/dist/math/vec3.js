// src/math/vec3.ts
function clamp(x, a, b) {
  return Math.max(a, Math.min(b, x));
}
function vsub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function vadd(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
function vdot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function vlen(a) {
  return Math.sqrt(vdot(a, a));
}
function vscale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}
function vcross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}
function vnorm(a) {
  const L = vlen(a);
  return L > 1e-12 ? [a[0] / L, a[1] / L, a[2] / L] : [1, 0, 0];
}
export {
  clamp,
  vadd,
  vcross,
  vdot,
  vlen,
  vnorm,
  vscale,
  vsub
};
