// src/math/interpolation.ts
function smoothArray(arr, passes = 3) {
  const n = arr.length;
  let result = arr.slice();
  for (let p = 0; p < passes; p++) {
    const tmp = result.slice();
    for (let i = 0; i < n; i++) {
      result[i] = 0.25 * tmp[(i - 1 + n) % n] + 0.5 * tmp[i] + 0.25 * tmp[(i + 1) % n];
    }
  }
  return result;
}
function catmullRomVec3(p0, p1, p2, p3, t) {
  const t2 = t * t, t3 = t2 * t;
  const out = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    out[i] = 0.5 * (2 * p1[i] + (-p0[i] + p2[i]) * t + (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * t2 + (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * t3);
  }
  return out;
}
export {
  catmullRomVec3,
  smoothArray
};
