/** Result of hip geometry auto-detection */
export interface HipGeometry {
  hipsCenterY: number;
  hipJointY: number;
  hipSpreadHalf: number;
  crotchY: number;
  hipShelfY: number;
}

/**
 * Auto-detect hip geometry from mesh cross-sections.
 * Scans mesh to find crotch (center density peak) and hip shelf (widest above crotch),
 * then derives hip positions using anatomical proportions.
 *
 * @param roughY - Rough Y estimate (groin click or stored position)
 * @param meshH - Total mesh height
 * @param meshRestPos - Interleaved rest positions [x,y,z, x,y,z, ...], or null for fallback
 */
export function detectHipGeometry(
  roughY: number,
  meshH: number,
  meshRestPos: Float32Array | null,
): HipGeometry {
  if (!meshRestPos) {
    // Fallback: proportional from rough Y
    const hipsCenterY = roughY + meshH * 0.096;
    const hipJointY = hipsCenterY - meshH * 0.0316;
    return {
      hipsCenterY,
      hipJointY,
      hipSpreadHalf: meshH * 0.0445,
      crotchY: roughY,
      hipShelfY: hipsCenterY + meshH * 0.014,
    };
  }

  const nMesh = meshRestPos.length / 3;
  const scanStep = meshH * 0.003;

  // 1. Find crotch Y via center vertex density peak
  const crotchScanMin = roughY - meshH * 0.10;
  const crotchScanMax = roughY + meshH * 0.05;
  let maxCenterDensity = 0;
  let crotchY = roughY;

  for (let y = crotchScanMin; y < crotchScanMax; y += scanStep) {
    let centerCount = 0;
    for (let i = 0; i < nMesh; i++) {
      if (Math.abs(meshRestPos[i * 3 + 1] - y) < scanStep && Math.abs(meshRestPos[i * 3]) < 0.015) {
        centerCount++;
      }
    }
    if (centerCount > maxCenterDensity) {
      maxCenterDensity = centerCount;
      crotchY = y;
    }
  }

  // 2. Find hip shelf (widest cross-section above crotch)
  let maxWidth = 0, hipShelfY = crotchY;
  for (let y = crotchY + meshH * 0.02; y < crotchY + meshH * 0.15; y += scanStep) {
    let minX = Infinity, maxX = -Infinity;
    for (let i = 0; i < nMesh; i++) {
      if (Math.abs(meshRestPos[i * 3 + 1] - y) < scanStep) {
        minX = Math.min(minX, meshRestPos[i * 3]);
        maxX = Math.max(maxX, meshRestPos[i * 3]);
      }
    }
    const w = maxX - minX;
    if (w > maxWidth) { maxWidth = w; hipShelfY = y; }
  }

  // 3. Derive using anatomical proportions (calibrated from Mixamo)
  const hipsCenterY = hipShelfY - meshH * 0.014;
  const hipJointY = hipsCenterY - meshH * 0.0316;
  const hipSpreadHalf = meshH * 0.0445;

  return { hipsCenterY, hipJointY, hipSpreadHalf, crotchY, hipShelfY };
}
