import type { BoneMapEntry } from '../types/index.js';
import { FBX_BONE_MAP } from './constants.js';

/**
 * Match an FBX bone name to internal joint name by suffix matching.
 * Sorts by suffix length (longest first) to avoid partial matches.
 */
export function matchFBXBone(boneName: string, boneMap: BoneMapEntry[] = FBX_BONE_MAP): string | null {
  const lower = boneName.toLowerCase();
  const sorted = boneMap.slice().sort((a, b) => b.suffix.length - a.suffix.length);
  for (const entry of sorted) {
    if (lower.endsWith(entry.suffix.toLowerCase())) return entry.name;
    if (entry.alt && lower.endsWith(entry.alt.toLowerCase())) return entry.name;
  }
  return null;
}

/**
 * Map FBX bone name to region ID for skin weight purposes.
 * Maps finger bones to their controlling wrist/hand region.
 */
export function matchFBXBoneToRegion(boneName: string, boneMap: BoneMapEntry[] = FBX_BONE_MAP): string | null {
  const mapped = matchFBXBone(boneName, boneMap);
  if (mapped) return mapped;
  const lower = boneName.toLowerCase();
  if (lower.includes('lefthand')) return 'l_wrist';
  if (lower.includes('righthand')) return 'r_wrist';
  return null;
}
