import { BoneMapEntry } from '../types/index.cjs';

/**
 * Match an FBX bone name to internal joint name by suffix matching.
 * Sorts by suffix length (longest first) to avoid partial matches.
 */
declare function matchFBXBone(boneName: string, boneMap?: BoneMapEntry[]): string | null;
/**
 * Map FBX bone name to region ID for skin weight purposes.
 * Maps finger bones to their controlling wrist/hand region.
 */
declare function matchFBXBoneToRegion(boneName: string, boneMap?: BoneMapEntry[]): string | null;

export { matchFBXBone, matchFBXBoneToRegion };
