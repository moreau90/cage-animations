import { BoneSegment, BoneMapEntry, FKChildrenMap } from '../types/index.cjs';

/** Maps FBX bone name suffixes to internal joint names (52 bones including all 40 finger bones) */
declare const FBX_BONE_MAP: BoneMapEntry[];
/** Full FK hierarchy including all finger chains */
declare const FK_CHILDREN: FKChildrenMap;
/** Bone segments for rigidity/bend diagnostics: [parent, child] */
declare const BONE_SEGMENTS: BoneSegment[];
/** Primary child for each joint — the bone segment that joint CONTROLS */
declare const JOINT_PRIMARY_CHILD: {
    [joint: string]: string;
};

export { BONE_SEGMENTS, FBX_BONE_MAP, FK_CHILDREN, JOINT_PRIMARY_CHILD };
