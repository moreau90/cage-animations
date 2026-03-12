/**
 * Keyframe Converter
 *
 * Converts Euler-angle keyframe files (leg_keyframes.json, arm_keyframes.json,
 * spine_keyframes.json) into JointPosData format with quaternions, suitable
 * for the FK/LBS processing pipeline.
 *
 * Euler angles are in degrees (x_rot, y_rot, z_rot) — absolute from T-pose.
 * Conversion: mat3_rotZ * mat3_rotY * mat3_rotX → mat3_to_quat
 */
import type { Vec3, Quat, JointPosData, JointKeyframeData } from 'cage-core';
import { mat3_rotX, mat3_rotY, mat3_rotZ, mat3_mul, mat3_to_quat } from 'cage-core';

const DEG2RAD = Math.PI / 180;

/** A single joint entry in the Euler keyframe files */
interface EulerJointData {
  times: number[];
  x_rot: number[];
  y_rot: number[];
  z_rot: number[];
  x_trans?: number[];
  y_trans?: number[];
  z_trans?: number[];
}

/** Structure of leg/arm/spine keyframe files on disk */
interface EulerKeyframeFile {
  duration: number;
  frame_count?: number;
  keyframes: { [jointName: string]: EulerJointData };
}

/** Structure of placed_joints.json (P section) */
interface PlacedJointsFile {
  P?: { [name: string]: { P: number[] } | number[] };
  primary?: { [name: string]: number[] };
}

/** Joint name mapping from keyframe file names to internal names */
const JOINT_NAME_MAP: Record<string, string> = {
  spine: 'spine_joint',
  spine1: 'spine1_joint',
  spine2: 'spine2_joint',
};

function mapJointName(name: string): string {
  return JOINT_NAME_MAP[name] ?? name;
}

/**
 * Convert Euler angles (degrees) to a quaternion via ZYX rotation order.
 * R = Rz * Ry * Rx
 */
function eulerToQuat(xDeg: number, yDeg: number, zDeg: number): Quat {
  const rx = mat3_rotX(xDeg * DEG2RAD);
  const ry = mat3_rotY(yDeg * DEG2RAD);
  const rz = mat3_rotZ(zDeg * DEG2RAD);
  const rzy = mat3_mul(rz, ry);
  const rzyx = mat3_mul(rzy, rx);
  return mat3_to_quat(rzyx);
}

/**
 * Convert Euler-angle keyframe files into JointPosData format.
 *
 * @param legKf - leg_keyframes.json content
 * @param armKf - arm_keyframes.json content
 * @param spineKf - spine_keyframes.json content
 * @param restJoints - placed_joints.json content (for rest pose positions)
 * @returns JointPosData with quaternion rotations
 */
export function convertEulerKeyframes(
  legKf: EulerKeyframeFile,
  armKf: EulerKeyframeFile,
  spineKf: EulerKeyframeFile,
  restJoints: PlacedJointsFile,
): JointPosData {
  const duration = spineKf.duration ?? legKf.duration ?? armKf.duration;
  const allKeyframes = {
    ...spineKf.keyframes,
    ...legKf.keyframes,
    ...armKf.keyframes,
  };

  // Extract rest pose positions from placed_joints.json
  const restPose: { [name: string]: Vec3 } = {};
  if (restJoints.P) {
    for (const [name, entry] of Object.entries(restJoints.P)) {
      if (Array.isArray(entry) && entry.length >= 3) {
        restPose[name] = [entry[0], entry[1], entry[2]];
      } else if (entry && typeof entry === 'object' && 'P' in entry) {
        const p = (entry as { P: number[] }).P;
        if (p && p.length >= 3) {
          restPose[name] = [p[0], p[1], p[2]];
        }
      }
    }
  }
  if (restJoints.primary) {
    for (const [name, pos] of Object.entries(restJoints.primary)) {
      if (pos && pos.length >= 3 && !restPose[name]) {
        restPose[name] = [pos[0], pos[1], pos[2]];
      }
    }
  }

  // Build JointPosData
  const joints: { [name: string]: JointKeyframeData } = {};
  const restQuats: { [name: string]: Quat } = {};
  let rootPositions: Vec3[] = [];

  // Determine FPS from first joint's time spacing
  const firstJoint = Object.values(allKeyframes)[0];
  const fps = firstJoint && firstJoint.times.length >= 2
    ? Math.round(1 / (firstJoint.times[1] - firstJoint.times[0]))
    : 30;

  for (const [rawName, euler] of Object.entries(allKeyframes)) {
    const name = mapJointName(rawName);
    const nFrames = euler.times.length;

    // Convert rotations to quaternions
    const quaternions: Quat[] = [];
    const positions: Vec3[] = [];

    for (let i = 0; i < nFrames; i++) {
      const q = eulerToQuat(euler.x_rot[i], euler.y_rot[i], euler.z_rot[i]);
      quaternions.push(q);

      // Positions: use translation if available (hips), otherwise rest pose
      if (euler.x_trans && euler.y_trans && euler.z_trans) {
        positions.push([euler.x_trans[i], euler.y_trans[i], euler.z_trans[i]]);
      } else {
        positions.push(restPose[name] ?? [0, 0, 0]);
      }
    }

    joints[name] = {
      times: euler.times,
      positions,
      quaternions,
    };

    // Rest quaternions are identity (Euler angles are absolute from T-pose)
    restQuats[name] = [0, 0, 0, 1];

    // Extract root positions from hips
    if (rawName === 'hips' && euler.x_trans && euler.y_trans && euler.z_trans) {
      rootPositions = euler.times.map((_, i) => [
        euler.x_trans![i],
        euler.y_trans![i],
        euler.z_trans![i],
      ] as Vec3);
    }
  }

  return {
    duration,
    fps,
    joints,
    rest_pose: restPose,
    rest_quats: restQuats,
    root_positions: rootPositions,
  };
}

// ─── CLI mode ───
async function main() {
  const args = process.argv.slice(2);

  if (args.length < 1 || args.includes('--help')) {
    console.log(`Usage: keyframe-converter --leg <path> --arm <path> --spine <path> --joints <path> --out <path>

Converts Euler-angle keyframe files to JointPosData format.

Options:
  --leg <path>      leg_keyframes.json
  --arm <path>      arm_keyframes.json
  --spine <path>    spine_keyframes.json
  --joints <path>   placed_joints.json (for rest pose)
  --out <path>      Output path (default: keyframes_converted.json)
`);
    process.exit(0);
  }

  const { default: fs } = await import('node:fs/promises');
  const { default: path } = await import('node:path');

  let legPath = '', armPath = '', spinePath = '', jointsPath = '', outPath = 'keyframes_converted.json';

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--leg': legPath = path.resolve(args[++i]); break;
      case '--arm': armPath = path.resolve(args[++i]); break;
      case '--spine': spinePath = path.resolve(args[++i]); break;
      case '--joints': jointsPath = path.resolve(args[++i]); break;
      case '--out': outPath = path.resolve(args[++i]); break;
    }
  }

  if (!legPath || !armPath || !spinePath || !jointsPath) {
    console.error('Error: --leg, --arm, --spine, and --joints are all required');
    process.exit(1);
  }

  const [legKf, armKf, spineKf, restJoints] = await Promise.all([
    fs.readFile(legPath, 'utf-8').then(JSON.parse),
    fs.readFile(armPath, 'utf-8').then(JSON.parse),
    fs.readFile(spinePath, 'utf-8').then(JSON.parse),
    fs.readFile(jointsPath, 'utf-8').then(JSON.parse),
  ]);

  const result = convertEulerKeyframes(legKf, armKf, spineKf, restJoints);

  await fs.writeFile(outPath, JSON.stringify(result, null, 2));
  console.log(`Converted ${Object.keys(result.joints).length} joints, ${result.duration}s duration, ${result.fps} fps`);
  console.log(`Written to: ${outPath}`);
}

// Run CLI if executed directly
const isMainModule = process.argv[1]?.includes('keyframe-converter');
if (isMainModule) {
  main().catch(err => {
    console.error('Conversion failed:', err);
    process.exit(1);
  });
}
