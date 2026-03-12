/**
 * AI Worker — Single Claude agent that places skeleton joints inside a humanoid mesh.
 *
 * The AI reads mesh geometry (cage.json vertices, regions), looks at screenshots,
 * understands anatomy, and places joints at correct positions. No scores needed —
 * just geometry and anatomical knowledge.
 *
 * Runs via `claude -p` (Claude Code headless mode, uses Pro subscription).
 */
import { execFile, spawn } from 'node:child_process';
import path from 'node:path';
import pino from 'pino';

const log = pino({ name: 'ai-workers' });

const AI_TIMEOUT_MS = 600_000; // 10 min

// ─── Result type ───

export interface AIPlacementResult {
  jointChanges: number;
  reasoning: string;
  lesson: string;
  success: boolean;
}

// ─── Shared utilities ───

/** Build a clean env for spawning claude subprocesses (unset CLAUDECODE to allow nesting). */
function cleanEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  delete env.CLAUDECODE;
  return env;
}

/** Run claude -p with full tool access (agent mode). */
async function runClaudeAgent(
  prompt: string,
  cwd: string,
  timeoutMs: number,
  extraArgs: string[] = [],
): Promise<string | null> {
  return new Promise((resolve) => {
    const args = [
      '-p', '-',
      '--output-format', 'text',
      '--allowedTools', 'Read,Write,Edit,Bash,Glob,Grep,WebSearch,WebFetch,Agent,NotebookEdit',
      ...extraArgs,
    ];

    const child = spawn('claude', args, {
      cwd,
      timeout: timeoutMs,
      env: cleanEnv(),
      stdio: ['pipe', 'pipe', 'pipe'],
      shell: true,
    });

    let stdout = '';
    let stderr = '';
    let resolved = false;

    child.stdout.on('data', (data: Buffer) => { stdout += data.toString(); });
    child.stderr.on('data', (data: Buffer) => { stderr += data.toString(); });

    child.on('close', (code) => {
      if (resolved) return;
      resolved = true;
      if (code !== 0) {
        log.warn({ code, stderr: stderr.slice(0, 500) }, 'claude agent exited non-zero');
      }
      resolve(stdout.trim() || null);
    });

    child.on('error', (err) => {
      if (resolved) return;
      resolved = true;
      log.warn({ err: err.message }, 'claude agent spawn error');
      resolve(null);
    });

    child.stdin.write(prompt);
    child.stdin.end();

    setTimeout(() => {
      if (resolved) return;
      resolved = true;
      try { child.kill('SIGTERM'); } catch { /* ignore */ }
      log.warn('claude agent timed out');
      resolve(stdout.trim() || null);
    }, timeoutMs);
  });
}

/** Parse JSON from Claude's response (handles fences, bare JSON, etc.) */
function parseJSON<T>(text: string): T | null {
  try { return JSON.parse(text) as T; } catch { /* */ }

  const fence = text.match(/```(?:json)?\s*\n?([\s\S]*?)\n?\s*```/);
  if (fence) {
    try { return JSON.parse(fence[1]) as T; } catch { /* */ }
  }

  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start >= 0 && end > start) {
    try { return JSON.parse(text.slice(start, end + 1)) as T; } catch { /* */ }
  }

  return null;
}

/** Check if claude CLI is available. */
export async function isClaudeAvailable(): Promise<boolean> {
  return new Promise((resolve) => {
    execFile('claude', ['--version'], { timeout: 5000, env: cleanEnv(), shell: true }, (error) => {
      resolve(!error);
    });
  });
}

// ─── Single AI Worker ───

interface PlacementInput {
  /** Path to the placed_joints.json artifact to modify */
  jointsPath: string;
  /** Screenshot paths for visual inspection */
  screenshotPaths: string[];
  /** Previous round's screenshots (for before/after comparison) */
  previousScreenshotPaths?: string[];
  /** Round number (1-based) */
  round: number;
  /** Lessons from previous rounds */
  lessons: string[];
  /** Directories */
  cageWalkDir: string;
  pipelineDir: string;
  cageCoreDir: string;
}

/**
 * Single AI worker that places skeleton joints inside a humanoid mesh.
 *
 * The AI gets:
 * - Mesh geometry (cage.json vertices, regions.json body parts)
 * - Screenshots of the current state
 * - Current placed_joints.json
 * - Full tool access to write scripts, analyze geometry, etc.
 *
 * It figures out where joints should go based on anatomy and geometry,
 * writes scripts to compute positions from vertex data, and places them.
 */
export async function runJointPlacementAI(input: PlacementInput): Promise<AIPlacementResult | null> {
  const screenshotSection = input.screenshotPaths.length > 0
    ? `## Screenshots (READ THESE with the Read tool — they are PNG images)
Current state:
${input.screenshotPaths.map(p => path.resolve(p)).join('\n')}
${input.previousScreenshotPaths && input.previousScreenshotPaths.length > 0
    ? `\nPrevious round (compare before/after):
${input.previousScreenshotPaths.map(p => path.resolve(p)).join('\n')}`
    : ''}`
    : '';

  const prompt = `You are a 3D geometry engineer. Your job: place skeleton joints correctly inside a humanoid mesh.

You have the mesh geometry, screenshots, and full tool access. Figure out where each joint belongs anatomically and place it there.

## Your approach
1. READ the screenshots to visually assess the current joint placement
2. READ cage.json to get the mesh vertices (V = array of [x,y,z])
3. READ regions.json to know which vertices belong to which body part
4. READ the current placed_joints.json to see where joints are now
5. WRITE Node.js scripts to analyze the geometry:
   - Slice cross-sections at joint heights to find centroids
   - Trace finger tube centerlines
   - Find anatomical landmarks (ball of foot, posterior spine wall, etc.)
6. Place each joint at the geometrically correct position
7. Write the updated placed_joints.json

## Anatomical rules
- **SPINE**: Should be ~1/3 from the posterior (back) wall of the torso, NOT at the centroid. Slice the torso at each spine height, find the front and back walls, place spine at back_z + (front_z - back_z) * 0.33.
- **HIPS**: At the femoral head — lateral edge of pelvis where leg meets torso.
- **KNEES/ANKLES/ELBOWS/WRISTS**: At the cross-section centroid of the limb at that height.
- **FINGERS**: Trace each finger's tube of vertices from knuckle to tip. The joint chain should follow the centerline of that tube.
- **TOES**: Toe joint = ball of foot (metatarsal heads, where toes meet sole), NOT tip of toes.
- **SHOULDERS**: At the arm-torso junction (glenohumeral joint).
- **NECK**: Centered in the neck, slightly posterior.

## File paths
- Joints to modify: ${input.jointsPath}
  Format: { "P": { "joint_name": [x, y, z], ... }, "primary": {...}, "bone_lengths": {...}, "mesh_height": N }
  Only modify values in "P". Preserve all other sections.
- Mesh geometry: ${input.cageWalkDir}/cage.json — has V (vertices), F (faces), region IDs
- Body regions: ${input.cageWalkDir}/regions.json — which vertices belong to which body part
- Geometry library: ${input.cageCoreDir} — has sliceMeshAtPlane and other utilities

## Coordinate system
Y = up, X = left(-)/right(+), Z = forward(+)/back(-). Units: meters.

${screenshotSection}

${input.lessons.length > 0 ? `## Lessons from previous rounds\n${input.lessons.join('\n')}` : ''}

## Round ${input.round}
${input.round === 1 ? 'First round. Assess everything. Fix what you can compute positions for.' : 'Check what improved from last round. Fix remaining issues.'}

Fix every joint that is wrong. For each one, compute the correct position from the mesh geometry — do not guess.
Only modify the file at ${input.jointsPath}. Do NOT write to any other placed_joints.json.

When done, output this JSON:
{ "jointChanges": N, "reasoning": "what you did and why", "lesson": "what you learned for next round", "success": true }`;

  log.info({ round: input.round, jointsPath: input.jointsPath }, 'Running joint placement AI');

  const extraArgs = [
    '--add-dir', input.pipelineDir,
    '--add-dir', input.cageCoreDir,
  ];

  // Run in pipelineDir so it can't accidentally overwrite the working placed_joints.json
  const output = await runClaudeAgent(prompt, input.pipelineDir, AI_TIMEOUT_MS, extraArgs);
  if (!output) {
    log.warn('AI returned no output');
    return null;
  }

  const result = parseJSON<AIPlacementResult>(output);
  if (!result) {
    log.warn('AI output not parseable as JSON — changes may still have been applied');
    return {
      jointChanges: 0,
      reasoning: output.slice(-500),
      lesson: '',
      success: false,
    };
  }

  log.info({ jointChanges: result.jointChanges, success: result.success }, 'Joint placement AI done');
  return result;
}
