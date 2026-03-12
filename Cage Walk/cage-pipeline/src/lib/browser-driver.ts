/**
 * Browser Driver — Playwright wrapper for headless interaction with Cage Walk.
 *
 * Launches headless Chromium, navigates to bridge server, waits for mesh load,
 * can load FBX files, extract runtime data, and take screenshots.
 */
import { chromium, type Browser, type BrowserContext, type Page } from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
import pino from 'pino';

const log = pino({ name: 'browser-driver' });

export interface CameraAngle {
  name: string;
  azimuth: number;   // degrees horizontal rotation
  elevation: number; // degrees vertical rotation
  distance?: number; // optional camera distance
}

export const DEFAULT_ANGLES: CameraAngle[] = [
  // Full body views from each axis
  { name: 'front', azimuth: 0, elevation: 0 },
  { name: 'right_side', azimuth: 90, elevation: 0 },
  { name: 'back', azimuth: 180, elevation: 0 },
  { name: 'left_side', azimuth: -90, elevation: 0 },
  { name: 'top_down', azimuth: 0, elevation: 85 },
  { name: 'three_quarter', azimuth: 45, elevation: 20 },
  // Hand close-ups (top-down view to check finger centering)
  { name: 'hand_left_top', azimuth: -90, elevation: 70, distance: 0.25 },
  { name: 'hand_right_top', azimuth: 90, elevation: 70, distance: 0.25 },
  { name: 'hand_left_front', azimuth: -60, elevation: -10, distance: 0.3 },
  { name: 'hand_right_front', azimuth: 60, elevation: -10, distance: 0.3 },
];

export interface BrowserDriverOptions {
  bridgeUrl?: string;
  headless?: boolean;
  viewportWidth?: number;
  viewportHeight?: number;
}

export interface ExtractedRuntimeData {
  deformedPositions: number[];
  restPositions: number[];
  boneWeights: (number[] | null)[];
  boneTransforms: Array<{ R: number[]; t: number[] } | null>;
  adjacency: {
    edgeA: number[];
    edgeB: number[];
    edgeRestLen: number[];
    nEdges: number;
    nVerts: number;
  } | null;
  joints: { [name: string]: [number, number, number] };
  fkJoints: { [name: string]: [number, number, number] };
}

export class BrowserDriver {
  private browser: Browser | null = null;
  private context: BrowserContext | null = null;
  private page: Page | null = null;
  private bridgeUrl: string;
  private headless: boolean;
  private viewportWidth: number;
  private viewportHeight: number;

  constructor(opts: BrowserDriverOptions = {}) {
    this.bridgeUrl = opts.bridgeUrl ?? 'http://localhost:8765';
    this.headless = opts.headless ?? true;
    this.viewportWidth = opts.viewportWidth ?? 1280;
    this.viewportHeight = opts.viewportHeight ?? 960;
  }

  /** Launch browser and navigate to Cage Walk */
  async launch(): Promise<void> {
    log.info({ bridgeUrl: this.bridgeUrl, headless: this.headless }, 'Launching browser');

    this.browser = await chromium.launch({ headless: this.headless });
    this.context = await this.browser.newContext({
      viewport: { width: this.viewportWidth, height: this.viewportHeight },
    });
    this.page = await this.context.newPage();

    // Navigate and wait for mesh to load
    await this.page.goto(this.bridgeUrl + '/index.html', { waitUntil: 'domcontentloaded' });

    log.info('Waiting for mesh initialization...');
    await this.page.waitForFunction('window._initDone === true', {}, { timeout: 30000 });
    log.info('Mesh initialized');
  }

  /** Load an FBX file via file input simulation */
  async loadFBX(fbxPath: string): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');

    const resolvedPath = path.resolve(fbxPath);
    await fs.access(resolvedPath); // verify file exists

    log.info({ fbxPath: resolvedPath }, 'Loading FBX file');

    // Find the file input element for FBX loading
    const fileInput = await this.page.$('input[type="file"]');
    if (!fileInput) {
      throw new Error('No file input found on page');
    }

    await fileInput.setInputFiles(resolvedPath);

    // Wait for FBX to finish loading — bone transforms will be set
    await this.page.waitForFunction(
      '!!window._lastBoneTransforms && window._lastBoneTransforms.length > 0',
      {},
      { timeout: 30000 },
    );

    log.info('FBX loaded successfully');
  }

  /**
   * Set camera to a specific angle.
   * Uses OrbitControls to position the camera.
   */
  async setCameraAngle(angle: CameraAngle): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');

    await this.page.evaluate((a) => {
      const controls = (window as any)._controls;
      const camera = (window as any)._camera;
      if (!controls || !camera) return;

      const azRad = a.azimuth * Math.PI / 180;
      const elRad = a.elevation * Math.PI / 180;
      const dist = a.distance ?? controls.getDistance?.() ?? 2.0;

      // Compute camera position in spherical coordinates
      const x = dist * Math.cos(elRad) * Math.sin(azRad);
      const y = dist * Math.sin(elRad) + 1.0; // offset up to center on mesh
      const z = dist * Math.cos(elRad) * Math.cos(azRad);

      camera.position.set(x, y, z);
      controls.target.set(0, 1.0, 0); // center on mesh
      controls.update();
    }, angle);

    // Let the render loop catch up
    await this.page.waitForTimeout(200);
  }

  /** Take a screenshot and save to disk */
  async takeScreenshot(outputPath: string): Promise<void> {
    if (!this.page) throw new Error('Browser not launched');

    const dir = path.dirname(outputPath);
    await fs.mkdir(dir, { recursive: true });

    // Hide UI elements for clean screenshot
    await this.page.evaluate(() => {
      const panels = document.querySelectorAll('.panel, .controls-panel, #controls, #info');
      panels.forEach(p => (p as HTMLElement).style.display = 'none');
    });

    await this.page.screenshot({ path: outputPath, type: 'png' });

    // Restore UI
    await this.page.evaluate(() => {
      const panels = document.querySelectorAll('.panel, .controls-panel, #controls, #info');
      panels.forEach(p => (p as HTMLElement).style.display = '');
    });

    log.info({ path: outputPath }, 'Screenshot saved');
  }

  /** Take screenshots from multiple camera angles */
  async takeMultiAngleScreenshots(
    outputDir: string,
    prefix: string = 'preview',
    angles: CameraAngle[] = DEFAULT_ANGLES,
  ): Promise<string[]> {
    const paths: string[] = [];

    for (const angle of angles) {
      await this.setCameraAngle(angle);
      const filePath = path.join(outputDir, `${prefix}_${angle.name}.png`);
      await this.takeScreenshot(filePath);
      paths.push(filePath);
    }

    return paths;
  }

  /**
   * Extract runtime data from the browser.
   * Pulls deformed/rest positions, bone weights, transforms, adjacency, joints.
   */
  async extractRuntimeData(): Promise<ExtractedRuntimeData> {
    if (!this.page) throw new Error('Browser not launched');

    const data = await this.page.evaluate(() => {
      const w = window as any;

      // Get deformed positions from mesh geometry
      const meshGeo = w._meshRestPos ? null : undefined;
      const scene = null; // We'll look for the mesh in the Three.js scene

      // Get rest positions
      const restPositions = w._meshRestPos
        ? Array.from(w._meshRestPos as Float32Array)
        : [];

      // Get current deformed positions from the live mesh geometry
      let deformedPositions: number[] = [];
      if (w._meshObj && w._meshObj.geometry && w._meshObj.geometry.attributes.position) {
        deformedPositions = Array.from(w._meshObj.geometry.attributes.position.array as Float32Array);
      } else {
        deformedPositions = restPositions.slice();
      }

      // Get bone transforms
      const boneTransforms = w._lastBoneTransforms
        ? (w._lastBoneTransforms as any[]).map((bt: any) => {
            if (!bt) return null;
            return {
              R: Array.from(bt.R as Float64Array || bt.R),
              t: Array.from(bt.t as number[]),
            };
          })
        : [];

      // Get joints from P
      const joints: { [name: string]: [number, number, number] } = {};
      if (w._P) {
        for (const [name, entry] of Object.entries(w._P as Record<string, any>)) {
          if (entry && entry.P && entry.P.length >= 3) {
            joints[name] = [entry.P[0], entry.P[1], entry.P[2]];
          }
        }
      }

      // Get bone weights
      const boneWeights: (number[] | null)[] = w._meshBoneWeights
        ? (w._meshBoneWeights as any[]).map((bw: any) => bw ? Array.from(bw) : null)
        : [];

      // Get adjacency
      let adjacency: { edgeA: number[]; edgeB: number[]; edgeRestLen: number[]; nEdges: number; nVerts: number } | null = null;
      if (w._meshAdjacency) {
        const adj = w._meshAdjacency;
        adjacency = {
          edgeA: Array.from(adj.edgeA) as number[],
          edgeB: Array.from(adj.edgeB) as number[],
          edgeRestLen: Array.from(adj.edgeRestLen) as number[],
          nEdges: adj.nEdges as number,
          nVerts: adj.nVerts as number,
        };
      }

      return {
        deformedPositions,
        restPositions,
        boneWeights,
        boneTransforms,
        adjacency,
        joints,
        fkJoints: {} as { [name: string]: [number, number, number] },
      };
    });

    log.info({
      restVerts: data.restPositions.length / 3,
      jointCount: Object.keys(data.joints).length,
      boneTransformCount: data.boneTransforms.length,
    }, 'Runtime data extracted');

    return data;
  }

  /** Close the browser */
  async close(): Promise<void> {
    if (this.context) await this.context.close();
    if (this.browser) await this.browser.close();
    this.page = null;
    this.context = null;
    this.browser = null;
    log.info('Browser closed');
  }

  /** Check if the browser is connected */
  get isConnected(): boolean {
    return this.browser?.isConnected() ?? false;
  }
}
