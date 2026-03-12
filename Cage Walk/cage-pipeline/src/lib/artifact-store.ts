import fs from 'node:fs/promises';
import path from 'node:path';
import { getEnv } from '../config/env.js';

function artifactDir(): string {
  return getEnv().ARTIFACT_DIR;
}

function runDir(runId: string): string {
  return path.join(artifactDir(), runId);
}

/** Ensure the run's artifact directory exists */
async function ensureRunDir(runId: string): Promise<string> {
  const dir = runDir(runId);
  await fs.mkdir(dir, { recursive: true });
  return dir;
}

/** Save a Float32Array to disk as raw binary */
export async function saveFloat32(runId: string, name: string, data: Float32Array): Promise<string> {
  const dir = await ensureRunDir(runId);
  const filePath = path.join(dir, `${name}.f32`);
  await fs.writeFile(filePath, Buffer.from(data.buffer, data.byteOffset, data.byteLength));
  return filePath;
}

/** Load a Float32Array from disk */
export async function loadFloat32(filePath: string): Promise<Float32Array> {
  const buf = await fs.readFile(filePath);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

/** Save a JSON artifact to disk */
export async function saveJSON(runId: string, name: string, data: unknown): Promise<string> {
  const dir = await ensureRunDir(runId);
  const filePath = path.join(dir, `${name}.json`);
  await fs.writeFile(filePath, JSON.stringify(data));
  return filePath;
}

/** Load a JSON artifact from disk */
export async function loadJSON<T = unknown>(filePath: string): Promise<T> {
  const raw = await fs.readFile(filePath, 'utf-8');
  return JSON.parse(raw) as T;
}

/** Get the size of a file in bytes */
export async function getFileSize(filePath: string): Promise<number> {
  const stat = await fs.stat(filePath);
  return stat.size;
}

/** Delete all artifacts for a run */
export async function deleteRunArtifacts(runId: string): Promise<void> {
  const dir = runDir(runId);
  await fs.rm(dir, { recursive: true, force: true });
}
