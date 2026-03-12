import 'dotenv/config';
import { z } from 'zod';

const envSchema = z.object({
  REDIS_URL: z.string().url().default('redis://localhost:6379'),
  SUPABASE_URL: z.string().url(),
  SUPABASE_SERVICE_KEY: z.string().min(1),
  ARTIFACT_DIR: z.string().default('./artifacts'),
  LOG_LEVEL: z.enum(['trace', 'debug', 'info', 'warn', 'error', 'fatal']).default('info'),
  /** Path to mesh.glb file (used by finger-alignment, joint-centering, processing workers) */
  MESH_GLB_PATH: z.string().optional(),
  /** Root directory of Cage Walk app (used by bridge server) */
  CAGE_WALK_DIR: z.string().optional(),
  /** Port for bridge server (default: 8765) */
  BRIDGE_PORT: z.coerce.number().default(8765),
});

export type Env = z.infer<typeof envSchema>;

let _env: Env | null = null;

export function getEnv(): Env {
  if (!_env) {
    _env = envSchema.parse(process.env);
  }
  return _env;
}
