import { getEnv } from './env.js';

export interface RedisConnectionOptions {
  host: string;
  port: number;
  password?: string;
  tls?: {};
  maxRetriesPerRequest: null;
}

let _opts: RedisConnectionOptions | null = null;

/** Parse REDIS_URL into connection options compatible with BullMQ */
export function getRedisConnection(): RedisConnectionOptions {
  if (!_opts) {
    const url = new URL(getEnv().REDIS_URL);
    _opts = {
      host: url.hostname,
      port: parseInt(url.port, 10) || 6379,
      password: url.password || undefined,
      maxRetriesPerRequest: null, // required by BullMQ
    };
    // Upstash uses rediss:// (TLS)
    if (url.protocol === 'rediss:') {
      _opts.tls = {};
    }
  }
  return _opts;
}

export async function closeRedis(): Promise<void> {
  _opts = null;
}
