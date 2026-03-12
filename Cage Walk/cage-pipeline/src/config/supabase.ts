import { createClient, type SupabaseClient } from '@supabase/supabase-js';
import { getEnv } from './env.js';

let _client: SupabaseClient | null = null;

export function getSupabase(): SupabaseClient {
  if (!_client) {
    const env = getEnv();
    _client = createClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);
  }
  return _client;
}
