import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const sb = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY);
const searchId = '7eb29beb-5b98-4156-982d-a5d48322a9f9';

// Find all QA candidates for v22
const { data: recent } = await sb.from('runs').select('id,goal,status,decision,parent_run_id')
  .order('created_at', { ascending: false }).limit(80);

const candidates = recent.filter(r => r.goal?.includes('jitter'));

// Get scores for each candidate
for (const c of candidates) {
  const { data: scores } = await sb.from('scores').select('metric,value').eq('run_id', c.id);
  if (scores) {
    const wjf = scores.find(s => s.metric === 'worst_joint_finger_deviation_mm');
    if (wjf && wjf.value < 8) {  // Only recent v22 candidates
      const fj = scores.filter(s =>
        s.metric.startsWith('joint_deviation_') &&
        s.metric.endsWith('_mm') &&
        s.metric.indexOf('_y_mm') < 0 &&
        s.metric.indexOf('_z_mm') < 0
      ).sort((a,b) => b.value - a.value);

      console.log(c.id.slice(0,8), 'worst_per_joint=' + wjf.value.toFixed(1) + 'mm', '| top5:', fj.slice(0,5).map(f => f.metric.replace('joint_deviation_','').replace('_mm','') + '=' + f.value.toFixed(1)).join(', '));
    }
  }
}
