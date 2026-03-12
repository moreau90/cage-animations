import postgres from 'postgres';
import fs from 'node:fs';

const sql = postgres('postgresql://postgres.wqhjeqviduvlgnalyour:pr2b7ro5QWp9XvbA@aws-1-us-east-1.pooler.supabase.com:5432/postgres', {
  ssl: 'require',
});

async function run() {
  // Test connection
  const result = await sql`SELECT 1 as ok`;
  console.log('Connected:', result[0].ok === 1 ? 'OK' : 'FAIL');

  const migration = fs.readFileSync('supabase/migrations/001_initial_schema.sql', 'utf-8');

  // Run the entire migration as one block
  await sql.unsafe(migration);
  console.log('Migration complete — all tables created');

  // Verify tables exist
  const tables = await sql`
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name
  `;
  console.log('Tables:', tables.map(t => t.table_name).join(', '));

  await sql.end();
}

run().catch(e => { console.error('Migration failed:', e.message); process.exit(1); });
