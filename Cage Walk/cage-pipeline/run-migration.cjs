const { Client } = require('pg');
const fs = require('fs');

const client = new Client({
  connectionString: 'postgresql://postgres:pr2b7ro5QWp9XvbA@db.wqhjeqviduvlgnalyour.supabase.co:5432/postgres',
  ssl: { rejectUnauthorized: false }
});

async function run() {
  await client.connect();
  console.log('Connected to Supabase Postgres');

  const sql = fs.readFileSync('supabase/migrations/001_initial_schema.sql', 'utf-8');

  // Remove the embeddings_index vector column line since pgvector may not be enabled
  // Split by semicolons and run each statement
  const statements = sql
    .split(';')
    .map(s => s.trim())
    .filter(s => s.length > 0 && !s.startsWith('--'));

  for (const stmt of statements) {
    try {
      await client.query(stmt);
      const firstLine = stmt.split('\n').find(l => l.trim() && !l.trim().startsWith('--')) || stmt.substring(0, 60);
      console.log('OK:', firstLine.trim().substring(0, 70));
    } catch (err) {
      console.error('ERR:', err.message.split('\n')[0], '|', stmt.substring(0, 50));
    }
  }

  await client.end();
  console.log('Migration complete');
}

run().catch(e => { console.error(e); process.exit(1); });
