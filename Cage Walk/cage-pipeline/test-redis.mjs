import 'dotenv/config';
import { Queue } from 'bullmq';

const url = new URL(process.env.REDIS_URL);
const connection = {
  host: url.hostname,
  port: parseInt(url.port, 10) || 6379,
  password: url.password || undefined,
  maxRetriesPerRequest: null,
};
if (url.protocol === 'rediss:') connection.tls = {};

const q = new Queue('test-ping', { connection });
await q.add('ping', { test: true });
console.log('Redis OK — job added');
await q.close();
process.exit(0);
