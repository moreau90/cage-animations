import { Queue, FlowProducer } from 'bullmq';
import { getRedisConnection } from '../config/redis.js';
import { JOB_NAMES } from '../types/job-payloads.js';

const queues = new Map<string, Queue>();

export function getQueue(name: string): Queue {
  let q = queues.get(name);
  if (!q) {
    q = new Queue(name, { connection: getRedisConnection() });
    queues.set(name, q);
  }
  return q;
}

let _flowProducer: FlowProducer | null = null;

export function getFlowProducer(): FlowProducer {
  if (!_flowProducer) {
    _flowProducer = new FlowProducer({ connection: getRedisConnection() });
  }
  return _flowProducer;
}

export async function closeQueues(): Promise<void> {
  for (const q of queues.values()) {
    await q.close();
  }
  queues.clear();
  if (_flowProducer) {
    await _flowProducer.close();
    _flowProducer = null;
  }
}

export { JOB_NAMES };
