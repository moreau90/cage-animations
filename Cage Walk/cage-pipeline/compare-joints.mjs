import fs from 'fs';
import { execSync } from 'child_process';
import path from 'path';

const cageWalkDir = path.resolve(import.meta.dirname, '..');
const current = JSON.parse(fs.readFileSync(path.join(cageWalkDir, 'placed_joints.json'), 'utf-8'));
const gitContent = execSync('git show HEAD:placed_joints.json', { encoding: 'utf-8', cwd: cageWalkDir });
const original = JSON.parse(gitContent);

const cP = current.P;
const oP = original.P;
const changed = [];
for (const [name, val] of Object.entries(cP)) {
  const ov = oP[name];
  if (!ov) { changed.push({ name, note: 'NEW' }); continue; }
  const cv = Array.isArray(val) ? val : val.P || val;
  const ovv = Array.isArray(ov) ? ov : ov.P || ov;
  const dx = (cv[0] - ovv[0]) * 1000;
  const dy = (cv[1] - ovv[1]) * 1000;
  const dz = (cv[2] - ovv[2]) * 1000;
  const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (dist > 0.01) {
    changed.push({ name, dx: dx.toFixed(2), dy: dy.toFixed(2), dz: dz.toFixed(2), dist: dist.toFixed(2) });
  }
}

console.log('Changed joints: ' + changed.length + ' / ' + Object.keys(cP).length);
changed.sort((a, b) => parseFloat(b.dist) - parseFloat(a.dist));
for (const c of changed) {
  console.log('  ' + c.name.padEnd(20) + ' dx=' + c.dx.padStart(7) + ' dy=' + c.dy.padStart(7) + ' dz=' + c.dz.padStart(7) + '  total=' + c.dist + 'mm');
}
console.log('Unchanged: ' + (Object.keys(cP).length - changed.length));
console.log('primary preserved: ' + (!!current.primary));
console.log('bone_lengths preserved: ' + (!!current.bone_lengths));
console.log('mesh_height: ' + current.mesh_height);
