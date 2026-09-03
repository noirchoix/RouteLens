import { rm } from 'node:fs/promises';
import { resolve } from 'node:path';

const staticRoot = resolve('../src/reacts/ui/static');
await Promise.all([
  rm(resolve(staticRoot, '_app'), { recursive: true, force: true }),
  rm(resolve(staticRoot, 'index.html'), { force: true })
]);
