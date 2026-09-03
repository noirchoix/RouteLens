import { copyFile, mkdir, readdir, rm, stat } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, relative, resolve } from 'node:path';
import process from 'node:process';

const uiRoot = resolve('.');
const stageRoot = resolve('.embedded-build');
const targetRoot = resolve('../src/reacts/ui/static');
const viteEntry = resolve('node_modules/vite/bin/vite.js');

async function filesUnder(root) {
  const entries = [];
  for (const name of await readdir(root)) {
    const path = resolve(root, name);
    const info = await stat(path);
    if (info.isDirectory()) entries.push(...await filesUnder(path));
    else if (info.isFile()) entries.push(path);
  }
  return entries;
}

await rm(stageRoot, { recursive: true, force: true });

const result = spawnSync(process.execPath, [viteEntry, 'build'], {
  cwd: uiRoot,
  env: { ...process.env, ROUTELENS_UI_OUTPUT_DIR: stageRoot },
  stdio: 'inherit'
});

if (result.error) {
  await rm(stageRoot, { recursive: true, force: true });
  throw result.error;
}

if (result.status !== 0) {
  await rm(stageRoot, { recursive: true, force: true });
  process.exit(result.status ?? 1);
}

const stagedIndex = resolve(stageRoot, 'index.html');
try {
  const info = await stat(stagedIndex);
  if (!info.isFile()) throw new Error('SvelteKit did not emit index.html');
} catch (error) {
  await rm(stageRoot, { recursive: true, force: true });
  throw new Error('Embedded build completed without a usable index.html', { cause: error });
}

await mkdir(targetRoot, { recursive: true });
for (const source of await filesUnder(stageRoot)) {
  const destination = resolve(targetRoot, relative(stageRoot, source));
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(source, destination);
}

if (!(await readdir(stageRoot)).includes('_app')) {
  await rm(resolve(targetRoot, '_app'), { recursive: true, force: true });
}

await rm(stageRoot, { recursive: true, force: true });
console.log(`Embedded RouteLens UI promoted to ${targetRoot}`);
