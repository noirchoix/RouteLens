import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

const outputDirectory = process.env.ROUTELENS_UI_OUTPUT_DIR || 'build';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({
      pages: outputDirectory,
      assets: outputDirectory,
      strict: true
    }),
    output: {
      bundleStrategy: 'inline'
    }
  }
};

export default config;
