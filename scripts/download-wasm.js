#!/usr/bin/env node
/**
 * scripts/download-wasm.js
 * Downloads the official hash256 WASM miner files into vendor/miner/.
 * Runs automatically on `npm install` (postinstall) and manually via `npm run refresh-miner`.
 * Works on Windows, macOS, and Linux — no curl required.
 */

import fs from 'node:fs';
import path from 'node:path';
import https from 'node:https';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEST = path.resolve(__dirname, '..', 'vendor', 'miner');

const FILES = [
  { url: 'https://hash256.fun/miner/hash_miner.js', dest: path.join(DEST, 'hash_miner.js') },
  { url: 'https://hash256.fun/miner/hash_miner_bg.wasm', dest: path.join(DEST, 'hash_miner_bg.wasm') },
];

fs.mkdirSync(DEST, { recursive: true });

async function download(url, dest) {
  return new Promise((resolve, reject) => {
    const tmp = dest + '.tmp';
    const file = fs.createWriteStream(tmp);
    const req = https.get(url, (res) => {
      if (res.statusCode === 301 || res.statusCode === 302) {
        file.close();
        fs.rmSync(tmp, { force: true });
        return download(res.headers.location, dest).then(resolve).catch(reject);
      }
      if (res.statusCode !== 200) {
        file.close();
        fs.rmSync(tmp, { force: true });
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
      }
      res.pipe(file);
      file.on('finish', () => {
        file.close(() => {
          fs.renameSync(tmp, dest);
          resolve();
        });
      });
    });
    req.on('error', (err) => {
      file.close();
      fs.rmSync(tmp, { force: true });
      reject(err);
    });
  });
}

let failed = false;
for (const { url, dest } of FILES) {
  const name = path.basename(dest);
  try {
    process.stdout.write(`  downloading ${name}... `);
    await download(url, dest);
    console.log('ok');
  } catch (err) {
    console.log(`FAILED (${err.message})`);
    console.warn(`  WARNING: Could not download ${name}. Run "npm run refresh-miner" later.`);
    failed = true;
  }
}

if (!failed) console.log('vendor/miner files ready.');
