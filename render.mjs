#!/usr/bin/env node
/*
 * render.mjs - get sound out of Web Audio code and into a WAV, with nobody
 * in the room. Uses real Chrome, so what gets measured is the same audio
 * engine a person would actually hear.
 *
 * Two modes:
 *
 *   node render.mjs patch synth.js out.wav --seconds 4 [--sr 48000] [--channels 2]
 *       synth.js is plain browser JS that defines:
 *           async function build(ctx, seconds) { ...connect things to ctx.destination... }
 *       It is rendered through an OfflineAudioContext: faster than real time
 *       and sample-exact, so two runs of the same code give the same file.
 *       --params '{"drive":1.2}' is exposed to the patch as window.EARS_PARAMS,
 *       so a script can render many variants of one patch (see tune.py).
 *
 *   node render.mjs page app.html out.wav --seconds 5 [--do click:#start] [--do wait:500] ...
 *       Opens a real page (file path or URL), taps everything any AudioContext
 *       sends to its speakers, performs the --do steps in order, and records
 *       in real time. Steps: click:<selector>  wait:<ms>  key:<Key>
 *       type:<text>  eval:<js>  move:<x>,<y>
 *       Recording starts before the steps run. Only Web Audio is captured;
 *       bare <audio>/<video> tags that never touch an AudioContext are not.
 *
 * Needs: npm i puppeteer-core, and Chrome (set CHROME=/path if not found).
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import puppeteer from "puppeteer-core";

const CHROME_CANDIDATES = [
  process.env.CHROME,
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
].filter(Boolean);

function findChrome() {
  for (const c of CHROME_CANDIDATES) if (fs.existsSync(c)) return c;
  throw new Error("Chrome not found. Set CHROME=/path/to/chrome");
}

function parseArgs(argv) {
  const pos = [];
  const opt = { seconds: 4, sr: 48000, channels: 2, do: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--do") opt.do.push(argv[++i]);
    else if (a.startsWith("--")) opt[a.slice(2)] = argv[++i];
    else pos.push(a);
  }
  opt.seconds = Number(opt.seconds);
  opt.sr = Number(opt.sr);
  opt.channels = Number(opt.channels);
  return { pos, opt };
}

function writeWav(file, channels, sampleRate) {
  const nch = channels.length;
  const n = nch ? channels[0].length : 0;
  const data = Buffer.alloc(n * nch * 4);
  for (let i = 0; i < n; i++)
    for (let c = 0; c < nch; c++) data.writeFloatLE(channels[c][i], (i * nch + c) * 4);
  const h = Buffer.alloc(44);
  h.write("RIFF", 0);
  h.writeUInt32LE(36 + data.length, 4);
  h.write("WAVE", 8);
  h.write("fmt ", 12);
  h.writeUInt32LE(16, 16);
  h.writeUInt16LE(3, 20); // IEEE float
  h.writeUInt16LE(nch, 22);
  h.writeUInt32LE(sampleRate, 24);
  h.writeUInt32LE(sampleRate * nch * 4, 28);
  h.writeUInt16LE(nch * 4, 32);
  h.writeUInt16LE(32, 34);
  h.write("data", 36);
  h.writeUInt32LE(data.length, 40);
  fs.writeFileSync(file, Buffer.concat([h, data]));
}

// Pull Float32 channel data out of the page in base64 slices.
async function pullChannels(page, nch, length) {
  const SLICE = 1 << 19;
  const out = [];
  for (let c = 0; c < nch; c++) {
    const ch = new Float32Array(length);
    for (let off = 0; off < length; off += SLICE) {
      const b64 = await page.evaluate(
        (c, off, len) => {
          const a = window.__ears_out[c].subarray(off, off + len);
          const u8 = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
          let s = "";
          for (let i = 0; i < u8.length; i += 0x8000)
            s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
          return btoa(s);
        },
        c, off, Math.min(SLICE, length - off));
      const buf = Buffer.from(b64, "base64");
      ch.set(new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4), off);
    }
    out.push(ch);
  }
  return out;
}

async function launch() {
  return puppeteer.launch({
    executablePath: findChrome(),
    headless: true,
    args: [
      "--no-sandbox",
      "--autoplay-policy=no-user-gesture-required",
      "--allow-file-access-from-files",
      "--mute-audio",
    ],
  });
}

function wireLogs(page) {
  page.on("console", (m) => console.error(`[page ${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => console.error(`[page error] ${e.message}`));
}

async function renderPatch(src, out, opt) {
  const code = fs.readFileSync(src, "utf8");
  const browser = await launch();
  try {
    const page = await browser.newPage();
    wireLogs(page);
    await page.goto("about:blank");
    if (opt.params) await page.evaluate((p) => { window.EARS_PARAMS = JSON.parse(p); }, opt.params);
    await page.addScriptTag({ content: code });
    const info = await page.evaluate(async (seconds, sr, channels) => {
      if (typeof build !== "function")
        throw new Error("patch must define: async function build(ctx, seconds)");
      const ctx = new OfflineAudioContext(channels, Math.ceil(seconds * sr), sr);
      await build(ctx, seconds);
      const buf = await ctx.startRendering();
      window.__ears_out = [];
      for (let c = 0; c < buf.numberOfChannels; c++) window.__ears_out.push(buf.getChannelData(c));
      return { nch: buf.numberOfChannels, length: buf.length, sr: buf.sampleRate };
    }, opt.seconds, opt.sr, opt.channels);
    const chans = await pullChannels(page, info.nch, info.length);
    writeWav(out, chans, info.sr);
    console.log(`rendered ${info.length / info.sr}s, ${info.nch}ch @ ${info.sr} Hz -> ${out}`);
  } finally {
    await browser.close();
  }
}

// Runs inside the page before any of its own scripts. Every connection to an
// AudioDestinationNode is also fed to a recorder.
function installTap() {
  const E = (window.__ears = { chunks: [[], []], frames: 0, sr: 0, contexts: 0, recording: false });
  const origConnect = AudioNode.prototype.connect;
  const taps = new WeakMap();
  function tapFor(ctx) {
    if (taps.has(ctx)) return taps.get(ctx);
    const sp = ctx.createScriptProcessor(4096, 2, 2);
    sp.onaudioprocess = (ev) => {
      if (!E.recording) return;
      const ib = ev.inputBuffer;
      E.chunks[0].push(new Float32Array(ib.getChannelData(0)));
      E.chunks[1].push(new Float32Array(ib.getChannelData(ib.numberOfChannels > 1 ? 1 : 0)));
      E.frames += ib.length;
    };
    origConnect.call(sp, ctx.destination); // must reach the output or it never runs
    E.sr = ctx.sampleRate;
    E.contexts++;
    taps.set(ctx, sp);
    return sp;
  }
  AudioNode.prototype.connect = function (dest, ...rest) {
    const r = origConnect.call(this, dest, ...rest);
    try {
      if (dest instanceof AudioDestinationNode && !(dest.context instanceof OfflineAudioContext)) {
        const tap = tapFor(dest.context);
        if (this !== tap) origConnect.call(this, tap, rest[0] || 0, 0);
      }
    } catch (e) {
      console.warn("ears tap failed: " + e.message);
    }
    return r;
  };
}

async function runStep(page, step) {
  const i = step.indexOf(":");
  const kind = i < 0 ? step : step.slice(0, i);
  const arg = i < 0 ? "" : step.slice(i + 1);
  if (kind === "click") await page.click(arg);
  else if (kind === "wait") await new Promise((r) => setTimeout(r, Number(arg)));
  else if (kind === "key") await page.keyboard.press(arg);
  else if (kind === "type") await page.keyboard.type(arg);
  else if (kind === "eval") await page.evaluate(arg);
  else if (kind === "move") {
    const [x, y] = arg.split(",").map(Number);
    await page.mouse.move(x, y);
  } else throw new Error(`unknown step: ${step}`);
}

async function recordPage(target, out, opt) {
  const url = /^https?:\/\//.test(target) ? target : pathToFileURL(path.resolve(target)).href;
  const browser = await launch();
  try {
    const page = await browser.newPage();
    wireLogs(page);
    await page.evaluateOnNewDocument(installTap);
    await page.goto(url, { waitUntil: "load" });
    await page.evaluate(() => { window.__ears.recording = true; });
    const t0 = Date.now();
    for (const step of opt.do) await runStep(page, step);
    const left = opt.seconds * 1000 - (Date.now() - t0);
    if (left > 0) await new Promise((r) => setTimeout(r, left));
    const info = await page.evaluate(() => {
      const E = window.__ears;
      E.recording = false;
      window.__ears_out = E.chunks.map((list) => {
        const a = new Float32Array(E.frames);
        let o = 0;
        for (const c of list) { a.set(c, o); o += c.length; }
        return a;
      });
      return { length: E.frames, sr: E.sr, contexts: E.contexts };
    });
    if (!info.contexts)
      throw new Error("No AudioContext ever connected to its destination. The page made no Web Audio sound " +
        "(or it needs a --do step, like a click, to start).");
    if (!info.length)
      throw new Error("An AudioContext exists but produced no audio frames. It is probably still suspended; " +
        "add a --do click:<selector> step that starts it.");
    const chans = await pullChannels(page, 2, info.length);
    writeWav(out, chans, info.sr);
    console.log(`recorded ${(info.length / info.sr).toFixed(2)}s from ${info.contexts} AudioContext(s) @ ${info.sr} Hz -> ${out}`);
  } finally {
    await browser.close();
  }
}

const { pos, opt } = parseArgs(process.argv.slice(2));
const [mode, src, out] = pos;
if (!["patch", "page"].includes(mode) || !src || !out) {
  console.error("usage:\n  node render.mjs patch synth.js out.wav --seconds 4 [--sr 48000] [--channels 2]\n" +
    "  node render.mjs page  app.html out.wav --seconds 5 [--do click:#start] [--do wait:500]");
  process.exit(64);
}
try {
  if (mode === "patch") await renderPatch(src, out, opt);
  else await recordPage(src, out, opt);
} catch (e) {
  console.error("render failed: " + e.message);
  process.exit(1);
}
