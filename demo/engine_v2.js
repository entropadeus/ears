// Engine v2, rebuilt from what ears.py said was wrong with v1:
//   v1 pitch jitter 1 cent (real engines measured 11-31)  -> per-firing timing jitter
//   v1 level shimmer 0.5 dB (real 1.1-1.4)                -> per-firing strength varies, cylinders unequal
//   v1 almost no energy above 500 Hz that is not harmonic -> combustion + mechanical noise
//   v1 called "Synthesizer"                               -> no oscillators at all: every
//      firing is its own pressure pulse, fed through fixed exhaust/body resonances
// Idle 1.5 s, rev up over 2.5 s, hold, 4-cylinder four-stroke.
function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }
function gauss(r) { return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()); }

function rpmAt(t) {
  if (t < 1.5) return 900;
  if (t < 4.0) { const u = (t - 1.5) / 2.5; return 900 + (4200 - 900) * u * u * (3 - 2 * u); }
  return 4200;
}

async function build(ctx, seconds) {
  const sr = ctx.sampleRate, n = Math.ceil(seconds * sr);
  const exc = ctx.createBuffer(1, n, sr), mech = ctx.createBuffer(1, n, sr);
  const e = exc.getChannelData(0), m = mech.getChannelData(0);
  const r = rng(1234);
  const cyl = [1.0, 0.82, 0.93, 0.74];           // cylinders never fire equally
  let t = 0, k = 0, misfire = 0;
  while (t < seconds) {
    const rpm = rpmAt(t), load = (rpm - 900) / 3300;
    const period = 60 / rpm / 2;                 // 2 firings per rev
    let a = cyl[k % 4] * (1 + 0.12 * gauss(r)) * (0.55 + 0.45 * load);
    if (r() < 0.012) a *= 0.35;                  // the odd weak burn
    const i0 = Math.floor(t * sr);
    // pressure pulse: fast rise, slower decay, length tied to the cycle
    const len = Math.min(n - i0, Math.floor(period * sr * 0.9));
    const rise = Math.max(8, Math.floor(0.0006 * sr)), tau = period * sr * 0.22;
    for (let i = 0; i < len; i++) {
      const env = (i < rise ? i / rise : 1) * Math.exp(-i / tau);
      e[i0 + i] += a * env * (1 + 0.5 * gauss(r) * Math.exp(-i / (tau * 0.4)));  // burn is turbulent
    }
    // mechanical clatter + exhaust rush riding on each firing
    const nl = Math.min(n - i0, Math.floor(period * sr));
    for (let i = 0; i < nl; i++) m[i0 + i] += gauss(r) * a * (0.25 + 0.75 * Math.exp(-i / (tau * 0.8)));
    t += period * (1 + 0.035 * gauss(r));        // timing scatter = pitch jitter
    k++;
  }
  // slow wander of the whole thing (governor hunting)
  for (let i = 0; i < n; i++) { const w = 1 + 0.08 * Math.sin(2 * Math.PI * 1.3 * i / sr) + 0.05 * Math.sin(2 * Math.PI * 3.1 * i / sr); e[i] *= w; m[i] *= w; }

  const src = new AudioBufferSourceNode(ctx, { buffer: exc });
  const nsrc = new AudioBufferSourceNode(ctx, { buffer: mech });
  const hp = new BiquadFilterNode(ctx, { type: "highpass", frequency: 22, Q: 0.7 });
  const sum = new GainNode(ctx, { gain: 1 });
  // exhaust pipe + body: fixed resonances that do NOT move with RPM
  const dry = new GainNode(ctx, { gain: 0.55 });
  src.connect(hp); hp.connect(dry).connect(sum);
  for (const [f, q, g] of [[95, 3, 1.0], [180, 4, 0.8], [330, 5, 0.5], [620, 6, 0.3], [1150, 5, 0.16]]) {
    const bp = new BiquadFilterNode(ctx, { type: "bandpass", frequency: f, Q: q });
    hp.connect(bp).connect(new GainNode(ctx, { gain: g })).connect(sum);
  }
  // noise path: broadband rush shaped like a pipe, plus tappet-ish band higher up
  const nlp = new BiquadFilterNode(ctx, { type: "lowpass", frequency: 2600, Q: 0.5 });
  const nbp = new BiquadFilterNode(ctx, { type: "bandpass", frequency: 1300, Q: 0.8 });
  const ng = new GainNode(ctx, { gain: 0.05 });
  ng.gain.setValueAtTime(0.05, 0); ng.gain.setValueAtTime(0.05, 1.5); ng.gain.linearRampToValueAtTime(0.11, 4.0);
  nsrc.connect(nlp).connect(nbp).connect(ng).connect(sum);
  // a little grit, then tame the top
  const shaper = new WaveShaperNode(ctx, { oversample: "4x" });
  const curve = new Float32Array(1024); for (let i = 0; i < 1024; i++) { const x = i / 511.5 - 1; curve[i] = Math.tanh(1.6 * x); }
  shaper.curve = curve;
  const pre = new GainNode(ctx, { gain: 0.5 });
  const lp = new BiquadFilterNode(ctx, { type: "lowpass", frequency: 5200, Q: 0.6 });
  const out = new GainNode(ctx, { gain: 0 });
  out.gain.setValueAtTime(0, 0); out.gain.linearRampToValueAtTime(0.5, 0.04);
  out.gain.setValueAtTime(0.5, seconds - 0.08); out.gain.linearRampToValueAtTime(0, seconds);
  sum.connect(pre).connect(shaper).connect(lp).connect(out).connect(ctx.destination);
  src.start(); nsrc.start();
}
