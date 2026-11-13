// A crude engine: sawtooth "firing" tone + sub, through a lowpass that opens
// with revs. RPM sweeps 1200 -> 6000 over the render. Deliberate bug planted:
// at 2.0s the gain is slammed to a new value with no ramp.
async function build(ctx, seconds) {
  const osc = new OscillatorNode(ctx, { type: "sawtooth" });
  const sub = new OscillatorNode(ctx, { type: "square" });
  const lp = new BiquadFilterNode(ctx, { type: "lowpass", Q: 4 });
  const g = new GainNode(ctx, { gain: 0 });
  const hz = (rpm) => (rpm / 60) * 2;           // 4-stroke, 4 cyl: 2 firings per rev
  osc.frequency.setValueAtTime(hz(1200), 0);
  osc.frequency.exponentialRampToValueAtTime(hz(6000), seconds);
  sub.frequency.setValueAtTime(hz(1200) / 2, 0);
  sub.frequency.exponentialRampToValueAtTime(hz(6000) / 2, seconds);
  lp.frequency.setValueAtTime(400, 0);
  lp.frequency.exponentialRampToValueAtTime(5000, seconds);
  g.gain.setValueAtTime(0, 0);
  g.gain.linearRampToValueAtTime(0.35, 0.05);
  g.gain.setValueAtTime(0.35, 2.0);
  g.gain.linearRampToValueAtTime(0.12, 2.05);
  g.gain.setValueAtTime(0.12, seconds - 0.05);
  g.gain.linearRampToValueAtTime(0, seconds);
  const subG = new GainNode(ctx, { gain: 0.5 });
  osc.connect(lp); sub.connect(subG).connect(lp); lp.connect(g).connect(ctx.destination);
  osc.start(); sub.start();
}
