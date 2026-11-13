#!/usr/bin/env python3
"""
tune.py - close the loop. Render many variants of a patch, let the ears score
each one, keep the best. This is the part where no human has to listen until
the end.

  python3 tune.py demo/engine_v3.js --want Engine --want Vehicle --avoid Synthesizer \\
      --seconds 6 --rounds 3 --per-round 14

The patch must read its numbers from window.EARS_PARAMS (see demo/engine_v3.js).
Search space comes from a JSON file of {"name": [low, high]} (--space), or the
built-in one for the demo engine.

Score = best wanted label minus best avoided label, minus a penalty for any
hard fault (clipping, clicks). Warning: this optimises what a neural network
thinks, and networks can be fooled. Listen to the winner before believing it.
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ears

ENGINE_SPACE = {
    "riseMs": [0.3, 4.0], "tauFrac": [0.1, 0.5], "burnNoise": [0.0, 1.2],
    "timingJitter": [0.01, 0.08], "ampJitter": [0.03, 0.3], "cylSpread": [0.3, 2.0],
    "clatter": [0.0, 0.2], "clatterRev": [0.02, 0.4], "clatterHz": [500, 3000],
    "floorNoise": [0.0, 0.25], "floorHz": [120, 1500], "subDb": [0, 12],
    "bodyGain": [0.3, 2.0], "dryGain": [0.1, 1.0], "drive": [0.6, 4.0], "topHz": [1500, 9000],
}


def render(patch, params, seconds, out):
    r = subprocess.run(["node", os.path.join(HERE, "render.mjs"), "patch", patch, out,
                        "--seconds", str(seconds), "--channels", "1", "--params", json.dumps(params)],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())


def score(wav, want, avoid, tg):
    sr, x = ears.load(wav)
    tr = tg.tag(x, sr)
    if tr is None:
        return -9.0, {}
    p, lab = tr["probs"], tg.labels()
    w = {lab[i]: float(p[i]) for q in want for i in tg.find_labels(q)}
    a = {lab[i]: float(p[i]) for q in avoid for i in tg.find_labels(q)}
    mono = x.mean(axis=1)
    faults = len(ears.find_clipping(x, sr)) > 0 or len(ears.find_clicks_all(x, sr)) > 0
    s = max(w.values(), default=0) - max(a.values(), default=0) - (0.5 if faults else 0)
    return s, {"want": w, "avoid": a, "faults": faults,
               "top": tg.top(p, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patch")
    ap.add_argument("--want", action="append", default=[])
    ap.add_argument("--avoid", action="append", default=[])
    ap.add_argument("--space", default=None)
    ap.add_argument("--seconds", type=float, default=6)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-round", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="where to write the winning WAV + params")
    a = ap.parse_args()
    tg = ears.get_tagger()
    if tg is None:
        sys.exit("tune needs the recognition layer (torch + torchaudio)")
    space = json.load(open(a.space)) if a.space else ENGINE_SPACE
    rng = np.random.default_rng(a.seed)
    tmp = tempfile.mkdtemp(prefix="tune_")
    wav = os.path.join(tmp, "v.wav")

    render(a.patch, {}, a.seconds, wav)
    best_s, info = score(wav, a.want, a.avoid, tg)
    best_p = {}
    print(f"start  score {best_s:+.3f}  {', '.join(f'{l} {v:.2f}' for l, v in info['top'])}")
    width = 1.0
    for rd in range(a.rounds):
        for k in range(a.per_round):
            cand = dict(best_p)
            for name, (lo, hi) in space.items():
                if rd == 0 and k < a.per_round // 2:
                    cand[name] = float(rng.uniform(lo, hi))            # explore wide first
                elif rng.random() < 0.5:
                    centre = cand.get(name, None)
                    if centre is None:
                        cand[name] = float(rng.uniform(lo, hi))
                    else:
                        cand[name] = float(np.clip(centre + rng.normal() * width * 0.25 * (hi - lo), lo, hi))
            try:
                render(a.patch, cand, a.seconds, wav)
            except RuntimeError as e:
                print("  render failed:", e)
                continue
            s, inf = score(wav, a.want, a.avoid, tg)
            if s > best_s:
                best_s, best_p, info = s, cand, inf
                print(f"r{rd + 1}.{k + 1:<3} score {s:+.3f}  {', '.join(f'{l} {v:.2f}' for l, v in inf['top'])}")
        width *= 0.6
    out = a.out or os.path.splitext(a.patch)[0] + "_tuned"
    render(a.patch, best_p, a.seconds, out + ".wav")
    json.dump({k: round(v, 4) for k, v in best_p.items()}, open(out + ".params.json", "w"), indent=1)
    print(f"\nbest score {best_s:+.3f}")
    print("params:", json.dumps({k: round(v, 3) for k, v in best_p.items()}))
    print(f"wrote {out}.wav and {out}.params.json")


if __name__ == "__main__":
    main()
