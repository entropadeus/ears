#!/usr/bin/env python3
"""Plant known defects in synthetic audio and check ears.py catches each one.
Run: python3 selftest.py   (writes WAVs into ./selftest_audio)"""
import os
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ears

SR = 48000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selftest_audio")
os.makedirs(OUT, exist_ok=True)


def save(name, x):
    p = os.path.join(OUT, name)
    wavfile.write(p, SR, x.astype(np.float32))
    return p


def glide(f_start, f_end, dur, harmonics=6):
    t = np.arange(int(dur * SR)) / SR
    f = f_start * (f_end / f_start) ** (t / dur)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = sum(np.sin(k * ph) / k for k in range(1, harmonics + 1))
    return x / np.max(np.abs(x)), f


def fade(x, ms=20):
    n = int(SR * ms / 1000)
    r = np.linspace(0, 1, n)
    x = x.copy()
    x[:n] *= r
    x[-n:] *= r[::-1]
    return x


results = []


def check(name, cond, detail):
    results.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name:<34} {detail}")


def levels(R):
    return {n["level"] for n in R["notes"]}


# 1. clean glide: nothing should be flagged, pitch should be tracked
x, _ = glide(110, 330, 3.0)
R, _ = ears.analyze(save("clean_glide.wav", fade(x) * 0.25), use_tagger=False)
check("clean: no false alarms", not ({"BAD", "BROKEN"} & levels(R)) and R["clicks"]["count"] == 0
      and R["clipping"]["events"] == 0, f"levels={sorted(levels(R))}")
check("clean: glide start pitch ~110", abs(R["pitch"]["start_hz"] - 112) < 6, f"{R['pitch']['start_hz']:.1f} Hz")
check("clean: glide end pitch ~330", abs(R["pitch"]["end_hz"] - 326) < 12, f"{R['pitch']['end_hz']:.1f} Hz")
check("clean: zero pitch jumps", R["pitch"]["jumps"] == 0, f"jumps={R['pitch']['jumps']}")

# 2. clipping
t = np.arange(2 * SR) / SR
R, _ = ears.analyze(save("clipped.wav", np.clip(1.6 * np.sin(2 * np.pi * 220 * t), -1, 1)), use_tagger=False)
check("clipping detected", R["clipping"]["events"] > 100, f"events={R['clipping']['events']}")

# 3. clicks: hard gain chop at 1.0s, phase jump at 2.0s
x = 0.4 * np.sin(2 * np.pi * 200 * np.arange(3 * SR) / SR)
x[SR + 60:SR + 60 + SR // 4] = 0.0             # chop mid-cycle, no ramp
x[2 * SR + 60:] = 0.4 * np.sin(2 * np.pi * 200 * np.arange(SR - 60) / SR + 2.0)
R, _ = ears.analyze(save("clicks.wav", fade(x)), use_tagger=False)
ts = R["clicks"]["times_s"]
check("click at 1.0s found", any(abs(v - 1.00125) < 0.01 for v in ts), f"times={ts}")
check("click at 2.0s found", any(abs(v - 2.00125) < 0.01 for v in ts), f"times={ts}")
check("chop-end click at 1.25s found", any(abs(v - 1.25125) < 0.01 for v in ts), f"times={ts}")
check("exactly the 3 planted clicks", len(ts) == 3, f"count={len(ts)}")

# 4. pitch jump in a glide that should be smooth
a, _ = glide(100, 150, 1.5)
b, _ = glide(300, 400, 1.5)
n = 480
xf = np.concatenate([a[:-n], a[-n:] * np.linspace(1, 0, n) + b[:n] * np.linspace(0, 1, n), b[n:]])
R, _ = ears.analyze(save("pitch_jump.wav", fade(xf) * 0.25), use_tagger=False)
jl = R["pitch"].get("jump_list", [])
check("pitch jump near 1.5s found", any(abs(j["t"] - 1.5) < 0.08 for j in jl), f"jumps={[(j['t'], j['semitones']) for j in jl]}")

# 5. total silence
R, _ = ears.analyze(save("silent.wav", np.zeros(SR)), use_tagger=False)
check("silence called out", "BROKEN" in levels(R), f"levels={sorted(levels(R))}")

# 6. dropout (ramped, so it should be a dropout and not a click)
x = 0.3 * np.sin(2 * np.pi * 330 * np.arange(3 * SR) / SR)
g = np.ones(3 * SR)
r = int(0.01 * SR)
s, e = int(1.2 * SR), int(1.6 * SR)
g[s:s + r] = np.linspace(1, 0, r)
g[s + r:e - r] = 0
g[e - r:e] = np.linspace(0, 1, r)
R, _ = ears.analyze(save("dropout.wav", fade(x * g)), use_tagger=False)
d = R["silence"]["dropouts"]
check("dropout at ~1.2s found", len(d) == 1 and abs(d[0][0] - 1.21) < 0.05 and abs(d[0][1] - 0.38) < 0.06, f"dropouts={d}")
check("ramped dropout is not a click", R["clicks"]["count"] == 0, f"clicks={R['clicks']['count']}")

# 7. loudness calibration: BS.1770 says 0 dBFS 997 Hz mono sine = -3.01 LUFS
t = np.arange(5 * SR) / SR
R, _ = ears.analyze(save("cal_997.wav", np.sin(2 * np.pi * 997 * t)), use_tagger=False)
check("LUFS calibration (-3.01)", abs(R["lufs"] + 3.01) < 0.1, f"{R['lufs']:.2f} LUFS")
check("pure tone peak at 997 Hz", R["peaks"] and abs(R["peaks"][0]["hz"] - 997) < 3, f"{R['peaks'][0]['hz'] if R['peaks'] else None}")

# 8. out-of-phase stereo
m = 0.3 * np.sin(2 * np.pi * 440 * t)
R, _ = ears.analyze(save("antiphase.wav", np.stack([m, -m], axis=1)), use_tagger=False)
check("phase inversion detected", R["stereo"]["correlation"] < -0.99 and "BAD" in levels(R), f"corr={R['stereo']['correlation']:.3f}")

# 9. white noise: no pitch, high flatness
rng = np.random.default_rng(7)
R, _ = ears.analyze(save("noise.wav", fade(0.1 * rng.standard_normal(2 * SR))), use_tagger=False)
check("noise: reads as noise", R["spectral"]["flatness"] > 0.3 and R["pitch"]["voiced_fraction"] < 0.1,
      f"flatness={R['spectral']['flatness']:.2f} voiced={R['pitch']['voiced_fraction']:.2f}")
check("noise: no phantom clicks", R["clicks"]["count"] == 0, f"clicks={R['clicks']['count']}")

# 10. DC offset
R, _ = ears.analyze(save("dc.wav", 0.2 * np.sin(2 * np.pi * 300 * t) + 0.1), use_tagger=False)
check("DC offset detected", abs(R["dc_offset"][0] - 0.1) < 0.005 and "WARN" in levels(R), f"dc={R['dc_offset'][0]:.3f}")

# 11. drum-like hits must not be mistaken for clicks
x = np.zeros(3 * SR)
for k in range(6):
    i = int((0.2 + 0.45 * k) * SR)
    n = int(0.25 * SR)
    tt = np.arange(n) / SR
    hit = np.sin(2 * np.pi * (55 + 120 * np.exp(-tt * 40)) * tt) * np.exp(-tt * 14)
    hit[:96] *= np.linspace(0, 1, 96)          # 2 ms attack
    hit[-480:] *= np.linspace(1, 0, 480)       # let the tail actually reach zero
    x[i:i + n] += 0.7 * hit
R, _ = ears.analyze(save("kicks.wav", x), use_tagger=False)
check("kick drums are not clicks", R["clicks"]["count"] == 0, f"clicks={R['clicks']['count']} at {R['clicks']['times_s']}")

# 12. a raw sawtooth has a hard edge every cycle; those are not clicks
tt = np.arange(3 * SR) / SR
fq = 60 * (240 / 60) ** (tt / 3)
ph = np.cumsum(fq) / SR
saw = 0.4 * (2 * (ph % 1) - 1)
R, _ = ears.analyze(save("raw_saw.wav", fade(saw)), use_tagger=False)
check("saw edges are not clicks", R["clicks"]["count"] == 0, f"clicks={R['clicks']['count']}")
check("saw sweep pitch 60->240", abs(R["pitch"]["start_hz"] - 61) < 4 and abs(R["pitch"]["end_hz"] - 237) < 8,
      f"{R['pitch']['start_hz']:.1f} -> {R['pitch']['end_hz']:.1f} Hz")

# 13. three separate notes: pitch reads the notes, not attack garbage
x = np.zeros(int(1.6 * SR))
for k, fq in enumerate([440, 554.37, 659.25]):
    n = int(0.4 * SR)
    tn = np.arange(n) / SR
    env = np.minimum(tn / 0.02, 1) * np.exp(-tn * 9)
    env[-480:] *= np.linspace(1, 0, 480)
    x[int((0.1 + 0.45 * k) * SR):][:n] += 0.3 * env * np.sin(2 * np.pi * fq * tn)
R, _ = ears.analyze(save("three_notes.wav", x), use_tagger=False)
p = R["pitch"]
check("three notes: range is 440-659", abs(p["min_hz"] - 440) < 3 and abs(p["max_hz"] - 659.25) < 4,
      f"{p['min_hz']:.1f}-{p['max_hz']:.1f} Hz")
check("three notes: gaps are not glide jumps", p["jumps"] == 0, f"jumps={p['jumps']}")

# ---- layer 2: the ear model, against the textbook reference signals
import psycho


def feel(x):
    f, tt_, P = ears.stft_db(x, SR)
    pt, f0, _ = ears.pitch_track(x, SR)
    et, er = ears.envelope(x, SR, 0.05)
    return psycho.analyze(x, SR, f, P, pt, f0, et, er)


t4 = np.arange(4 * SR) / SR
spl = lambda d: 10 ** ((d - psycho.SPL_AT_FS) / 20)
tone = lambda d, fc=1000: spl(d) * np.sin(2 * np.pi * fc * t4)
am = lambda fm: spl(60) * (1 + np.sin(2 * np.pi * fm * t4)) * np.sin(2 * np.pi * 1000 * t4) / np.sqrt(1.5)
a40, a50, a60 = feel(tone(40)), feel(tone(50)), feel(tone(60))
check("1 kHz at 40 dB = 1 sone", abs(a40["loudness_sone"] - 1) < 0.05, f"{a40['loudness_sone']:.2f}")
check("+10 dB roughly doubles loudness", 1.7 < a50["loudness_sone"] / a40["loudness_sone"] < 2.5
      and 1.7 < a60["loudness_sone"] / a50["loudness_sone"] < 2.5,
      f"x{a50['loudness_sone'] / a40['loudness_sone']:.2f}, x{a60['loudness_sone'] / a50['loudness_sone']:.2f}")
check("steady tone: not rough, not pulsing", a60["roughness"] < 0.03 and a60["fluctuation"] < 0.03,
      f"rough={a60['roughness']:.3f} fluct={a60['fluctuation']:.3f}")
r70, r4, r250 = feel(am(70)), feel(am(4)), feel(am(250))
check("70 Hz AM reads as rough (1.0)", abs(r70["roughness"] - 1) < 0.05 and r70["fluctuation"] < 0.1,
      f"rough={r70['roughness']:.2f} fluct={r70['fluctuation']:.2f}")
check("4 Hz AM reads as pulsing (1.0)", abs(r4["fluctuation"] - 1) < 0.05 and r4["roughness"] < 0.1,
      f"fluct={r4['fluctuation']:.2f} rough={r4['roughness']:.2f}")
check("250 Hz AM is neither", r250["roughness"] < 0.1 and r250["fluctuation"] < 0.1,
      f"rough={r250['roughness']:.2f} fluct={r250['fluctuation']:.2f}")
s250, s1k, s4k = feel(tone(60, 250)), a60, feel(tone(60, 4000))
check("sharpness rises with frequency", s250["sharpness_acum"] < 0.6 < s1k["sharpness_acum"] < 1.3 < s4k["sharpness_acum"],
      f"250Hz {s250['sharpness_acum']:.2f}, 1k {s1k['sharpness_acum']:.2f}, 4k {s4k['sharpness_acum']:.2f}")
xg, _ = glide(110, 330, 3.0)
hp = feel(fade(xg) * 0.25)["harmonics"]
want = [20 * np.log10(1 / k) for k in range(1, 7)]
check("harmonic recipe survives a glide", hp is not None and max(abs(a - b) for a, b in zip(hp["levels_db"][:6], want)) < 1.5,
      f"{hp['levels_db'][:6] if hp else None}")
st = sum(np.sin(2 * np.pi * 150 * k * t4) / k for k in range(1, 7)) * 0.1
hp = feel(fade(st))["harmonics"]
check("steady tone: absent harmonic reads absent", hp is not None and hp["levels_db"][6] < -40
      and max(abs(a - b) for a, b in zip(hp["levels_db"][:6], want)) < 1.0, f"H7 = {hp['levels_db'][6] if hp else None} dB")
sq = np.sign(np.sin(2 * np.pi * 110 * t4)) * 0.2
hs = feel(sq)
check("square wave reads as hollow", any(w["word"] == "hollow" for w in hs["words"]),
      f"odd-even {hs['harmonics']['odd_minus_even_db']:.1f} dB")
rj = np.random.default_rng(3)
cyc = []
while sum(len(c) for c in cyc) < 4 * SR:      # every cycle a slightly different length
    n_ = int(round(SR / 110 * (1 + 0.02 * rj.standard_normal())))
    cyc.append(np.sin(2 * np.pi * np.arange(n_) / n_))
jv = feel(0.2 * np.concatenate(cyc)).get("pitch_jitter_cents")
js = feel(tone(60, 110)).get("pitch_jitter_cents")
check("pitch wobble is measured", jv is not None and js is not None and jv > 3 and js < 0.3,
      f"2% cycle jitter reads {jv:.1f} cents, steady tone {js:.3f}")

# ---- layer 3: the recognition network (skipped if torch is not installed)
tg = ears.get_tagger()
if tg is None:
    print("SKIP  recognition tests (torch not installed)")
else:
    def first(x):
        r = tg.tag(x, SR)
        return [l for l, _ in tg.top(r["probs"], 3)], r
    l, _ = first(0.1 * np.random.default_rng(5).standard_normal(3 * SR))
    check("network: noise is called noise", any(w in l for w in ("White noise", "Static", "Noise", "Pink noise")), f"{l}")
    l, _ = first(0.3 * np.sin(2 * np.pi * 997 * np.arange(5 * SR) / SR))
    check("network: sine is called a sine", "Sine wave" in l, f"{l}")
    l, r1 = first(0.3 * np.sin(2 * np.pi * 997 * np.arange(int(1.2 * SR)) / SR))
    check("network: short clips do not blow up", "Sine wave" in l and (r1["probs"] > 0.9).sum() < 3,
          f"{l}, labels over 0.9: {(r1['probs'] > 0.9).sum()}")
    check("network: silence returns nothing", tg.tag(np.zeros(SR), SR) is None, "None")

fails = [r for r in results if not r[1]]
print(f"\n{len(results) - len(fails)}/{len(results)} passed")
sys.exit(1 if fails else 0)
