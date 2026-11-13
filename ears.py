#!/usr/bin/env python3
"""
ears.py - a way for something with no ears to audit audio.

It does not hear. It measures, draws, and writes down what it found, the way
a deaf engineer reads meters. Feed it a sound file; it produces:

  <name>.ears.png    waveform / spectrogram / pitch+brightness / average spectrum
  <name>.ears.json   every number it measured
  stdout             plain-language listening notes

Three layers, each closer to what a person would say about the sound:

  1. faults       clipping, clicks, dropouts, wrong pitch        (ears.py)
  2. perception   loudness, sharpness, roughness, harmonics, as
                  a model of the human ear would register them   (psycho.py)
  3. recognition  what a network trained on two million human-
                  labelled clips says the sound IS               (tagger.py)

Usage:
  python3 ears.py listen  sound.wav [--out DIR] [--fmin 30] [--fmax 2000]
  python3 ears.py compare mine.wav reference.wav [--out DIR]
  python3 ears.py judge   sound.wav --want Motorcycle --avoid Synthesizer
  python3 ears.py rate    sound.wav 8 [--note "good growl"]
  python3 ears.py taste   sound.wav
  python3 ears.py labels  [engine]

Needs numpy, scipy, matplotlib. ffmpeg is used for anything that isn't a
plain WAV. Layer 3 also needs torch + torchaudio; without them it is skipped.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy import signal
from scipy.io import wavfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import psycho

_TAGGER = None


def get_tagger():
    """The recognition network is optional (needs torch). None if unavailable."""
    global _TAGGER
    if _TAGGER is None:
        try:
            import tagger as _t
            _t._load()
            _TAGGER = _t
        except Exception as e:  # torch missing, weights unreachable, ...
            _TAGGER = False
            print(f"(recognition layer unavailable: {type(e).__name__}: {e})", file=sys.stderr)
    return _TAGGER or None


EPS = 1e-12
BANDS = [
    ("sub", 20, 60),
    ("bass", 60, 250),
    ("low-mid", 250, 500),
    ("mid", 500, 2000),
    ("high-mid", 2000, 6000),
    ("high", 6000, 20000),
]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), EPS))


def pdb(x):
    return 10.0 * np.log10(np.maximum(x, EPS))


def hz_to_note(f):
    if not f or f <= 0:
        return None
    m = 69 + 12 * np.log2(f / 440.0)
    n = int(round(m))
    cents = int(round((m - n) * 100))
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}{cents:+d}c"


# ----------------------------------------------------------------- loading

def _to_float(data):
    if data.dtype == np.uint8:
        return (data.astype(np.float64) - 128.0) / 128.0
    if data.dtype == np.int16:
        return data.astype(np.float64) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float64) / 2147483648.0
    return data.astype(np.float64)


def load(path):
    """Returns (sr, samples[n, channels] float64)."""
    try:
        if not path.lower().endswith(".wav"):
            raise ValueError("not wav")
        sr, data = wavfile.read(path)
    except Exception:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", path, "-vn",
                 "-c:a", "pcm_f32le", tmp.name],
                check=True)
            sr, data = wavfile.read(tmp.name)
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
    x = _to_float(data)
    if x.ndim == 1:
        x = x[:, None]
    return int(sr), x


# ---------------------------------------------------------------- loudness

def _k_weight(x, sr):
    """ITU-R BS.1770 K-weighting, coefficients derived for any sample rate."""
    # stage 1: high shelf
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = np.tan(np.pi * f0 / sr)
    Vh = 10.0 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b = [(Vh + Vb * K / Q + K * K) / a0, 2.0 * (K * K - Vh) / a0,
         (Vh - Vb * K / Q + K * K) / a0]
    a = [1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0]
    y = signal.lfilter(b, a, x, axis=0)
    # stage 2: high pass
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = np.tan(np.pi * f0 / sr)
    a0 = 1.0 + K / Q + K * K
    b = [1.0, -2.0, 1.0]
    a = [1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0]
    return signal.lfilter(b, a, y, axis=0)


def lufs(x, sr):
    """Integrated loudness with BS.1770 gating. None if too short/quiet."""
    block = int(0.4 * sr)
    hop = int(0.1 * sr)
    if x.shape[0] < block:
        return None
    y = _k_weight(x, sr)
    sq = np.sum(y * y, axis=1)  # channel weights 1.0 for L/R/mono
    c = np.concatenate([[0.0], np.cumsum(sq)])
    starts = np.arange(0, x.shape[0] - block + 1, hop)
    z = (c[starts + block] - c[starts]) / block
    l = -0.691 + pdb(z)
    keep = l > -70.0
    if not keep.any():
        return None
    rel = -0.691 + pdb(np.mean(z[keep])) - 10.0
    keep2 = keep & (l > rel)
    if not keep2.any():
        return None
    return float(-0.691 + pdb(np.mean(z[keep2])))


# ------------------------------------------------------------ time domain

def find_clipping(x, sr, thresh=0.999, min_run=3):
    """Runs of consecutive samples pinned at the rails."""
    events = []
    for ch in range(x.shape[1]):
        hot = np.abs(x[:, ch]) >= thresh
        if not hot.any():
            continue
        edges = np.diff(np.concatenate([[0], hot.astype(np.int8), [0]]))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for s, e in zip(starts, ends):
            if e - s >= min_run:
                events.append((s / sr, int(e - s), ch))
    events.sort()
    return events


def find_clicks(mono, sr, ratio=10.0, floor=0.01):
    """Single-sample discontinuities: a jump far bigger than everything
    around it. Catches envelope chops, phase jumps, buffer glitches and bad
    loop points. Works on the second difference, which is near zero for any
    smooth waveform and spikes at a step, so a small step riding on a loud
    low note still shows. Real drum hits ramp over many samples and do not
    trip this. Blind spots: a click smaller than the waveform's own edges,
    and a glitch repeating faster than ~15 times a second (that is a buzz,
    and shows up as pitch/spectrum instead)."""
    d = np.abs(np.diff(mono, n=2))
    if d.size < 256:
        return []
    frame = 64
    n = d.size // frame
    fm = d[: n * frame].reshape(n, frame)
    fmax = fm.max(axis=1)
    # neighbourhood level: median of frame maxima around each frame
    ctx = signal.medfilt(fmax, kernel_size=31)
    cand = np.flatnonzero((fmax > floor) & (fmax > ratio * (ctx + 1e-6)))
    clicks = []
    last = -1e9
    for f in cand:
        i = f * frame + int(np.argmax(fm[f]))
        lo, hi = max(0, i - 48), min(d.size, i + 49)
        around = np.concatenate([d[lo:max(lo, i - 3)], d[i + 4:hi]])
        if around.size == 0:
            continue
        local = np.percentile(around, 90)
        if d[i] > ratio * (local + 1e-6):
            # A saw or square wave has a hard edge every cycle. Those are the
            # waveform, not a fault: if several edges of similar size sit
            # within 100 ms either side, this is one of them.
            span = int(0.1 * sr / frame)
            nb = fmax[max(0, f - span): f + span + 1]
            if np.count_nonzero(nb >= 0.5 * d[i]) - 1 >= 3:
                continue
            t = (i + 1) / sr
            if t - last > 0.005:
                clicks.append((t, float(d[i])))
                last = t
    return clicks


def find_clicks_all(x, sr):
    """Run the click detector on every channel and merge hits within 5 ms."""
    hits = []
    for ch in range(x.shape[1]):
        hits += find_clicks(x[:, ch], sr)
    hits.sort()
    out = []
    for t, v in hits:
        if out and t - out[-1][0] < 0.005:
            if v > out[-1][1]:
                out[-1] = (out[-1][0], v)
        else:
            out.append((t, v))
    return out


def envelope(mono, sr, win=0.05):
    w = max(1, int(win * sr))
    n = mono.size // w
    if n == 0:
        return np.array([0.0]), np.array([float(np.sqrt(np.mean(mono ** 2)))])
    r = np.sqrt(np.mean(mono[: n * w].reshape(n, w) ** 2, axis=1))
    t = (np.arange(n) + 0.5) * w / sr
    return t, r


def find_silence(mono, sr, thresh_db=-60.0, min_gap=0.08):
    t, r = envelope(mono, sr, 0.02)
    quiet = db(r) < thresh_db
    total = float(np.mean(quiet)) if quiet.size else 1.0
    lead = trail = 0.0
    gaps = []
    if quiet.all():
        return {"all_silent": True, "silent_fraction": 1.0, "lead": mono.size / sr,
                "trail": 0.0, "gaps": []}
    first = int(np.argmax(~quiet))
    last = int(quiet.size - 1 - np.argmax(~quiet[::-1]))
    lead = first * 0.02
    trail = (quiet.size - 1 - last) * 0.02
    inner = quiet[first:last + 1]
    edges = np.diff(np.concatenate([[0], inner.astype(np.int8), [0]]))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        dur = (e - s) * 0.02
        if dur >= min_gap:
            gaps.append(((first + s) * 0.02, dur))
    return {"all_silent": False, "silent_fraction": total, "lead": lead,
            "trail": trail, "gaps": gaps}


# ------------------------------------------------------------------ pitch

def pitch_track(mono, sr, fmin=30.0, fmax=2000.0, thresh=0.15):
    """YIN. Returns (times, f0 with NaN where unvoiced, confidence)."""
    tau_max = int(sr / fmin) + 2
    tau_min = max(2, int(sr / fmax))
    half = tau_max + 2
    W = 2 * half
    hop = max(256, int(sr * 0.01))
    if mono.size < W + 1:
        return np.array([]), np.array([]), np.array([])
    nfr = 1 + (mono.size - W) // hop
    idx = np.arange(W)[None, :] + hop * np.arange(nfr)[:, None]
    nfft = 1 << int(np.ceil(np.log2(W + half)))
    f0 = np.full(nfr, np.nan)
    conf = np.zeros(nfr)
    CH = 512
    taus = np.arange(1, half)
    for c0 in range(0, nfr, CH):
        fr = mono[idx[c0:c0 + CH]]
        fr = fr - fr.mean(axis=1, keepdims=True)
        a = fr[:, :half]
        A = np.fft.rfft(a, nfft, axis=1)
        B = np.fft.rfft(fr, nfft, axis=1)
        r = np.fft.irfft(np.conj(A) * B, nfft, axis=1)[:, :half]
        cs = np.concatenate([np.zeros((fr.shape[0], 1)),
                             np.cumsum(fr * fr, axis=1)], axis=1)
        e0 = cs[:, half] - cs[:, 0]
        et = cs[:, half:2 * half] - cs[:, :half]  # energy of window at lag tau
        d = np.maximum(e0[:, None] + et - 2.0 * r, 0.0)
        cm = np.cumsum(d[:, 1:], axis=1)
        dn = np.ones_like(d)
        dn[:, 1:] = d[:, 1:] * taus[None, :] / (cm + EPS)
        for k in range(fr.shape[0]):
            if e0[k] / half < 1e-8:  # ~ -80 dBFS: nothing to pitch
                continue
            # frame straddles a note starting or stopping: one half is far
            # quieter than the other, and any "pitch" found there is junk
            e_late = cs[k, 2 * half] - cs[k, half]
            if min(e0[k], e_late) < 0.05 * max(e0[k], e_late):
                continue
            row = dn[k]
            seg = row[tau_min:tau_max]
            below = np.flatnonzero(seg < thresh)
            if below.size:
                t = below[0] + tau_min
                while t + 1 < tau_max and row[t + 1] < row[t]:
                    t += 1
            else:
                t = int(np.argmin(seg)) + tau_min
            v = row[t]
            if v > 0.35:
                continue
            if t <= tau_min or t >= tau_max - 1:
                continue  # pinned to the edge of the search range: not a real dip
            if 1 <= t < half - 1:
                y0, y1, y2 = row[t - 1], row[t], row[t + 1]
                den = y0 - 2 * y1 + y2
                shift = 0.5 * (y0 - y2) / den if abs(den) > EPS else 0.0
                t = t + float(np.clip(shift, -1, 1))
            f0[c0 + k] = sr / t
            conf[c0 + k] = 1.0 - v
    times = (np.arange(nfr) * hop + half) / sr
    # frames more than 40 dB under the loudest one are tails, not notes
    fr_e = np.array([np.mean(mono[i:i + W] ** 2) for i in range(0, nfr * hop, hop)])
    f0[fr_e < fr_e.max() * 1e-4] = np.nan
    # a pitch value that disagrees with both neighbours by 3+ semitones
    # (or has no voiced neighbour at all) is an estimator hiccup
    v = ~np.isnan(f0)
    lg = np.log2(np.where(v, f0, 1.0)) * 12
    drop = np.zeros(nfr, bool)
    for i in range(nfr):
        if not v[i]:
            continue
        ok = False
        for j in (i - 1, i + 1):
            if 0 <= j < nfr and v[j] and abs(lg[i] - lg[j]) < 3:
                ok = True
        drop[i] = not ok
    f0[drop] = np.nan
    conf[np.isnan(f0)] = 0
    return times, f0, conf


def pitch_jumps(times, f0, semis=1.5):
    """Places where pitch leaps between adjacent voiced frames. Fine for
    melodies, a bug for anything meant to glide."""
    out = []
    if f0.size < 3:
        return out
    v = ~np.isnan(f0)
    for i in range(1, f0.size):
        if v[i] and v[i - 1]:
            st = 12 * np.log2(f0[i] / f0[i - 1])
            if abs(st) >= semis:
                out.append((float(times[i]), float(f0[i - 1]), float(f0[i]), float(st)))
    return out


# --------------------------------------------------------------- spectral

def stft_db(mono, sr, nfft=None):
    if nfft is None:
        nfft = 4096 if sr >= 32000 else 2048
    nfft = min(nfft, max(256, 1 << int(np.floor(np.log2(max(mono.size, 256))))))
    f, t, Z = signal.stft(mono, sr, window="hann", nperseg=nfft,
                          noverlap=nfft - nfft // 4, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    return f, t, P


def spectral_features(f, P):
    tot = P.sum(axis=0) + EPS
    centroid = (f[:, None] * P).sum(axis=0) / tot
    cum = np.cumsum(P, axis=0) / tot
    roll = f[np.minimum((cum < 0.85).sum(axis=0), f.size - 1)]
    flat = np.exp(np.mean(np.log(P + EPS), axis=0)) / (np.mean(P, axis=0) + EPS)
    return centroid, roll, flat


def band_energy(f, avg):
    tot = avg.sum() + EPS
    out = {}
    for name, lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out[name] = float(avg[m].sum() / tot)
    return out


def third_octave(f, avg):
    centers = 1000.0 * 2.0 ** (np.arange(-17, 14) / 3.0)  # ~20 Hz .. 20 kHz
    cs, vals = [], []
    for c in centers:
        lo, hi = c * 2 ** (-1 / 6), c * 2 ** (1 / 6)
        m = (f >= lo) & (f < hi)
        if hi > f[-1] or not m.any():
            continue
        cs.append(c)
        vals.append(pdb(avg[m].sum()))
    return np.array(cs), np.array(vals)


def top_peaks(f, avg, n=8, min_hz=20.0):
    s = pdb(avg)
    ref = s.max()
    pk, props = signal.find_peaks(s, prominence=6.0, height=ref - 60.0)
    pk = [p for p in pk if f[p] >= min_hz]
    pk.sort(key=lambda p: -s[p])
    out = []
    for p in pk[:n]:
        # parabolic interpolation for a better frequency estimate
        if 0 < p < s.size - 1:
            y0, y1, y2 = s[p - 1], s[p], s[p + 1]
            den = y0 - 2 * y1 + y2
            sh = 0.5 * (y0 - y2) / den if abs(den) > EPS else 0.0
        else:
            sh = 0.0
        out.append((float(f[p] + sh * (f[1] - f[0])), float(s[p] - ref)))
    out.sort()
    return out


# --------------------------------------------------------------- analysis

def analyze(path, fmin=30.0, fmax=2000.0, use_tagger=True):
    sr, x = load(path)
    n, nch = x.shape
    mono = x.mean(axis=1)
    R = {"file": os.path.basename(path), "sample_rate": sr, "channels": nch,
         "samples": int(n), "duration_s": n / sr}
    if n and nch >= 2:
        # If the channels fight each other, the mono sum lies (it cancels).
        # Analyse the louder channel instead so the report describes real sound.
        e_sum = np.mean(mono ** 2)
        e_ch = np.mean(x ** 2, axis=0)
        if e_sum < 0.25 * e_ch.max():
            mono = x[:, int(np.argmax(e_ch))].copy()
            R["analysis_signal"] = "loudest channel (mono sum cancels)"
    if n == 0:
        R["empty"] = True
        return R, None

    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    R["peak_dbfs"] = float(db(peak))
    R["rms_dbfs"] = float(db(rms))
    R["crest_db"] = float(db(peak) - db(rms))
    R["lufs"] = lufs(x, sr)
    R["dc_offset"] = [float(v) for v in x.mean(axis=0)]
    R["non_finite"] = bool(not np.isfinite(x).all())

    clips = find_clipping(x, sr)
    R["clipping"] = {"events": len(clips),
                     "samples": int(sum(c[1] for c in clips)),
                     "times_s": [round(float(c[0]), 4) for c in clips[:40]]}
    clicks = find_clicks_all(x, sr)
    R["clicks"] = {"count": len(clicks),
                   "times_s": [round(float(c[0]), 4) for c in clicks[:40]],
                   "sizes": [round(c[1], 4) for c in clicks[:40]]}
    sil = find_silence(mono, sr)
    R["silence"] = {"all_silent": sil["all_silent"],
                    "silent_fraction": round(sil["silent_fraction"], 4),
                    "leading_s": round(sil["lead"], 3),
                    "trailing_s": round(sil["trail"], 3),
                    "dropouts": [(round(float(a), 3), round(float(b), 3)) for a, b in sil["gaps"][:40]]}

    et, er = envelope(mono, sr, 0.05)

    if nch >= 2:
        L, Rr = x[:, 0], x[:, 1]
        denom = np.sqrt(np.sum(L * L) * np.sum(Rr * Rr))
        corr = float(np.sum(L * Rr) / denom) if denom > EPS else 0.0
        mid, side = (L + Rr) / 2, (L - Rr) / 2
        R["stereo"] = {
            "correlation": corr,
            "balance_db": float(db(np.sqrt(np.mean(L ** 2))) - db(np.sqrt(np.mean(Rr ** 2)))),
            "side_vs_mid_db": float(db(np.sqrt(np.mean(side ** 2))) - db(np.sqrt(np.mean(mid ** 2)))),
        }

    f, t, P = stft_db(mono, sr)
    avg = P.mean(axis=1)
    cen, roll, flat = spectral_features(f, P)
    loud = P.sum(axis=0)
    act = loud > loud.max() * 1e-5 if loud.max() > 0 else np.zeros_like(loud, bool)
    if act.any():
        w = loud[act] / loud[act].sum()
        R["spectral"] = {
            "centroid_hz": float(np.sum(cen[act] * w)),
            "centroid_min_hz": float(cen[act].min()),
            "centroid_max_hz": float(cen[act].max()),
            "rolloff85_hz": float(np.sum(roll[act] * w)),
            "flatness": float(np.sum(flat[act] * w)),
        }
    else:
        R["spectral"] = None
    R["bands"] = band_energy(f, avg)
    top = f >= min(16000.0, 0.8 * sr / 2)
    R["energy_near_nyquist"] = float(avg[top].sum() / (avg.sum() + EPS))
    R["peaks"] = [{"hz": round(a, 1), "db_rel": round(b, 1), "note": hz_to_note(a)}
                  for a, b in top_peaks(f, avg)]

    pt, f0, conf = pitch_track(mono, sr, fmin, fmax)
    voiced = ~np.isnan(f0) if f0.size else np.array([], bool)
    if voiced.any():
        fv = f0[voiced]
        jumps = pitch_jumps(pt, f0)
        R["pitch"] = {
            "voiced_fraction": float(voiced.mean()),
            "median_hz": float(np.median(fv)), "median_note": hz_to_note(float(np.median(fv))),
            "min_hz": float(fv.min()), "max_hz": float(fv.max()),
            "start_hz": float(np.median(fv[: max(1, min(5, fv.size))])),
            "end_hz": float(np.median(fv[-max(1, min(5, fv.size)):])),
            "range_semitones": float(12 * np.log2(fv.max() / fv.min())),
            "jumps": len(jumps),
            "jump_list": [{"t": round(a, 3), "from_hz": round(b, 1), "to_hz": round(c, 1),
                           "semitones": round(d, 2)} for a, b, c, d in jumps[:30]],
        }
    else:
        R["pitch"] = {"voiced_fraction": 0.0}

    try:
        R["perception"] = psycho.analyze(mono, sr, f, P, pt, f0, et, er)
    except Exception as e:
        R["perception"] = None
        print(f"(perception layer failed: {type(e).__name__}: {e})", file=sys.stderr)

    R["sounds_like"] = None
    emb = None
    tg = get_tagger() if use_tagger else None
    if tg is not None and not sil["all_silent"]:
        tr = tg.tag(x, sr)
        if tr is not None:
            emb = tr["embedding"]
            R["sounds_like"] = {
                "top": [{"label": l, "p": round(v, 3)} for l, v in tg.top(tr["probs"], 8)],
                "timeline": [{"from_s": round(a, 1), "to_s": round(b, 1),
                              "top": [{"label": l, "p": round(v, 3)} for l, v in tg.top(p, 3)]}
                             for a, b, p in tr["timeline"]] if len(tr["timeline"]) > 1 else [],
            }
            R["_probs"] = tr["probs"]

    R["notes"] = make_notes(R)
    arrays = dict(emb=emb, sr=sr, x=x, mono=mono, f=f, t=t, P=P, avg=avg, cen=cen, act=act,
                  pt=pt, f0=f0, et=et, er=er, clips=clips, clicks=clicks,
                  gaps=sil["gaps"])
    return R, arrays


def make_notes(R):
    """Plain-language verdicts. Each one is backed by a number in the report."""
    N = []

    def add(level, text):
        N.append({"level": level, "text": text})

    if R.get("non_finite"):
        add("BROKEN", "File contains NaN/Inf samples. Something divided by zero.")
    s = R["silence"]
    if s["all_silent"]:
        add("BROKEN", "It is silent. Entirely. Nothing above -60 dBFS.")
        return N
    c = R["clipping"]
    if c["events"]:
        add("BAD", f"Clipping: {c['events']} run(s), {c['samples']} samples pinned at full scale. "
                   f"First at {c['times_s'][0]}s. That is audible distortion.")
    elif R["peak_dbfs"] > -0.3:
        add("WARN", f"Peak at {R['peak_dbfs']:.2f} dBFS. Not clipping, but no headroom at all.")
    k = R["clicks"]
    if k["count"]:
        add("BAD", f"{k['count']} click(s)/discontinuities, first at {k['times_s'][0]}s. "
                   "Usually a gain or phase jump with no ramp, or a bad loop point.")
    if s["dropouts"]:
        a, b = s["dropouts"][0]
        add("WARN", f"{len(s['dropouts'])} dropout(s) to silence mid-file, first at {a}s for {b}s.")
    if s["leading_s"] > 0.25:
        add("INFO", f"{s['leading_s']}s of dead air before anything happens.")
    if s["trailing_s"] > 1.0:
        add("INFO", f"{s['trailing_s']}s of silence at the end.")
    dc = max(abs(v) for v in R["dc_offset"])
    if dc > 0.01:
        add("WARN", f"DC offset of {dc:.3f}. Wastes headroom and clicks on start/stop.")
    if R["lufs"] is not None:
        l = R["lufs"]
        if l < -30:
            add("WARN", f"Quiet: {l:.1f} LUFS. Most music sits around -14 to -9.")
        elif l > -8:
            add("WARN", f"Very loud: {l:.1f} LUFS.")
        else:
            add("INFO", f"Loudness {l:.1f} LUFS.")
    if R["crest_db"] < 3.5 and not c["events"]:
        add("INFO", f"Crest factor {R['crest_db']:.1f} dB: basically a flat-out square or heavily squashed.")
    b = R["bands"]
    if b["sub"] + b["bass"] > 0.9:
        add("INFO", "Nearly all energy is below 250 Hz. Will vanish on a phone speaker.")
    if b["high-mid"] + b["high"] > 0.6:
        add("INFO", "More than 60% of energy is above 2 kHz. Thin, bright, or hissy.")
    if b["sub"] > 0.5:
        add("WARN", "Over half the energy is sub-60 Hz rumble.")
    if R["energy_near_nyquist"] > 0.05:
        add("WARN", f"{R['energy_near_nyquist'] * 100:.0f}% of energy is crammed near the top of the "
                    "spectrum. With a synth that often means aliasing.")
    sp = R.get("spectral")
    if sp:
        if sp["flatness"] > 0.3:
            add("INFO", f"Noise-like (flatness {sp['flatness']:.2f}). Reads as hiss/static, not a tone.")
        elif sp["flatness"] < 0.01:
            add("INFO", f"Strongly tonal (flatness {sp['flatness']:.4f}).")
    p = R["pitch"]
    if p.get("voiced_fraction", 0) > 0.2:
        add("INFO", f"Pitch: median {p['median_hz']:.1f} Hz ({p['median_note']}), "
                    f"range {p['min_hz']:.1f}-{p['max_hz']:.1f} Hz "
                    f"({p['range_semitones']:.1f} semitones), "
                    f"starts ~{p['start_hz']:.1f} Hz, ends ~{p['end_hz']:.1f} Hz.")
        if p["jumps"]:
            j = p["jump_list"][0]
            add("WARN", f"{p['jumps']} pitch jump(s) of 1.5+ semitones between adjacent frames, "
                        f"first at {j['t']}s ({j['from_hz']} -> {j['to_hz']} Hz). "
                        "Normal for a melody. A bug for anything that is supposed to glide.")
    else:
        add("INFO", "No stable pitch found. Noise, percussion, or dense polyphony.")
    st = R.get("stereo")
    if st:
        if st["correlation"] < -0.3:
            add("BAD", f"Stereo correlation {st['correlation']:.2f}: channels are out of phase. "
                       "It will thin out or disappear in mono.")
        elif st["correlation"] > 0.999:
            add("INFO", "Both channels are identical. This is mono wearing a stereo costume.")
        if abs(st["balance_db"]) > 3:
            side = "left" if st["balance_db"] > 0 else "right"
            add("WARN", f"Leans {abs(st['balance_db']):.1f} dB to the {side}.")
    if not any(n["level"] in ("BAD", "BROKEN") for n in N):
        add("OK", "No clipping, clicks, or phase disasters found. Not broken. Whether it is "
                  "the RIGHT sound is what the sections above and `judge` are for.")
    return N


# --------------------------------------------------------------- pictures

def _spec_axes(ax, f, t, P, sr, vmax=None):
    S = pdb(P)
    if vmax is None:
        vmax = S.max()
    lo = 20.0
    m = f >= lo
    mesh = ax.pcolormesh(t, f[m], S[m], shading="auto", cmap="magma",
                         vmin=vmax - 90, vmax=vmax, rasterized=True)
    ax.set_yscale("log")
    ax.set_ylim(lo, sr / 2)
    ticks = [t_ for t_ in (30, 60, 125, 250, 500, 1000, 2000, 4000, 8000, 16000) if t_ < sr / 2]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(v) if v < 1000 else f"{v // 1000}k" for v in ticks])
    ax.set_ylabel("Hz")
    return mesh


def draw(R, A, out_png):
    sr, mono = A["sr"], A["mono"]
    dur = mono.size / sr
    fig, axs = plt.subplots(4, 1, figsize=(13, 13),
                            gridspec_kw={"height_ratios": [1.1, 2.4, 1.3, 1.5]})
    fig.suptitle(f"{R['file']}   {dur:.2f}s  {sr} Hz  {R['channels']}ch   "
                 f"peak {R['peak_dbfs']:.1f} dBFS  rms {R['rms_dbfs']:.1f} dBFS  "
                 f"LUFS {R['lufs']:.1f}" if R["lufs"] is not None else
                 f"{R['file']}   {dur:.2f}s  {sr} Hz  {R['channels']}ch   "
                 f"peak {R['peak_dbfs']:.1f} dBFS  rms {R['rms_dbfs']:.1f} dBFS",
                 fontsize=11)

    ax = axs[0]
    # min/max decimation so short clicks stay visible
    cols = 2600
    step = max(1, mono.size // cols)
    n = mono.size // step
    blk = mono[: n * step].reshape(n, step)
    tt = (np.arange(n) + 0.5) * step / sr
    ax.fill_between(tt, blk.min(axis=1), blk.max(axis=1), color="#3b6ea5", lw=0)
    ax.plot(A["et"], A["er"], color="#f0a030", lw=1.0, label="RMS")
    for tclip, _, _ in A["clips"][:400]:
        ax.axvline(tclip, color="red", alpha=0.6, lw=0.8)
    for tk, _ in A["clicks"][:400]:
        ax.axvline(tk, color="#00c8c8", alpha=0.9, lw=0.9, ls="--")
    for a, b in A["gaps"][:200]:
        ax.axvspan(a, a + b, color="grey", alpha=0.3)
    ax.axhline(1, color="red", lw=0.5, alpha=0.5)
    ax.axhline(-1, color="red", lw=0.5, alpha=0.5)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlim(0, dur)
    ax.set_ylabel("amplitude")
    ax.set_title("waveform   red = clipping   cyan dashed = click   grey = dropout",
                 fontsize=9, loc="left")

    ax = axs[1]
    _spec_axes(ax, A["f"], A["t"], A["P"], sr)
    ax.set_xlim(0, dur)
    ax.set_title("spectrogram (log frequency, 90 dB range)", fontsize=9, loc="left")

    ax = axs[2]
    if A["f0"].size and (~np.isnan(A["f0"])).any():
        ax.plot(A["pt"], A["f0"], ".", ms=2.5, color="#2ca02c", label="pitch (f0)")
    cen = np.where(A["act"], A["cen"], np.nan)
    ax.plot(A["t"], cen, color="#d62728", lw=1.0, alpha=0.8, label="brightness (centroid)")
    ax.set_yscale("log")
    ax.set_xlim(0, dur)
    ax.set_ylabel("Hz")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("seconds")
    ax.set_title("pitch and brightness over time", fontsize=9, loc="left")

    ax = axs[3]
    f, avg = A["f"], A["avg"]
    m = f >= 20
    s = pdb(avg[m])
    s = s - s.max()
    ax.semilogx(f[m], s, color="#444", lw=0.9)
    for pk in R["peaks"]:
        ax.annotate(f"{pk['hz']:.0f}", (pk["hz"], pk["db_rel"]), fontsize=7,
                    xytext=(0, 4), textcoords="offset points", ha="center", color="#b00")
    ax.set_xlim(20, sr / 2)
    ax.set_ylim(-90, 5)
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB rel. to peak")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title("average spectrum (labelled peaks in Hz)", fontsize=9, loc="left")

    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------- text

def spark(vals, lo=-60.0, hi=0.0):
    chars = " .:-=+*#%@"
    out = []
    for v in vals:
        k = int(np.clip((v - lo) / (hi - lo), 0, 0.999) * len(chars))
        out.append(chars[k])
    return "".join(out)


def text_report(R, A=None):
    L = []
    L.append(f"=== {R['file']} ===")
    L.append(f"{R['duration_s']:.3f}s, {R['sample_rate']} Hz, {R['channels']} channel(s)")
    if R.get("empty"):
        L.append("EMPTY FILE")
        return "\n".join(L)
    lu = f"{R['lufs']:.1f}" if R["lufs"] is not None else "n/a"
    L.append(f"peak {R['peak_dbfs']:.2f} dBFS | rms {R['rms_dbfs']:.2f} dBFS | "
             f"crest {R['crest_db']:.1f} dB | loudness {lu} LUFS")
    if A is not None:
        et, er = A["et"], A["er"]
        k = max(1, er.size // 64)
        n = er.size // k
        if n:
            seg = db(er[: n * k].reshape(n, k).mean(axis=1))
            L.append(f"level over time  |{spark(seg)}|  (' '=-60 dBFS, '@'=0)")
    L.append("energy by band:  " + "  ".join(
        f"{n} {R['bands'][n] * 100:.0f}%" for n, _, _ in BANDS))
    if R.get("spectral"):
        sp = R["spectral"]
        L.append(f"brightness: centroid {sp['centroid_hz']:.0f} Hz "
                 f"(moves {sp['centroid_min_hz']:.0f}-{sp['centroid_max_hz']:.0f}), "
                 f"85% rolloff {sp['rolloff85_hz']:.0f} Hz, flatness {sp['flatness']:.4f}")
    if R["peaks"]:
        L.append("strongest partials: " + ", ".join(
            f"{p['hz']:.0f}Hz({p['db_rel']:+.0f}dB)" for p in R["peaks"]))
    if R.get("stereo"):
        st = R["stereo"]
        L.append(f"stereo: correlation {st['correlation']:.3f}, balance {st['balance_db']:+.1f} dB (L-R), "
                 f"side/mid {st['side_vs_mid_db']:.1f} dB")
    pc = R.get("perception")
    if pc:
        L.append("")
        L.append("HOW IT WOULD FEEL (ear model, approximate)")
        ph = f", ~{pc['loudness_phon']:.0f} phon" if pc.get("loudness_phon") else ""
        L.append(f"  loudness {pc['loudness_sone']:.1f} sone{ph} if full scale plays at {psycho.SPL_AT_FS:.0f} dB SPL")
        ro = f"{pc['roughness']:.2f}" if pc["roughness"] is not None else "n/a"
        fl = f"{pc['fluctuation']:.2f}" if pc["fluctuation"] is not None else "n/a"
        L.append(f"  sharpness {pc['sharpness_acum']:.2f} acum (1 = dull-ish tone, 3 = hiss) | "
                 f"roughness {ro} | pulsing {fl}   (1.0 = fully modulated reference tone)")
        h = pc.get("harmonics")
        if h:
            lv = " ".join(f"{v:+.0f}" for v in h["levels_db"][:12])
            tilt = f"{h['tilt_db_per_octave']:.1f} dB/oct" if h["tilt_db_per_octave"] is not None else "n/a"
            L.append(f"  harmonic recipe H1..H12 (dB): {lv}")
            L.append(f"  strongest H{h['strongest']}, odd-even {h['odd_minus_even_db']:+.1f} dB, "
                     f"tilt {tilt}, non-harmonic share {h['noise_share'] * 100:.0f}%")
        j, sh = pc.get("pitch_jitter_cents"), pc.get("level_shimmer_db")
        if j is not None or sh is not None:
            L.append("  steadiness: " + ", ".join(
                x for x in (f"pitch jitter {j:.2f} cents" if j is not None else None,
                            f"level shimmer {sh:.2f} dB" if sh is not None else None) if x))
        if pc["words"]:
            L.append("  in words: " + "; ".join(f"{w['word']} ({w['because']})" for w in pc["words"]))
    sl = R.get("sounds_like")
    if sl:
        L.append("")
        L.append("WHAT IT SOUNDS LIKE (network trained on human-labelled audio)")
        L.append("  " + ", ".join(f"{t['label']} {t['p']:.2f}" for t in sl["top"]))
        for seg in sl["timeline"]:
            L.append(f"    {seg['from_s']:>5.1f}-{seg['to_s']:<5.1f}s  " +
                     ", ".join(f"{t['label']} {t['p']:.2f}" for t in seg["top"]))
    L.append("")
    L.append("LISTENING NOTES")
    for n in R["notes"]:
        L.append(f"  [{n['level']:<6}] {n['text']}")
    return "\n".join(L)


# ---------------------------------------------------------------- compare

def compare(pa, pb, outdir, fmin, fmax):
    Ra, Aa = analyze(pa, fmin, fmax)
    Rb, Ab = analyze(pb, fmin, fmax)
    L = [f"=== COMPARE  A: {Ra['file']}   vs   B: {Rb['file']} ===",
         "(differences are A minus B; B is treated as the reference)", ""]

    def row(label, a, b, unit="", fmt="{:.1f}"):
        if a is None or b is None:
            L.append(f"{label:<22} A {a}  B {b}")
            return
        L.append(f"{label:<22} A {fmt.format(a):>9}{unit}   B {fmt.format(b):>9}{unit}   "
                 f"diff {fmt.format(a - b):>8}{unit}")

    row("duration", Ra["duration_s"], Rb["duration_s"], " s", "{:.2f}")
    row("loudness", Ra["lufs"], Rb["lufs"], " LUFS")
    row("peak", Ra["peak_dbfs"], Rb["peak_dbfs"], " dBFS")
    row("crest factor", Ra["crest_db"], Rb["crest_db"], " dB")
    if Ra.get("spectral") and Rb.get("spectral"):
        row("brightness (centroid)", Ra["spectral"]["centroid_hz"], Rb["spectral"]["centroid_hz"], " Hz", "{:.0f}")
        row("flatness (noisiness)", Ra["spectral"]["flatness"], Rb["spectral"]["flatness"], "", "{:.4f}")
    if "median_hz" in Ra["pitch"] and "median_hz" in Rb["pitch"]:
        row("median pitch", Ra["pitch"]["median_hz"], Rb["pitch"]["median_hz"], " Hz")
        row("pitch range", Ra["pitch"]["range_semitones"], Rb["pitch"]["range_semitones"], " st")
    pa_, pb_ = Ra.get("perception"), Rb.get("perception")
    verdicts = []
    if pa_ and pb_:
        row("sharpness", pa_["sharpness_acum"], pb_["sharpness_acum"], " acum", "{:.2f}")
        row("roughness", pa_["roughness"], pb_["roughness"], "", "{:.2f}")
        row("pulsing", pa_["fluctuation"], pb_["fluctuation"], "", "{:.2f}")
        row("pitch jitter", pa_.get("pitch_jitter_cents"), pb_.get("pitch_jitter_cents"), " cents", "{:.2f}")
        ha, hb = pa_.get("harmonics"), pb_.get("harmonics")
        if ha and hb:
            row("non-harmonic share", ha["noise_share"] * 100, hb["noise_share"] * 100, " %", "{:.0f}")
            L.append("")
            L.append("harmonic recipe, dB relative to each sound's strongest harmonic:")
            L.append("        " + " ".join(f"H{k:<3}" for k in range(1, 13)))
            L.append("  A     " + " ".join(f"{v:>+4.0f}" for v in ha["levels_db"][:12]))
            L.append("  B     " + " ".join(f"{v:>+4.0f}" for v in hb["levels_db"][:12]))
            L.append("  A-B   " + " ".join(f"{a - b:>+4.0f}" for a, b in zip(ha["levels_db"][:12], hb["levels_db"][:12])))
            if abs(ha["noise_share"] - hb["noise_share"]) > 0.15:
                d = ha["noise_share"] - hb["noise_share"]
                verdicts.append(f"A is {'noisier' if d > 0 else 'cleaner'} than B: "
                                f"{ha['noise_share'] * 100:.0f}% vs {hb['noise_share'] * 100:.0f}% non-harmonic energy")
        if pa_["roughness"] is not None and pb_["roughness"] is not None and abs(pa_["roughness"] - pb_["roughness"]) > 0.2:
            d = pa_["roughness"] - pb_["roughness"]
            verdicts.append(f"A is {'rougher' if d > 0 else 'smoother'} than B (roughness {pa_['roughness']:.2f} vs {pb_['roughness']:.2f})")
        if abs(pa_["sharpness_acum"] - pb_["sharpness_acum"]) > 0.3:
            d = pa_["sharpness_acum"] - pb_["sharpness_acum"]
            verdicts.append(f"A is {'sharper' if d > 0 else 'duller'} than B (sharpness {pa_['sharpness_acum']:.2f} vs {pb_['sharpness_acum']:.2f} acum)")
    sim = None
    tg = get_tagger()
    if tg is not None and Aa is not None and Ab is not None and Aa.get("emb") is not None and Ab.get("emb") is not None:
        sim = tg.similarity(Aa["emb"], Ab["emb"])
        L.append("")
        kind = ("same kind of sound" if sim >= 0.6 else "borderline: low end of same-kind" if sim >= 0.45
                else "different kinds of sound")
        L.append(f"resemblance to the network: {sim:.2f} -> {kind}")
        L.append("  (measured on 40 real recordings: two clips of the same thing score 0.47-0.81, median 0.67;"
                 " different things 0.20-0.43, median 0.30)")
        if sim < 0.45:
            verdicts.append(f"to the network these are different kinds of sound (resemblance {sim:.2f})")
        L.append("  A sounds like: " + ", ".join(f"{t['label']} {t['p']:.2f}" for t in Ra["sounds_like"]["top"][:5]))
        L.append("  B sounds like: " + ", ".join(f"{t['label']} {t['p']:.2f}" for t in Rb["sounds_like"]["top"][:5]))
    L.append("")
    L.append("share of energy by band (loudness-independent):")
    for n, lo, hi in BANDS:
        a, b = Ra["bands"][n], Rb["bands"][n]
        d = pdb(a) - pdb(b)
        L.append(f"  {n:<9} A {a * 100:5.1f}%   B {b * 100:5.1f}%   {d:+6.1f} dB")
        if max(a, b) > 0.02 and abs(d) >= 3:
            verdicts.append(f"A has {'more' if d > 0 else 'less'} {n} ({lo}-{hi} Hz) than B by {abs(d):.1f} dB")
    ca, va = third_octave(Aa["f"], Aa["avg"])
    cb, vb = third_octave(Ab["f"], Ab["avg"])
    common = np.intersect1d(np.round(ca, 3), np.round(cb, 3))
    ia = np.isin(np.round(ca, 3), common)
    ib = np.isin(np.round(cb, 3), common)
    va_n = va[ia] - pdb(np.sum(10 ** (va[ia] / 10)))
    vb_n = vb[ib] - pdb(np.sum(10 ** (vb[ib] / 10)))
    diff = va_n - vb_n
    L.append("")
    L.append("VERDICT")
    if not verdicts:
        L.append("  Tonal balance is within 3 dB of the reference in every band that matters.")
    for v in verdicts:
        L.append("  " + v)
    if Ra["lufs"] is not None and Rb["lufs"] is not None and abs(Ra["lufs"] - Rb["lufs"]) >= 2:
        d = Ra["lufs"] - Rb["lufs"]
        L.append(f"  A is {abs(d):.1f} LU {'louder' if d > 0 else 'quieter'} than B")
    for tag, Rx in (("A", Ra), ("B", Rb)):
        for n in Rx["notes"]:
            if n["level"] in ("BAD", "BROKEN"):
                L.append(f"  {tag}: {n['text']}")

    base = os.path.join(outdir, f"{os.path.splitext(Ra['file'])[0]}_vs_{os.path.splitext(Rb['file'])[0]}")
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.3, 1.1])
    vmax = max(pdb(Aa["P"]).max(), pdb(Ab["P"]).max())
    for col, (Rx, Ax, tag) in enumerate(((Ra, Aa, "A"), (Rb, Ab, "B"))):
        ax = fig.add_subplot(gs[0, col])
        _spec_axes(ax, Ax["f"], Ax["t"], Ax["P"], Ax["sr"], vmax)
        ax.set_title(f"{tag}: {Rx['file']}", fontsize=9, loc="left")
        ax.set_xlabel("seconds")
    ax = fig.add_subplot(gs[1, :])
    ax.semilogx(ca[ia], va_n, "o-", ms=3, label=f"A {Ra['file']}", color="#d62728")
    ax.semilogx(cb[ib], vb_n, "o-", ms=3, label=f"B {Rb['file']}", color="#1f77b4")
    ax.set_ylabel("dB (level-matched)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("third-octave spectrum, loudness removed", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[2, :])
    ax.bar(np.arange(diff.size), diff, color=np.where(diff > 0, "#d62728", "#1f77b4"))
    ax.set_xticks(np.arange(diff.size))
    ax.set_xticklabels([f"{c:.0f}" if c < 1000 else f"{c / 1000:.1f}k" for c in ca[ia]],
                       rotation=60, fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("A minus B (dB)")
    ax.set_title("where A differs from the reference", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(base + ".compare.png", dpi=110)
    plt.close(fig)
    with open(base + ".compare.json", "w") as fh:
        json.dump({"a": _clean(Ra), "b": _clean(Rb), "resemblance": sim,
                   "third_octave_hz": [float(c) for c in ca[ia]],
                   "third_octave_diff_db": [float(d) for d in diff],
                   "verdict": verdicts}, fh, indent=1)
    L.append("")
    L.append(f"picture: {base}.compare.png")
    return "\n".join(L)


def _clean(R):
    return {k: v for k, v in R.items() if not k.startswith("_")}


# ------------------------------------------------------------------ judge

def judge(path, want, avoid, fmin, fmax):
    """Does it sound like what it is supposed to be? Pass/fail on labels."""
    tg = get_tagger()
    if tg is None:
        print("judge needs the recognition layer (pip install torch torchaudio).")
        return 3
    R, A = analyze(path, fmin, fmax)
    if R.get("sounds_like") is None:
        print("Nothing to judge: the file is silent.")
        return 2
    probs = R["_probs"]
    lab = tg.labels()

    def resolve(qs):
        out = []
        for q in qs:
            idx = tg.find_labels(q)
            if not idx:
                print(f"  no label matches '{q}'. Try: python3 ears.py labels {q.split()[0]}")
                continue
            out += [(lab[i], float(probs[i])) for i in idx]
        return out

    print(f"=== JUDGE {R['file']} ===")
    print("it sounds most like: " + ", ".join(f"{t['label']} {t['p']:.2f}" for t in R["sounds_like"]["top"]))
    W, V = resolve(want), resolve(avoid)
    if W:
        print("wanted:")
        for l, p in sorted(W, key=lambda t: -t[1]):
            print(f"  {p:.2f}  {l}")
    if V:
        print("to avoid:")
        for l, p in sorted(V, key=lambda t: -t[1]):
            print(f"  {p:.2f}  {l}")
    bw = max((p for _, p in W), default=None)
    bv = max((p for _, p in V), default=0.0)
    top_p = R["sounds_like"]["top"][0]["p"] if R["sounds_like"]["top"] else 0.0
    ok = None
    if bw is not None:
        ok = bw >= 0.2 and bw > bv
        if ok:
            print(f"VERDICT: PASS. Best wanted label {bw:.2f}, beats everything to avoid ({bv:.2f}).")
        elif bw <= bv:
            print(f"VERDICT: FAIL. It sounds more like what you wanted to avoid ({bv:.2f}) than what you wanted ({bw:.2f}).")
        else:
            print(f"VERDICT: FAIL. Best wanted label is only {bw:.2f} (need 0.20). "
                  f"The network's first guess is '{R['sounds_like']['top'][0]['label']}' at {top_p:.2f}.")
    pc = R.get("perception")
    if pc and pc["words"]:
        print("character: " + "; ".join(f"{w['word']} ({w['because']})" for w in pc["words"]))
    return 0 if ok or ok is None else 2


# ------------------------------------------------------------------ taste
# The one thing no model knows is what YOU like. So: you rate sounds, the
# ratings are stored with the network's fingerprint of each sound, and a new
# sound is scored by the rated sounds it most resembles.

def _book_load(book):
    if os.path.exists(book):
        with open(book) as fh:
            return json.load(fh)
    return {"ratings": []}


def rate(path, score, note, book, fmin, fmax):
    tg = get_tagger()
    if tg is None:
        print("rate needs the recognition layer (pip install torch torchaudio).")
        return 3
    R, A = analyze(path, fmin, fmax)
    if A is None or A.get("emb") is None:
        print("Cannot rate a silent file.")
        return 2
    B = _book_load(book)
    B["ratings"] = [r for r in B["ratings"] if r["file"] != os.path.abspath(path)]
    pc = R.get("perception") or {}
    B["ratings"].append({
        "file": os.path.abspath(path), "score": float(score), "note": note or "",
        "embedding": [round(float(v), 5) for v in A["emb"]],
        "features": {"sharpness": pc.get("sharpness_acum"), "roughness": pc.get("roughness"),
                     "pulsing": pc.get("fluctuation"),
                     "noise_share": (pc.get("harmonics") or {}).get("noise_share"),
                     "centroid_hz": (R.get("spectral") or {}).get("centroid_hz"),
                     "lufs": R.get("lufs")},
    })
    with open(book, "w") as fh:
        json.dump(B, fh)
    print(f"logged {os.path.basename(path)} = {score}/10. {len(B['ratings'])} rating(s) in {book}")
    return 0


def taste(path, book, fmin, fmax):
    tg = get_tagger()
    if tg is None:
        print("taste needs the recognition layer (pip install torch torchaudio).")
        return 3
    B = _book_load(book)
    rs = B["ratings"]
    if len(rs) < 4:
        print(f"Only {len(rs)} rating(s) in {book}. Need at least 4, spread across good and bad, "
              "before a guess means anything. Use: python3 ears.py rate <file> <0-10>")
        return 3
    R, A = analyze(path, fmin, fmax)
    if A is None or A.get("emb") is None:
        print("Cannot score a silent file.")
        return 2
    sims = np.array([tg.similarity(A["emb"], r["embedding"]) for r in rs])
    scores = np.array([r["score"] for r in rs])
    order = np.argsort(sims)[::-1]
    k = order[: min(5, len(rs))]
    w = np.maximum(sims[k] - sims.min(), 1e-3) ** 2
    pred = float(np.sum(w * scores[k]) / np.sum(w))
    print(f"=== TASTE {R['file']} ===")
    print(f"predicted score: {pred:.1f}/10   (from {len(rs)} ratings; a guess, and a weak one under ~15 ratings)")
    print("closest things you have rated:")
    for i in k:
        note = f"  \"{rs[i]['note']}\"" if rs[i]["note"] else ""
        print(f"  resemblance {sims[i]:.2f}  you gave {rs[i]['score']:g}/10  {os.path.basename(rs[i]['file'])}{note}")
    if sims[order[0]] < 0.45:
        print("warning: nothing you have rated sounds much like this. The prediction is mostly noise.")
    # which measurable trait tracks your scores
    cors = []
    for key in ("sharpness", "roughness", "pulsing", "noise_share", "centroid_hz", "lufs"):
        v = np.array([r["features"].get(key) if r["features"].get(key) is not None else np.nan for r in rs], float)
        m = ~np.isnan(v)
        if m.sum() >= 4 and np.std(v[m]) > 0 and np.std(scores[m]) > 0:
            cors.append((float(np.corrcoef(v[m], scores[m])[0, 1]), key))
    if cors and len(rs) >= 6:
        cors.sort(key=lambda t: -abs(t[0]))
        print("what your ratings track so far: " + ", ".join(
            f"{'more' if c > 0 else 'less'} {k_} (r={c:+.2f})" for c, k_ in cors[:3]))
    return 0


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Audit audio without ears.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("listen", help="analyze one file")
    a.add_argument("file")
    a.add_argument("--no-tagger", action="store_true", help="skip the recognition network")
    b = sub.add_parser("compare", help="compare a file against a reference")
    b.add_argument("file")
    b.add_argument("reference")
    j = sub.add_parser("judge", help="does it sound like what it should?")
    j.add_argument("file")
    j.add_argument("--want", action="append", default=[], help="label it should sound like (repeatable)")
    j.add_argument("--avoid", action="append", default=[], help="label it should NOT sound like (repeatable)")
    r = sub.add_parser("rate", help="log your own 0-10 rating of a sound")
    r.add_argument("file")
    r.add_argument("score", type=float)
    r.add_argument("--note", default="")
    t = sub.add_parser("taste", help="predict your rating from sounds you have rated")
    t.add_argument("file")
    lb = sub.add_parser("labels", help="list the 527 things the network knows, optionally filtered")
    lb.add_argument("filter", nargs="?", default="")
    for p in (a, b):
        p.add_argument("--out", default=None, help="output directory (default: next to the file)")
    for p in (a, b, j, r, t):
        p.add_argument("--fmin", type=float, default=30.0, help="lowest pitch to look for")
        p.add_argument("--fmax", type=float, default=2000.0, help="highest pitch to look for")
    for p in (r, t):
        p.add_argument("--book", default="ears_taste.json", help="where ratings are kept")
    args = ap.parse_args()

    if args.cmd == "labels":
        tg = get_tagger()
        if tg is None:
            sys.exit(3)
        q = args.filter.lower()
        for l in tg.labels():
            if q in l.lower():
                print(l)
        return
    if args.cmd == "judge":
        sys.exit(judge(args.file, args.want, args.avoid, args.fmin, args.fmax))
    if args.cmd == "rate":
        sys.exit(rate(args.file, args.score, args.note, args.book, args.fmin, args.fmax))
    if args.cmd == "taste":
        sys.exit(taste(args.file, args.book, args.fmin, args.fmax))

    outdir = args.out or os.path.dirname(os.path.abspath(args.file))
    os.makedirs(outdir, exist_ok=True)
    if args.cmd == "listen":
        R, A = analyze(args.file, args.fmin, args.fmax, use_tagger=not args.no_tagger)
        base = os.path.join(outdir, os.path.splitext(os.path.basename(args.file))[0])
        if A is not None:
            draw(R, A, base + ".ears.png")
        with open(base + ".ears.json", "w") as fh:
            json.dump(_clean(R), fh, indent=1)
        print(text_report(R, A))
        if A is not None:
            print(f"\npicture: {base}.ears.png")
        bad = any(n["level"] in ("BAD", "BROKEN") for n in R.get("notes", []))
        sys.exit(2 if bad else 0)
    else:
        print(compare(args.file, args.reference, outdir, args.fmin, args.fmax))


if __name__ == "__main__":
    main()
