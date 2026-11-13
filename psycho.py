#!/usr/bin/env python3
"""
psycho.py - a rough model of the human ear, so numbers come out in units of
"how it feels" instead of "how big the wave is".

Everything here is a simplified version of standard psychoacoustics (Zwicker
loudness, DIN 45692 sharpness, modulation-based roughness and fluctuation
strength). Each is calibrated against the textbook reference signal for its
unit and checked in selftest.py, but these are approximations: trust the
direction and the rough size, not the second decimal.

Levels assume digital full scale plays back at 100 dB SPL. Change SPL_AT_FS if
you know better; only loudness-in-sones depends on it much.
"""
import numpy as np
from scipy import signal

EPS = 1e-12
SPL_AT_FS = 100.0

# Critical band (Bark) edges in Hz, after Zwicker
BARK_EDGES = np.array([20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480,
                       1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700,
                       9500, 12000, 15500], dtype=float)
BARK_CENTERS = np.sqrt(BARK_EDGES[:-1] * BARK_EDGES[1:])
NB = BARK_CENTERS.size

# Calibration constants. Each is set so the reference signal for the unit reads
# 1.0, and selftest.py checks they still do.
_CAL_SONE = 1.4737      # 1 kHz tone at 40 dB SPL = 1 sone
_CAL_ACUM = 1.0022      # 1 kHz tone at 60 dB SPL = 1 acum (reference is narrow-band noise there)
_CAL_ROUGH = 1.1176     # 1 kHz tone, 100% AM at 70 Hz, 60 dB = 1.0
_CAL_FLUCT = 1.0000     # 1 kHz tone, 100% AM at 4 Hz, 60 dB = 1.0


def _threshold_quiet_db(f):
    k = np.maximum(f, 20.0) / 1000.0
    return 3.64 * k ** -0.8 - 6.5 * np.exp(-0.6 * (k - 3.3) ** 2) + 1e-3 * k ** 4


_LTQ = _threshold_quiet_db(BARK_CENTERS)


def band_levels(f, P):
    """STFT power (freq x time, scipy 'spectrum' scaling, Hann) -> mean-square
    per Bark band per frame, where a full-scale sine comes out as 0.5."""
    out = np.zeros((NB, P.shape[1]))
    for i in range(NB):
        m = (f >= BARK_EDGES[i]) & (f < BARK_EDGES[i + 1])
        if m.any():
            out[i] = P[m].sum(axis=0) * (2.0 / 1.5)
    return out


def _spread(L):
    """Excitation spreading across Bark bands. L: band levels in dB SPL
    (bands x frames). Masking spreads steeply downward in frequency (27 dB per
    Bark) and more gently upward, flatter the louder the band is."""
    E = 10.0 ** (L / 10.0)
    out = E.copy()
    idx = np.arange(NB)
    for j in range(NB):
        up = np.clip(24.0 + 230.0 / BARK_CENTERS[j] - 0.2 * L[j], 5.0, 40.0)  # dB/Bark, per frame
        for z in range(NB):
            if z == j:
                continue
            dz = z - j
            att = 27.0 * (-dz) if dz < 0 else up * dz
            out[z] += E[j] * 10.0 ** (-att / 10.0)
    return out


def specific_loudness(f, P):
    """Returns (N' per band per frame in sone/Bark, total sones per frame)."""
    ms = band_levels(f, P)
    L = 10.0 * np.log10(ms + EPS) + 3.01 + SPL_AT_FS  # full-scale sine -> SPL_AT_FS
    E = _spread(L)
    Etq = 10.0 ** (_LTQ / 10.0)[:, None]
    n = 0.08 * Etq ** 0.23 * ((0.5 + 0.5 * E / Etq) ** 0.23 - 1.0)
    n = np.maximum(n, 0.0) * _CAL_SONE
    return n, n.sum(axis=0)


def sharpness(n_spec):
    """DIN 45692-style sharpness in acum from mean specific loudness per band."""
    z = np.arange(NB) + 0.5
    g = np.where(z < 15.8, 1.0, 0.15 * np.exp(0.42 * (z - 15.8)) + 0.85)
    tot = n_spec.sum()
    if tot <= 0:
        return 0.0
    return float(_CAL_ACUM * 0.11 * np.sum(n_spec * g * z) / tot)


def _envelopes(x, sr, env_sr):
    """Hilbert envelope of each Bark band, downsampled to env_sr."""
    nyq = sr / 2
    envs = []
    q = max(1, int(round(sr / env_sr)))
    for i in range(NB):
        # Three Bark wide, centred on band i, overlapping its neighbours. A
        # band exactly one Bark wide would chop the sidebands off a modulated
        # tone sitting mid-band and under-read its modulation.
        lo = BARK_EDGES[max(i - 1, 0)]
        hi = min(BARK_EDGES[min(i + 2, NB)], nyq * 0.98)
        if BARK_EDGES[i] >= nyq * 0.98 or lo >= hi:
            envs.append(None)
            continue
        sos = signal.butter(3, [lo, hi], btype="band", fs=sr, output="sos")
        y = signal.sosfiltfilt(sos, x)
        n = y.size
        nf = int(2 ** np.ceil(np.log2(n)))
        e = np.abs(signal.hilbert(y, nf))[:n]
        envs.append(signal.resample_poly(e, 1, q) if q > 1 else e)
    return envs, sr / q


def modulation(x, sr, n_spec):
    """Roughness and fluctuation strength.

    Both come from how hard the loudness of each critical band is being
    modulated. Around 70 times a second the ear hears that as roughness
    (growl, rasp, buzz). Around 4 times a second it hears pulsing or wobble.
    Each band's modulation depth is weighted by that band's share of loudness.
    Scale: 1.0 = a 1 kHz tone fully amplitude-modulated at the given rate."""
    if x.size < int(0.3 * sr):
        return None, None, None
    envs, esr = _envelopes(x, sr, 2000.0)
    w = n_spec / (n_spec.sum() + EPS)
    r_sos = signal.butter(2, [35.0, 140.0], btype="band", fs=esr, output="sos")
    do_f = x.size >= int(1.5 * sr)
    if do_f:
        fq = 10
        f_sos = signal.butter(2, [2.0, 8.0], btype="band", fs=esr / fq, output="sos")
    r_tot = f_tot = 0.0
    r_band = np.zeros(NB)
    means = np.array([e.mean() if e is not None else 0.0 for e in envs])
    # Only bands that really carry signal count. Masking spread gives quiet
    # neighbours a loudness weight, but what is in them is filter leakage.
    live = means > 0.03 * means.max()
    w = np.where(live, w, 0.0)
    w = w / (w.sum() + EPS)
    for i, e in enumerate(envs):
        if e is None or w[i] < 1e-4:
            continue
        mean = means[i]
        cut = int(0.05 * esr)  # drop filter edge junk
        ac = signal.sosfiltfilt(r_sos, e)[cut:-cut or None]
        m = min(1.0, np.sqrt(2.0) * np.sqrt(np.mean(ac ** 2)) / mean)
        r_band[i] = m
        r_tot += w[i] * m * m
        if do_f:
            ed = signal.resample_poly(e, 1, fq)
            cutf = int(0.4 * esr / fq)
            acf = signal.sosfiltfilt(f_sos, ed)[cutf:-cutf or None]
            mf = min(1.0, np.sqrt(2.0) * np.sqrt(np.mean(acf ** 2)) / mean)
            f_tot += w[i] * mf * mf
    rough = float(_CAL_ROUGH * np.sqrt(r_tot))
    fluct = float(_CAL_FLUCT * np.sqrt(f_tot)) if do_f else None
    return rough, fluct, r_band


def harmonic_profile(x, sr, pt, f0, nh=16):
    """Order tracking: follow the pitch and read off how strong each harmonic
    is, frame by frame. The result is the sound's recipe independent of what
    note (or RPM) it is at. Also estimates how much of the sound is NOT
    harmonics, i.e. noise, breath, hiss, grit."""
    v = ~np.isnan(f0)
    if v.sum() < 4:
        return None
    med = float(np.median(f0[v]))
    nfft = int(2 ** np.clip(np.round(np.log2(sr / (med / 8.0))), 11, 14))
    nfft = min(nfft, int(2 ** np.floor(np.log2(max(x.size, 2048)))))
    hop = nfft // 4
    f, t, Z = signal.stft(x, sr, window="hann", nperseg=nfft, noverlap=nfft - hop,
                          boundary=None, padded=False)
    P = np.abs(Z) ** 2
    df = f[1] - f[0]
    H = np.zeros(nh)
    harm_e = tot_e = 0.0
    used = 0
    for ci, tc in enumerate(t):
        j = int(np.argmin(np.abs(pt - tc)))
        if not v[j] or abs(pt[j] - tc) > 0.05:
            continue
        fo = f0[j]
        if fo < 4 * df:  # harmonics closer than the analysis can separate
            continue
        col = P[:, ci]
        tot = col[f >= 20].sum()
        if tot <= 0:
            continue
        he = 0.0
        for k in range(1, nh + 1):
            fc = k * fo
            if fc >= sr / 2 - 3 * df:
                break
            b = int(round(fc / df))
            tol = max(2, int(np.ceil(0.02 * fc / df)))
            lo, hi = max(0, b - tol), min(col.size, b + tol + 1)
            p = lo + int(np.argmax(col[lo:hi]))
            e = col[max(0, p - 2): p + 3].sum()
            H[k - 1] += e
            he += e
        harm_e += min(he, tot)
        tot_e += tot
        used += 1
    if used < 3 or H.max() <= 0:
        return None
    db = 10.0 * np.log10(H / H.max() + 1e-10)
    odd = H[0::2].sum()
    even = H[1::2].sum()
    present = db > -50
    ks = np.arange(1, nh + 1)
    tilt = None
    if present.sum() >= 3:
        tilt = float(np.polyfit(np.log2(ks[present]), db[present], 1)[0])
    return {
        "levels_db": [round(float(d), 1) for d in db],
        "strongest": int(np.argmax(H)) + 1,
        "count_within_30db": int(np.sum(db > -30)),
        "odd_minus_even_db": float(10 * np.log10((odd + EPS) / (even + EPS))),
        "tilt_db_per_octave": tilt,
        "noise_share": float(np.clip(1.0 - harm_e / (tot_e + EPS), 0.0, 1.0)),
        "frames_used": used,
    }


def steadiness(pt, f0, et, er):
    """How machine-perfect the pitch and level are. Real sound sources wobble.
    Returns pitch jitter in cents and level shimmer in dB (both measured as
    fast deviation from the local trend)."""
    out = {"pitch_jitter_cents": None, "level_shimmer_db": None}
    v = ~np.isnan(f0)
    # Deviation from a short local curve fit, so a smooth glide or fade counts
    # as zero wobble and only frame-to-frame unsteadiness is left.
    if v.sum() >= 12:
        c = 1200.0 * np.log2(f0[v])
        res = (c - signal.savgol_filter(c, 9, 2))[4:-4]
        res = res[np.abs(res) < 100]  # ignore note changes
        if res.size >= 4:
            out["pitch_jitter_cents"] = float(np.std(res))
    if er.size >= 12:
        act = er > er.max() * 0.05
        if act.sum() >= 12:
            d = 20 * np.log10(er[act] + EPS)
            out["level_shimmer_db"] = float(np.std((d - signal.savgol_filter(d, 9, 2))[4:-4]))
    return out


def analyze(x, sr, f, P, pt, f0, et, er):
    """x: mono analysis signal. f, P: STFT from ears.py. Returns dict."""
    n, tot = specific_loudness(f, P)
    act = tot > tot.max() * 0.05 if tot.max() > 0 else np.zeros_like(tot, bool)
    if not act.any():
        return None
    n_mean = n[:, act].mean(axis=1)
    sones = float(tot[act].mean())
    rough, fluct, r_band = modulation(x, sr, n_mean)
    R = {
        "loudness_sone": sones,
        "loudness_phon": float(40 + 10 * np.log2(sones)) if sones >= 1 else None,
        "sharpness_acum": sharpness(n_mean),
        "roughness": rough,
        "fluctuation": fluct,
        "specific_loudness": [round(float(v), 3) for v in n_mean],
        "harmonics": harmonic_profile(x, sr, pt, f0),
    }
    R.update(steadiness(pt, f0, et, er))
    R["words"] = describe(R)
    return R


def describe(R):
    """Turn the numbers into the adjectives a person would reach for.
    Thresholds are judgement calls, written down here so they can be argued with."""
    W = []
    s = R["sharpness_acum"]
    if s >= 2.6:
        W.append(("piercing", f"sharpness {s:.2f} acum"))
    elif s >= 1.9:
        W.append(("bright/harsh", f"sharpness {s:.2f} acum"))
    elif s <= 0.75:
        W.append(("dull/muffled", f"sharpness {s:.2f} acum"))
    r = R["roughness"]
    if r is not None:
        if r >= 0.6:
            W.append(("rough/growly", f"roughness {r:.2f}"))
        elif r >= 0.3:
            W.append(("gritty", f"roughness {r:.2f}"))
        elif r <= 0.06:
            W.append(("smooth", f"roughness {r:.2f}"))
    fl = R["fluctuation"]
    if fl is not None and fl >= 0.4:
        W.append(("pulsing/wobbling", f"fluctuation {fl:.2f}"))
    h = R["harmonics"]
    if h:
        ns = h["noise_share"]
        if ns <= 0.03:
            W.append(("sterile", f"only {ns * 100:.0f}% of the energy is anything but clean harmonics"))
        elif ns >= 0.6:
            W.append(("noisy/breathy", f"{ns * 100:.0f}% non-harmonic energy"))
        if h["odd_minus_even_db"] >= 10:
            W.append(("hollow", f"odd harmonics lead even by {h['odd_minus_even_db']:.0f} dB, square-wave territory"))
        if h["count_within_30db"] >= 12:
            W.append(("buzzy", f"{h['count_within_30db']} harmonics within 30 dB of the strongest"))
        elif h["count_within_30db"] <= 2:
            W.append(("pure/plain", f"only {h['count_within_30db']} harmonic(s) of any strength"))
    j = R.get("pitch_jitter_cents")
    if j is not None and j < 0.5:
        W.append(("machine-steady pitch", f"jitter {j:.2f} cents; real sources wobble several"))
    sh = R.get("level_shimmer_db")
    if sh is not None and sh < 0.05:
        W.append(("machine-steady level", f"shimmer {sh:.3f} dB"))
    return [{"word": a, "because": b} for a, b in W]
