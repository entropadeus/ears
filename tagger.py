#!/usr/bin/env python3
"""
tagger.py - borrowed ears.

A small neural network (MobileNetV3, "mn10_as" from EfficientAT by Florian
Schmid et al., MIT licence) trained on AudioSet: about two million ten-second
clips that humans labelled with 527 kinds of sound. Asking it "what is this"
is the closest thing available to asking a room of people who can hear.

This file is a from-scratch, inference-only rewrite of the architecture so
the released weights load without the original repo or torchvision. Weights
are loaded strictly: if this definition drifted from the real one by a single
layer, loading would fail rather than quietly produce nonsense.

Needs: torch, torchaudio, numpy, scipy.
"""
import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "models", "mn10_as_mAP_471.pt")
LABELS = os.path.join(HERE, "models", "audioset_labels.csv")
WEIGHTS_URL = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt"
SR = 32000


def _divisible(v, d=8):
    n = max(d, int(v + d / 2) // d * d)
    return n + d if n < 0.9 * v else n


def _cna(cin, cout, k=1, stride=1, groups=1, act=None):
    layers = [nn.Conv2d(cin, cout, k, stride, (k - 1) // 2, groups=groups, bias=False),
              nn.BatchNorm2d(cout, eps=0.001, momentum=0.01)]
    if act is not None:
        layers.append(act(inplace=True))
    return nn.Sequential(*layers)


class _SE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, _divisible(dim // 4))
        self.fc2 = nn.Linear(_divisible(dim // 4), dim)

    def forward(self, x):
        s = x.mean((2, 3))
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s[:, :, None, None]


class _ConcSE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conc_se_layers = nn.ModuleList([_SE(dim)])

    def forward(self, x):
        return self.conc_se_layers[0](x)


class _Block(nn.Module):
    def __init__(self, cin, k, cexp, cout, se, hs, stride):
        super().__init__()
        act = nn.Hardswish if hs else nn.ReLU
        layers = []
        if cexp != cin:
            layers.append(_cna(cin, cexp, 1, act=act))
        layers.append(_cna(cexp, cexp, k, stride, groups=cexp, act=act))
        if se:
            layers.append(_ConcSE(cexp))
        layers.append(_cna(cexp, cout, 1))
        self.block = nn.Sequential(*layers)
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.block(x)
        return y + x if self.res else y


_CONF = [  # cin, kernel, expanded, cout, SE, hardswish, stride
    (16, 3, 16, 16, False, False, 1), (16, 3, 64, 24, False, False, 2),
    (24, 3, 72, 24, False, False, 1), (24, 5, 72, 40, True, False, 2),
    (40, 5, 120, 40, True, False, 1), (40, 5, 120, 40, True, False, 1),
    (40, 3, 240, 80, False, True, 2), (80, 3, 200, 80, False, True, 1),
    (80, 3, 184, 80, False, True, 1), (80, 3, 184, 80, False, True, 1),
    (80, 3, 480, 112, True, True, 1), (112, 3, 672, 112, True, True, 1),
    (112, 5, 672, 160, True, True, 2), (160, 5, 960, 160, True, True, 1),
    (160, 5, 960, 160, True, True, 1),
]


class MN10(nn.Module):
    def __init__(self, classes=527):
        super().__init__()
        layers = [_cna(1, 16, 3, 2, act=nn.Hardswish)]
        layers += [_Block(*c) for c in _CONF]
        layers.append(_cna(160, 960, 1, act=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(960, 1280),
            nn.Hardswish(), nn.Dropout(0.2), nn.Linear(1280, classes))

    def forward(self, x):
        f = self.features(x)
        return self.classifier(f), f.mean((2, 3))


class Mel(nn.Module):
    """Log-mel front end, matching how the network was trained."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(800, periodic=False))
        self.register_buffer("pre", torch.tensor([[[-0.97, 1.0]]]))
        mb, _ = kaldi.get_mel_banks(128, 1024, SR, 0.0, 15000.0, vtln_low=100.0,
                                    vtln_high=-500.0, vtln_warp_factor=1.0)
        self.register_buffer("mel", F.pad(mb, (0, 1)))

    def forward(self, wav):
        x = F.conv1d(wav.unsqueeze(1), self.pre).squeeze(1)
        s = torch.stft(x, 1024, hop_length=320, win_length=800, center=True,
                       window=self.window, return_complex=True)
        p = s.real ** 2 + s.imag ** 2
        return ((self.mel @ p + 1e-5).log() + 4.5) / 5.0


_cache = {}


def _load():
    if "model" in _cache:
        return _cache["model"], _cache["mel"], _cache["labels"]
    if not os.path.exists(WEIGHTS):
        os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL, WEIGHTS, progress=False)
    model = MN10()
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    mel = Mel().eval()
    with open(LABELS) as fh:
        labels = [r[2] for r in list(csv.reader(fh))[1:]]
    _cache.update(model=model, mel=mel, labels=labels)
    return model, mel, labels


def labels():
    return _load()[2]


def prepare(samples, sr):
    """(n, channels) or (n,) float array at any rate -> mono float32 at 32 kHz."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 2:
        e = np.mean(x ** 2, axis=0)
        m = x.mean(axis=1)
        x = x[:, int(np.argmax(e))] if np.mean(m ** 2) < 0.25 * e.max() else m
    if sr != SR:
        g = math.gcd(int(sr), SR)
        x = signal.resample_poly(x, SR // g, int(sr) // g)
    return x.astype(np.float32)


def tag(samples, sr, window_s=10.0, hop_s=5.0):
    """Returns dict with:
       probs      (527,) mean probability per label over the whole file
       timeline   list of (start_s, end_s, probs) per analysis window
       embedding  (960,) what the network "thinks" the sound is, as a vector;
                  two sounds with similar vectors sound alike to it
    The network was trained on real-world recordings at normal levels, so the
    audio is level-normalised first: it is judged on character, not volume."""
    model, mel, _ = _load()
    x = prepare(samples, sr)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1e-5:
        return None
    rms = float(np.sqrt(np.mean(x ** 2)))
    x = x * min(0.1 / max(rms, 1e-9), 0.98 / peak)  # ~ -20 dBFS rms, no clipping
    W, H = int(window_s * SR), int(hop_s * SR)
    # Trim dead air off both ends: silence is not the sound being judged.
    fr = 320
    nfr = x.size // fr
    if nfr > 2:
        lvl = np.sqrt(np.mean(x[: nfr * fr].reshape(nfr, fr) ** 2, axis=1))
        live = np.flatnonzero(lvl > 0.1 * 10 ** (-40 / 20))
        if live.size:
            x = x[live[0] * fr: (live[-1] + 1) * fr]
    # The network only behaves on roughly ten seconds of input. Measured: on
    # clips under ~4 s its outputs blow up and it calls everything everything.
    # Shorter audio is looped (with a short crossfade at the seam) to fill the
    # window, which keeps it in the range it was trained on.
    if x.size < W:
        xf = min(960, x.size // 4)
        if xf > 0:
            ramp = np.linspace(0, 1, xf, dtype=np.float32)
            body = x[xf:].copy()
            body[-xf:] = body[-xf:] * (1 - ramp) + x[:xf] * ramp
        else:
            body = x
        reps = int(np.ceil(W / max(body.size, 1)))
        x = np.tile(body, reps)[:W]
    starts = [0] if x.size <= W else list(range(0, x.size - W + 1, H))
    if x.size > W and starts[-1] + W < x.size:
        starts.append(x.size - W)
    tl, embs = [], []
    with torch.no_grad():
        for s in starts:
            seg = torch.from_numpy(x[s:s + W])[None]
            logits, feat = model(mel(seg)[:, None])
            tl.append((s / SR, min(x.size, s + W) / SR, torch.sigmoid(logits)[0].numpy()))
            embs.append(feat[0].numpy())
    probs = np.mean([t[2] for t in tl], axis=0)
    emb = np.mean(embs, axis=0)
    return {"probs": probs, "timeline": tl, "embedding": emb}


def top(probs, n=8, floor=0.02):
    lab = labels()
    idx = np.argsort(probs)[::-1][:n]
    return [(lab[i], float(probs[i])) for i in idx if probs[i] >= floor]


def find_labels(query):
    """Case-insensitive lookup: exact label match first, else substring."""
    lab = labels()
    q = query.strip().lower()
    exact = [i for i, l in enumerate(lab) if l.lower() == q]
    if exact:
        return exact
    return [i for i, l in enumerate(lab) if q in l.lower()]


def similarity(emb_a, emb_b):
    a, b = np.asarray(emb_a), np.asarray(emb_b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    import ears
    for p in sys.argv[1:]:
        sr, x = ears.load(p)
        r = tag(x, sr)
        print(os.path.basename(p), "->",
              ", ".join(f"{l} {v:.2f}" for l, v in top(r["probs"])) if r else "silence")
