# ears

A way for something with no ears to audit audio. It does not hear. It renders
sound to a file, then describes it at three levels, each closer to what a
person would say:

1. **Faults** (`ears.py`): is it broken? Clipping, clicks, dropouts, wrong pitch.
2. **Perception** (`psycho.py`): how would it feel? A model of the human ear
   turns the signal into loudness, sharpness, roughness, pulsing, a harmonic
   recipe, and how machine-steady it is. Then into words: "gritty", "hollow",
   "sterile", "piercing".
3. **Recognition** (`tagger.py`): what IS it? A network trained on about two
   million clips that humans labelled with 527 kinds of sound says what this
   one sounds like, and how closely two sounds resemble each other.

Plus `render.mjs` to get sound out of Web Audio code with nobody in the room,
and `tune.py` to render many variants of a patch and keep the one the ears
score highest.

## Setup

    pip install numpy scipy matplotlib
    pip install torch torchaudio      # only for layer 3; CPU build is fine
    npm install puppeteer-core        # only needed for render.mjs
    # plus Chrome, and ffmpeg if you feed it anything other than WAV

The recognition weights (19 MB, EfficientAT `mn10_as`, MIT) download from
GitHub into `models/` the first time layer 3 runs.

## Get the sound

Render a synth patch offline (fast, sample-exact, repeatable):

    node render.mjs patch synth.js out.wav --seconds 4 --channels 1

`synth.js` is plain browser JS defining `async function build(ctx, seconds)`
that connects things to `ctx.destination`. See `demo/engine_patch.js`.

Record a real page in real time, clicking whatever starts it:

    node render.mjs page app.html out.wav --seconds 5 --do click:#start --do wait:500

Steps run in order: `click:<selector>` `wait:<ms>` `key:<Key>` `type:<text>`
`eval:<js>` `move:<x>,<y>`. Works on file paths and URLs.

## Listen

    python3 ears.py listen out.wav
    python3 ears.py compare mine.wav reference.wav

`listen` prints notes, and writes `out.ears.png` (waveform, spectrogram,
pitch/brightness, average spectrum) and `out.ears.json` (every number).
Exit code is 2 if it found something broken, so it can gate a build.

For low sounds, widen the pitch search: `--fmin 15 --fmax 800`.

## Judge: is it the right sound?

    python3 ears.py judge engine.wav --want Engine --want Vehicle --avoid Synthesizer
    python3 ears.py labels engine      # what labels exist

Passes if the best wanted label scores at least 0.20 and beats everything to
avoid. Exit code 0 or 2.

`compare` also reports resemblance between the two files as the network sees
them. Measured on 40 real recordings: two clips of the same kind of thing score
0.47 to 0.81 (median 0.67), different things 0.20 to 0.43 (median 0.30).

## Tune: let it iterate without you

    python3 tune.py demo/engine_v3.js --want Engine --want Vehicle --avoid Synthesizer --avoid Music

The patch reads its numbers from `window.EARS_PARAMS`. tune.py tries variants,
scores each with the network, and writes the winner. About one second per
variant.

## Taste: teach it what YOU like

    python3 ears.py rate take3.wav 8 --note "good growl"
    python3 ears.py rate take1.wav 2 --note "toy"
    python3 ears.py taste new_take.wav

Ratings are stored with the network's fingerprint of each sound. A new sound is
scored by the rated sounds it most resembles. Needs at least 4 ratings to say
anything and around 15 before it is worth much.

## What it catches

| Thing | How |
|---|---|
| Silence, dropouts, dead air | level below -60 dBFS |
| Clipping | 3+ consecutive samples at full scale |
| Clicks and pops | one-off spikes in the second difference |
| Wrong pitch, pitch that jumps instead of glides | YIN pitch tracking |
| Too loud / too quiet | BS.1770 loudness (LUFS), peak, crest factor |
| Tonal balance | energy per band, centroid, rolloff, third-octave diff vs a reference |
| Tone vs noise | spectral flatness |
| Possible aliasing | energy piled up near Nyquist |
| Phase-inverted or lopsided stereo | channel correlation and balance |
| DC offset, NaN/Inf samples | direct measurement |

## What it still cannot do

- Hear. Layer 3 is a second-hand opinion from a network, layer 2 is a
  simplified ear model. Both are checked against references, neither is you.
- Know your taste until you have rated things. `taste` with few ratings is a guess.
- Be trusted blindly after `tune`. Optimising against a network can find
  sounds that fool the network. Listen to the winner.
- Name things outside its 527 labels, or split fine distinctions inside one
  (it knows "Motorcycle", not which motorcycle).
- Judge clips under about 4 seconds natively. The network falls apart on
  short input, so short clips are looped to 10 s first.
- Read harmonics cleanly through a fast glide: absent harmonics can show up
  as phantoms around -30 dB. Steady notes read down to -50.
- Find a click smaller than the waveform's own edges, or a glitch repeating
  faster than about 15 times a second (that shows up as pitch/spectrum).
- Track pitch in chords or dense mixes. YIN follows one note at a time, and
  on lumpy low sounds (a real engine idling) it hops between octaves, which
  shows up as "pitch jumps".
- Capture `<audio>`/`<video>` tags that never touch an AudioContext.
- Promise that live `page` recordings are glitch-free: they run in real time
  in headless Chrome, so a starved machine can add its own dropouts. Use
  `patch` mode when you need exact results.
- Give true sones/acum/asper. The perception numbers are calibrated to the
  textbook reference signals and behave correctly around them, but they are
  approximations, and loudness assumes full scale plays at 100 dB SPL.

## Proof it works

    python3 selftest.py

39 checks. Generates audio with known planted defects and checks each one is
caught, that clean audio, kick drums, raw saw waves and noise raise no false
alarms, that the ear model reads the textbook reference signals correctly
(1 kHz at 40 dB = 1 sone, 70 Hz AM = rough, 4 Hz AM = pulsing), and that the
network calls noise noise and a sine a sine.

## The demo, as it actually went

`demo/engine_patch.js` is a first-attempt engine: sawtooth plus sub through a
filter. The tool said: Synthesizer 0.64, Engine 0.01, pitch jitter 1 cent.
Four real engine recordings measured 11 to 34 cents of jitter and scored
Vehicle 0.32 to 0.63. `demo/engine_v3.js` was rebuilt from those differences
(no oscillators, one pressure pulse per firing, uneven cylinders, noise) and
then tuned over 49 variants in 45 seconds. Result: Vehicle 0.56, Synthesizer
0.00, jitter 33 cents, resemblance to a real engine recording 0.24 -> 0.49.

Credits: recognition model and weights from EfficientAT (Schmid, Koutini,
Widmer), MIT licence, trained on AudioSet. `tagger.py` is a rewrite of the
inference path only.
