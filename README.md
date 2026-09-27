# ComfyUI Audio Nodes

Custom ComfyUI nodes for restoring high-frequency detail in generated audio, plus batch file I/O
that keeps your filenames.

Written for a specific problem: AI music generators (Suno and friends) produce stems that sound
dull or "smeared" up top, and generic audio upscalers either do nothing useful or add a layer of
hissy artefacts across the whole track — including the silent parts.

## Nodes

### Hybrid HF Blend (base + gated HF)
Takes a clean **base** track and an **upscaled donor**, and blends in *only* the donor's
high-frequency content — gated by the base signal's own envelope.

The gating is the point. Silence in the base stays silent instead of filling with upscaler hiss,
and quiet passages get only a fraction of the added treble rather than the same flat boost as a
loud chorus.

### Auto Instrument Blend (by filename)
Same blend, but it reads the stem's filename to work out what instrument it is, then:

- picks the crossover frequency and gain suited to that instrument,
- sets the gate threshold adaptively from the stem's own loudness,
- and **passes the audio through untouched** when the stem has no real high-frequency content to
  extend at all — bass, sub, and dark pads are left alone instead of having imaginary treble
  invented for them.

### HF Artifact Repair (de-cricket)
Finds and repairs the chirpy "cricket" artefacts upscalers leave behind — unsupported
high-frequency energy sitting above what the source could plausibly contain. Reports what it found
(worst offender in dB over the ceiling, timestamp, percentage of frames flagged) and can run in
**detect-only** mode to measure a file without altering it.

### Load Audio Batch (folder)
Point it at a folder; each queued run loads the next file, sequentially or seeded-random. Also
outputs the file's base name.

### Save Audio With Name / Save Audio (WAV)
Saves using a filename passed in as a string, so output keeps the original name instead of a fixed
prefix plus a counter. WAV variant writes 16/24-bit PCM or 32-bit float, with peak-scaling before
integer conversion so clipping can't sneak through.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/RmeKLV/comfyui-audio-nodes
pip install -r comfyui-audio-nodes/requirements.txt
```

Restart ComfyUI. Nodes appear under the **audio** category.

## Typical chain

```
Load Audio Batch ──> (your upscaler) ──> Auto Instrument Blend ──> HF Artifact Repair ──> Save Audio With Name
                          donor              base = original
```

Run `HF Artifact Repair` in detect-only mode first if you want to see how bad a file actually is
before committing to a repair pass.

## YuE2 tools

Post-processing for songs made with **YuE2** (ComfyUI core ≥ v0.36.0), designed to sit at
the end of a YuE2 graph: decode → stems → score → optional clean-up → remix + master → save.
Category **audio/yue2 tools**. The workflows that use them live in
[comfyui-workflows](https://github.com/RmeKLV/comfyui-workflows).

| Node | What it does |
|---|---|
| **Stem Separate (Demucs)** | vocals / drums / bass / other (+ guitar / piano with `htdemucs_6s`), an `instrumental` sum, and a **residual** (input minus all stems). Stems come back at the input sample rate, not Demucs' native 44.1 kHz. |
| **Lyric Accuracy Score (Whisper)** | transcribes the vocal stem and scores it against the lyrics (1 − WER). Tags the filename with the score and appends every run to `output/audio/yue2_scores.csv`, so a batch of takes sorts by quality. |
| **Enhance Stems (optional)** | `off` or `clean vocal`: ClearVoice MossFormer2 SE on the vocal, with its effect transferred as a mask onto L/R so the stereo image survives (ClearVoice itself outputs mono). Needs [FL ClearVoice](https://github.com/filliptm/ComfyUI_FL-ClearVoice). |
| **Stem Remix + Master** | per-stem gain, then loudness normalisation (−14 / −11 / −9 LUFS or off) with a 4×-oversampled true-peak ceiling and a lookahead limiter. Fades the ending only if the song stops loud. |
| **Save Stems + Quality Report** | one folder per take; per-stem level / peak / clipping / abrupt-ending report. Anything peaking above 0 dBFS is written as 32-bit float WAV instead of FLAC so it cannot clip. |

What was measured to get here (RX 7900 XTX, ROCm, fixed seeds):

- **The HF-restoration chain above does not suit YuE2.** YuE2 output is already full-band
  (content to ~21.5 kHz) and mastered (≈ −13 LUFS). Run through AudioSR + Auto Instrument Blend,
  `HF Artifact Repair` flagged **42–67 %** of frames as severe, against 15–32 % on the untouched
  stems and 0.7 % on the mix. ClearVoice SR took 19.5 min on a 2-minute vocal for no measurable
  change. Only ClearVoice **SE** helped: instrument bleed in the gaps between vocal lines −8.6 dB,
  singing −0.2 dB. That is why Enhance Stems offers nothing else.
- **Remix is lossless:** all gains at 0 dB with the residual connected reproduces the mix to
  −91 dB (below one 16-bit LSB). Leave the residual unconnected and you lose whatever Demucs
  could not assign.
- **Raw YuE2 decodes can exceed 0 dBFS** (peaks of 1.17 seen). Use the master.
- **Whisper needs `vad_filter=False` on songs** — Silero VAD treats singing over music as
  non-speech (4 of 86 words heard with it, 85 without). It runs in a subprocess because
  ctranslate2's OpenMP runtime collides with torch's (OMP Error #15) and would take ComfyUI down.
- **Do not use `pedalboard.Limiter` as a ceiling**: it applies make-up gain (+2.4 dB measured).
- The lyric score measures intelligibility, not musicality. It reliably catches the common
  failure — a take that slurs or skips a verse — but listening still picks the winner.

## Requirements

HF-blend and batch nodes: `numpy`, `torch`, `soundfile` — all of which a working ComfyUI install
already has.

YuE2 tools additionally: `pip install -r requirements-yue2.txt` (demucs, faster-whisper, jiwer,
pyloudnorm). If they are missing, only the node that needs them fails; the rest load normally.

## Licence

MIT — see [LICENSE](LICENSE).
