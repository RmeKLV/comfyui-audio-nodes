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

## Requirements

`numpy`, `torch`, `soundfile` — all of which a working ComfyUI install already has.

## Licence

MIT — see [LICENSE](LICENSE).
