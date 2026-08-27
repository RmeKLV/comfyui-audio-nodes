"""
Hybrid HF Blend nodes.

HybridHFBlend: combines a clean "base" audio with only the high-frequency
content of an "upscaled" donor, gated by the base envelope. Keeps silence
silent and gives quiet passages only a fraction of the added treble.

AutoInstrumentBlend: same blend, but reads a SUNO stem filename to pick the
right crossover/gain per instrument, sets the gate adaptively from the stem's
own loudness, and passes audio through unchanged when the stem has no real
high-frequency content to extend (bass, sub, dark pads).
"""

import numpy as np
import torch


def _to_np(audio):
    wf = audio["waveform"]
    if isinstance(wf, torch.Tensor):
        wf = wf.cpu().numpy()
    wf = np.asarray(wf, dtype=np.float64)
    if wf.ndim == 3:
        wf = wf[0]
    elif wf.ndim == 1:
        wf = wf[np.newaxis, :]
    return wf, int(audio["sample_rate"])


def _win_rms(x, lw):
    """Per-sample RMS via non-overlapping windows, held across each window."""
    nw = max(1, len(x) // lw)
    r = np.sqrt(np.mean(x[: nw * lw].reshape(nw, lw) ** 2, axis=1))
    out = np.repeat(r, lw)
    if len(out) < len(x):
        out = np.pad(out, (0, len(x) - len(out)), mode="edge")
    return out[: len(x)]


def _blend_core(base, donor, sr, crossover_hz, gate_low_db, gate_high_db,
                hf_gain, release_ms, hf_ceiling=0.6):
    from scipy.signal import firwin, fftconvolve

    if donor.shape[0] != base.shape[0]:
        if donor.shape[0] == 1:
            donor = np.repeat(donor, base.shape[0], axis=0)
        else:
            donor = np.repeat(donor.mean(axis=0, keepdims=True), base.shape[0], axis=0)
    n = min(base.shape[1], donor.shape[1])
    base, donor = base[:, :n], donor[:, :n]

    numtaps = 1025
    cutoff = min(crossover_hz, sr / 2 * 0.95)
    hp = firwin(numtaps, cutoff, fs=sr, pass_zero=False)
    delay = (numtaps - 1) // 2
    hf = np.zeros_like(donor)
    for ch in range(donor.shape[0]):
        conv = fftconvolve(donor[ch], hp, mode="full")
        hf[ch] = conv[delay:delay + n]

    win = max(1, int(0.02 * sr))
    mono = base.mean(axis=0)
    n_win = max(1, len(mono) // win)
    rms = np.sqrt(np.mean(mono[: n_win * win].reshape(n_win, win) ** 2, axis=1))
    rms_db = 20 * np.log10(rms + 1e-12)

    lo, hi = min(gate_low_db, gate_high_db), max(gate_low_db, gate_high_db)
    span = max(hi - lo, 1e-6)
    t = np.clip((rms_db - lo) / span, 0.0, 1.0)
    gain_w = t * t * (3 - 2 * t)

    g = np.zeros_like(gain_w)
    prev = 0.0
    rel = np.exp(-win / (release_ms / 1000.0 * sr))
    for i, target in enumerate(gain_w):
        prev = target if target > prev else rel * prev + (1 - rel) * target
        g[i] = prev

    gain = np.repeat(g, win)
    if len(gain) < n:
        gain = np.pad(gain, (0, n - len(gain)), mode="edge")
    gain = gain[:n]

    gained_hf = hf * (gain[np.newaxis, :] * hf_gain)

    # HF-ceiling limiter: the SR model can hallucinate near-pure-HF bursts on
    # sibilants/transients. Cap the added HF's local energy to hf_ceiling * the
    # base's local energy, so added treble can never dominate the signal.
    if hf_ceiling and hf_ceiling > 0:
        lw = max(1, int(0.03 * sr))
        base_env = _win_rms(base.mean(axis=0), lw)
        hf_env = _win_rms(gained_hf.mean(axis=0), lw)
        ratio = hf_env / (base_env + 1e-9)
        lim = np.where(ratio > hf_ceiling, hf_ceiling / (ratio + 1e-9), 1.0)
        # smooth to avoid zipper
        k = max(1, lw // 2)
        lim = np.convolve(lim, np.ones(k) / k, mode="same")
        gained_hf = gained_hf * lim[np.newaxis, :]

    result = base + gained_hf
    peak = np.max(np.abs(result))
    if peak > 0.999:
        result *= 0.999 / peak
    return result


def _active_level_db(mono, sr):
    """Median RMS (dBFS) of 50 ms windows that are above -50 dBFS."""
    w = max(1, int(0.05 * sr))
    nw = len(mono) // w
    if nw == 0:
        return -120.0
    wr = np.sqrt(np.mean(mono[: nw * w].reshape(nw, w) ** 2, axis=1))
    active = wr > 10 ** (-50 / 20)
    if not active.any():
        return -120.0
    return 20 * np.log10(np.median(wr[active]) + 1e-12)


def _rolloff99_hz(mono, sr):
    """Frequency below which 99% of spectral energy lies."""
    nfft = 16384
    if len(mono) < nfft:
        nfft = 1 << int(np.log2(max(256, len(mono))))
    hop = nfft // 2
    acc = np.zeros(nfft // 2 + 1)
    win = np.hanning(nfft)
    used = 0
    nf = (len(mono) - nfft) // hop
    for i in range(0, max(1, nf), max(1, nf // 200) if nf > 0 else 1):
        seg = mono[i * hop: i * hop + nfft]
        if len(seg) < nfft:
            break
        acc += np.abs(np.fft.rfft(seg * win)) ** 2
        used += 1
    if used == 0:
        return sr / 2
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    c = np.cumsum(acc) / acc.sum()
    return float(freqs[np.searchsorted(c, 0.99)])


# (keyword, (process, crossover_hz, hf_gain)) — checked in order, first match wins.
_PRESETS = [
    ("backing vocal", (True, 10500, 0.75)),
    ("back vocal",    (True, 10500, 0.75)),
    ("bassoon",       (True, 10000, 0.80)),   # woodwind, guard against 'bass' match
    ("lead vocal",    (True, 10500, 1.00)),
    ("vocal",         (True, 10500, 1.00)),
    ("drum",          (True, 11000, 1.00)),
    ("percussion",    (True, 11000, 1.00)),
    ("perc",          (True, 11000, 1.00)),
    ("cymbal",        (True, 10000, 1.10)),
    ("hat",           (True, 10000, 1.10)),
    ("synth",         (True, 11000, 0.90)),
    ("guitar",        (True, 11000, 0.90)),
    ("acoustic",      (True, 10500, 0.90)),
    ("violin",        (True, 10500, 0.85)),
    ("cello",         (True, 10000, 0.80)),
    ("string",        (True, 11000, 0.80)),
    ("trumpet",       (True, 10000, 0.85)),
    ("brass",         (True, 10000, 0.80)),
    ("horn",          (True, 10000, 0.80)),
    ("sax",           (True, 10000, 0.80)),
    ("flute",         (True, 10000, 0.85)),
    ("wood",          (True, 10000, 0.80)),
    ("piano",         (True, 11000, 0.70)),
    ("organ",         (True, 11000, 0.70)),
    ("key",           (True, 12000, 0.60)),
    ("808",           (False, 0, 0)),
    ("sub",           (False, 0, 0)),
    ("bass",          (False, 0, 0)),
    ("other",         (True, 12000, 0.80)),
]
_FALLBACK = (True, 12000, 0.80)


def _lookup_preset(filename):
    name = (filename or "").lower()
    for kw, preset in _PRESETS:
        if kw in name:
            return kw, preset
    return "(default)", _FALLBACK


class HybridHFBlend:
    DESCRIPTION = (
        "Blend base audio with envelope-gated high frequencies from a donor "
        "(e.g. super-resolved) version of the same audio."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_audio": ("AUDIO", {"tooltip": "Clean base (e.g. denoised vocal)"}),
                "donor_audio": ("AUDIO", {"tooltip": "HF donor (e.g. super-resolved vocal)"}),
                "crossover_hz": ("FLOAT", {"default": 10500.0, "min": 2000.0, "max": 20000.0, "step": 100.0,
                                           "tooltip": "Only donor content above this frequency is added"}),
                "gate_low_db": ("FLOAT", {"default": -55.0, "min": -90.0, "max": 0.0, "step": 1.0,
                                          "tooltip": "Base level (dBFS) below which no HF is added"}),
                "gate_high_db": ("FLOAT", {"default": -35.0, "min": -90.0, "max": 0.0, "step": 1.0,
                                           "tooltip": "Base level (dBFS) above which full HF is added"}),
                "hf_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                                      "tooltip": "Overall amount of added HF (1.0 = as-is)"}),
                "release_ms": ("FLOAT", {"default": 200.0, "min": 10.0, "max": 1000.0, "step": 10.0,
                                         "tooltip": "Gate release time in milliseconds"}),
                "hf_ceiling": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 4.0, "step": 0.05,
                                         "tooltip": "Cap added-HF local energy to this fraction of the base (0 = off). Stops SR hallucination bursts."}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "blend"
    CATEGORY = "audio"

    def blend(self, base_audio, donor_audio, crossover_hz=10500.0,
              gate_low_db=-55.0, gate_high_db=-35.0, hf_gain=1.0, release_ms=200.0,
              hf_ceiling=0.6):
        base, sr = _to_np(base_audio)
        donor, donor_sr = _to_np(donor_audio)
        if donor_sr != sr:
            import librosa
            donor = np.stack([librosa.resample(donor[ch], orig_sr=donor_sr, target_sr=sr)
                              for ch in range(donor.shape[0])])
        result = _blend_core(base, donor, sr, crossover_hz, gate_low_db,
                             gate_high_db, hf_gain, release_ms, hf_ceiling)
        out = torch.from_numpy(result.astype(np.float32)).unsqueeze(0)
        return ({"waveform": out, "sample_rate": sr},)


class AutoInstrumentBlend:
    DESCRIPTION = (
        "Auto HF enhancement for SUNO stems: picks crossover/gain from the "
        "instrument name in the filename, sets the gate from the stem's own "
        "loudness, and passes through unchanged when there is no real HF to "
        "extend (bass, sub, dark pads)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_audio": ("AUDIO", {}),
                "donor_audio": ("AUDIO", {}),
                "filename": ("STRING", {"forceInput": True, "tooltip": "Stem name, e.g. from Load Audio Batch"}),
                "hf_gain_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                                           "tooltip": "Global multiplier on the per-instrument gain"}),
                "min_content_hz": ("FLOAT", {"default": 2500.0, "min": 0.0, "max": 12000.0, "step": 100.0,
                                             "tooltip": "If the stem's own 99% rolloff is below this, pass through untouched"}),
                "process_mode": (["auto", "always", "never"], {"default": "auto",
                                 "tooltip": "auto = use name + content checks; always = force blend; never = passthrough"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "auto_blend"
    CATEGORY = "audio"

    def auto_blend(self, base_audio, donor_audio, filename, hf_gain_mult=1.0,
                   min_content_hz=2500.0, process_mode="auto"):
        base, sr = _to_np(base_audio)
        mono = base.mean(axis=0)
        level = _active_level_db(mono, sr)
        rolloff = _rolloff99_hz(mono, sr)
        kw, (proc, crossover, hf_gain) = _lookup_preset(filename)

        # decide whether to process
        skip = False
        reason = ""
        if process_mode == "never":
            skip, reason = True, "process_mode=never"
        elif process_mode == "auto":
            if not proc:
                skip, reason = True, f"preset '{kw}' is skip-type"
            elif rolloff < min_content_hz:
                skip, reason = True, f"rolloff {rolloff:.0f}Hz < {min_content_hz:.0f}Hz (no HF to extend)"

        if skip:
            info = f"PASSTHROUGH [{filename}] match='{kw}' level={level:.1f}dB rolloff={rolloff:.0f}Hz — {reason}"
            print(f"[AutoInstrumentBlend] {info}")
            out = torch.from_numpy(base.astype(np.float32)).unsqueeze(0)
            return ({"waveform": out, "sample_rate": sr}, info)

        # adaptive gate from the stem's own loudness: full HF for content within
        # ~13 dB of the median level, none below ~33 dB under it
        gate_high = float(np.clip(level - 13.0, -85.0, -5.0))
        gate_low = float(np.clip(level - 33.0, -90.0, -10.0))
        eff_gain = hf_gain * hf_gain_mult

        donor, donor_sr = _to_np(donor_audio)
        if donor_sr != sr:
            import librosa
            donor = np.stack([librosa.resample(donor[ch], orig_sr=donor_sr, target_sr=sr)
                              for ch in range(donor.shape[0])])
        result = _blend_core(base, donor, sr, crossover, gate_low, gate_high, eff_gain, 200.0)
        info = (f"BLEND [{filename}] match='{kw}' xover={crossover}Hz gain={eff_gain:.2f} "
                f"gate={gate_low:.0f}/{gate_high:.0f}dB (level={level:.1f} rolloff={rolloff:.0f}Hz)")
        print(f"[AutoInstrumentBlend] {info}")
        out = torch.from_numpy(result.astype(np.float32)).unsqueeze(0)
        return ({"waveform": out, "sample_rate": sr}, info)


def _hf_repair_core(x, sr, protect_hz, tilt_db_per_khz, headroom_db, margin_db,
                    use_median, max_cut_db, med_window_s, lowpass_hz, detect_only):
    """
    Find and pull down hallucinated HF bursts ('crickets') left by SR models.

    Two ceilings, both derived from the audio itself so no reference is needed:

      tilt   - each frame's level in the octave below protect_hz is a band the SR
               model reproduces faithfully. Extrapolate it upward along a natural
               rolloff, add headroom. Because the anchor moves with the frame, a
               real broadband transient (cymbal crash) lifts its own ceiling and
               passes; an HF-only burst does not.
      median - running median of each HF band over med_window_s. Sustained 'air'
               sits at the median, a chirp spikes far above it. Sharper, but it
               also catches legitimate isolated HF transients, so it is off for
               percussive material.

    The lower ceiling wins. Nothing below protect_hz is altered.
    Returns (repaired, stats_dict).
    """
    from scipy.signal import stft, istft, firwin, filtfilt
    from scipy.ndimage import (uniform_filter1d, maximum_filter1d,
                               minimum_filter1d, median_filter)

    nper = 2048 if sr >= 32000 else 1024
    hop = nper // 4
    nyq = sr / 2.0
    protect_hz = float(min(protect_hz, nyq * 0.9))
    anchor_lo, anchor_hi = protect_hz * 0.8, protect_hz

    edges = [protect_hz, protect_hz * 1.1, protect_hz * 1.2, protect_hz * 1.3,
             protect_hz * 1.4, protect_hz * 1.5, protect_hz * 1.6,
             protect_hz * 1.8, protect_hz * 2.1, nyq + 1.0]
    edges = sorted({min(e, nyq + 1.0) for e in edges})

    out = np.zeros_like(x)
    worst_cut = 0.0
    touched = 0
    nframes = 1
    prom_before = 0.0
    worst_at = 0.0
    n_bad = 0
    n_live = 0

    for ch in range(x.shape[0]):
        f, t, Z = stft(x[ch], fs=sr, nperseg=nper, noverlap=nper - hop,
                       window="hann", boundary="zeros", padded=True)
        S = np.abs(Z)
        s_db = 20 * np.log10(S + 1e-12)
        hi = f >= protect_hz
        if not hi.any():
            out[ch] = x[ch]
            continue

        am = (f >= anchor_lo) & (f < anchor_hi)
        anchor = 20 * np.log10(S[am].mean(axis=0) + 1e-12)
        anchor = maximum_filter1d(anchor, size=5, mode="nearest")

        ceil = np.full_like(s_db, np.inf)
        ceil[hi, :] = (anchor[None, :]
                       - tilt_db_per_khz * ((f[hi, None] - protect_hz) / 1000.0)
                       + headroom_db)

        if use_median:
            mw = max(3, int(round(med_window_s * sr / hop)) | 1)
            for lo, hg in zip(edges[:-1], edges[1:]):
                if hg <= edges[0] * 1.05:
                    continue
                m = (f >= lo) & (f < hg)
                if not m.any():
                    continue
                lvl = 20 * np.log10(np.sqrt((S[m] ** 2).mean(axis=0)) + 1e-12)
                med = median_filter(lvl, size=mw, mode="nearest")
                ceil[m, :] = np.minimum(
                    ceil[m, :],
                    (med + margin_db)[None, :] + (s_db[m, :] - lvl[None, :]).clip(max=0))

        g = np.clip(np.minimum(0.0, ceil - s_db), -abs(max_cut_db), 0.0)
        g[~hi, :] = 0.0
        g = uniform_filter1d(g, size=9, axis=0, mode="nearest")
        g = minimum_filter1d(g, size=5, axis=1, mode="nearest")
        g = uniform_filter1d(g, size=3, axis=1, mode="nearest")

        worst_cut = min(worst_cut, float(g.min()))
        touched += int((g.min(axis=0) < -3.0).sum())
        nframes = g.shape[1]

        # The ceiling test *is* the detector, so report from it directly. Measuring
        # instead how far a burst rises above a running HF median does not work:
        # a loud consonant and a hallucinated chirp look identical that way (it
        # scored a clean control the same as a badly-damaged file). What separates
        # them is HF energy unsupported by the band just below it, which is exactly
        # what excess-over-ceiling measures.
        # Only frames carrying real signal are scored. In digital silence the
        # anchor band sits at the noise floor, so the ceiling collapses and
        # inaudible dither reads as a 100+ dB "artifact" -- that made a clean
        # control score worse than a badly damaged file.
        bb = 20 * np.log10(S.sum(axis=0) + 1e-12)
        live = bb > (np.percentile(bb, 95) - 50.0)
        excess = np.maximum(0.0, (s_db - ceil)[hi]).max(axis=0)
        excess = np.where(live, excess, 0.0)
        k = int(np.argmax(excess))
        if float(excess[k]) > prom_before:
            prom_before = float(excess[k])
            worst_at = float(t[k])
        n_bad += int((excess > 8.0).sum())
        n_live += int(live.sum())

        if detect_only:
            out[ch] = x[ch]
            continue

        _, y = istft(Z * 10 ** (g / 20.0), fs=sr, nperseg=nper,
                     noverlap=nper - hop, window="hann")
        L = min(len(y), x.shape[1])
        out[ch, :L] = y[:L]

    if lowpass_hz and lowpass_hz > 0 and not detect_only:
        cut = min(float(lowpass_hz), nyq * 0.98)
        taps = firwin(1025, cut / nyq, window="blackman")
        for ch in range(out.shape[0]):
            out[ch] = filtfilt(taps, [1.0], out[ch])

    return out, {
        "worst_cut_db": worst_cut,
        "frames_touched": touched // max(1, x.shape[0]),
        "frames_total": nframes,
        "worst_excess_db": prom_before,
        "worst_at_s": worst_at,
        "bad_frames": n_bad // max(1, x.shape[0]),
        "live_frames": max(1, n_live // max(1, x.shape[0])),
    }


class HFArtifactRepair:
    DESCRIPTION = (
        "Detects and repairs hallucinated high-frequency bursts ('cricket' chirps) "
        "that SR/upscale models add on sibilants and transients. Self-referential: "
        "needs no clean reference. Insert directly before any Save node. Set "
        "detect_only to use it purely as a verifier."
    )

    # Tuned on a stem set with 13 known chirps, scored against two controls that
    # must survive untouched: a clean never-upscaled vocal (real sibilance) and an
    # AudioSR drum stem (real cymbal transients). Chosen points keep worst-case
    # collateral damage on both controls under ~6 dB.
    PRESETS = {
        # mode:        (protect_hz, headroom, tilt, max_cut)
        "vocal":       (10000.0, 12.0, 3.0, 25.0),   # chirps -8.9 dB, controls -4.8 dB worst
        "vocal_hard":  (10000.0,  8.0, 4.0, 30.0),   # chirps -11.0 dB, controls -9.2 dB worst
        "instrument":  (10000.0, 12.0, 2.0, 20.0),
        "percussive":  (12000.0, 14.0, 2.0, 15.0),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "mode": (["vocal", "vocal_hard", "instrument", "percussive", "custom"],
                         {"default": "vocal",
                          "tooltip": "vocal is the safe default. vocal_hard removes more but "
                                     "thins real sibilance. instrument/percussive are gentler and "
                                     "raise the protected floor so cymbals survive."}),
                "detect_only": ("BOOLEAN", {"default": False,
                                "tooltip": "Report what would be repaired without changing the audio"}),
                "lowpass_hz": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 24000.0, "step": 500.0,
                               "tooltip": "Low-pass after repair. 0 = off. Use ~16000 when the source "
                                          "was hard-cut there, since everything above is invented."}),
            },
            "optional": {
                "protect_hz": ("FLOAT", {"default": 10000.0, "min": 2000.0, "max": 22000.0, "step": 250.0,
                               "tooltip": "custom mode: nothing below this is ever altered"}),
                "headroom_db": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 30.0, "step": 0.5,
                                "tooltip": "custom mode: dB allowed above the modelled rolloff. Lower = more removal + more damage"}),
                "tilt_db_per_khz": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 8.0, "step": 0.25,
                                    "tooltip": "custom mode: assumed natural HF rolloff. Steeper = tighter ceiling up high"}),
                "max_cut_db": ("FLOAT", {"default": 25.0, "min": 3.0, "max": 60.0, "step": 1.0,
                               "tooltip": "custom mode: deepest attenuation allowed"}),
                "use_median": ("BOOLEAN", {"default": False,
                               "tooltip": "custom mode only. Adds a temporal-outlier test. WARNING: it cannot "
                                          "tell a hallucinated chirp from a real sibilant or cymbal hit and "
                                          "measured up to 19 dB of damage on clean material. Leave off unless "
                                          "the source genuinely has no HF transients."}),
                "margin_db": ("FLOAT", {"default": 14.0, "min": 3.0, "max": 40.0, "step": 0.5,
                              "tooltip": "custom mode: dB allowed above the running median (only if use_median)"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    FUNCTION = "repair"
    CATEGORY = "audio"

    def repair(self, audio, mode="vocal", detect_only=False, lowpass_hz=0.0,
               protect_hz=10000.0, headroom_db=12.0, tilt_db_per_khz=3.0,
               max_cut_db=25.0, use_median=False, margin_db=14.0):
        x, sr = _to_np(audio)
        if mode != "custom":
            protect_hz, headroom_db, tilt_db_per_khz, max_cut_db = self.PRESETS[mode]
            use_median = False

        y, st = _hf_repair_core(x, sr, protect_hz, tilt_db_per_khz, headroom_db, margin_db,
                                use_median, max_cut_db, 0.7, lowpass_hz, detect_only)

        worst = st["worst_excess_db"]
        pct_bad = 100.0 * st["bad_frames"] / st["live_frames"]
        # Graded on how *widespread* the excess is, not the single worst frame:
        # measured ~10% flagged on both a clean vocal and a real drum stem, vs
        # 53% and 93% on the two known-damaged stems. One loud outlier frame is
        # normal material; a large flagged fraction is a model gone wrong.
        verdict = ("clean" if pct_bad < 15.0
                   else "mild artifacts" if pct_bad < 40.0
                   else "SEVERE artifacts")
        ts = st["worst_at_s"]
        report = (f"[{mode}{' /detect-only' if detect_only else ''}] {verdict}: worst unsupported HF "
                  f"{worst:.1f} dB over ceiling at {int(ts // 60)}:{ts % 60:05.2f}; "
                  f"{pct_bad:.2f}% of frames flagged"
                  + ("" if detect_only else
                     f", deepest cut {st['worst_cut_db']:.1f} dB")
                  + (f"; low-passed at {lowpass_hz:.0f} Hz" if lowpass_hz and not detect_only else ""))
        print(f"[HFArtifactRepair] {report}")

        out = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)
        return ({"waveform": out, "sample_rate": sr}, report)


NODE_CLASS_MAPPINGS = {
    "HybridHFBlend": HybridHFBlend,
    "AutoInstrumentBlend": AutoInstrumentBlend,
    "HFArtifactRepair": HFArtifactRepair,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HybridHFBlend": "Hybrid HF Blend (base + gated HF)",
    "AutoInstrumentBlend": "Auto Instrument Blend (by filename)",
    "HFArtifactRepair": "HF Artifact Repair (de-cricket)",
}
