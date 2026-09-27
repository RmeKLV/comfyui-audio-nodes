"""YuE2 tools: post-processing for YuE2 songs generated in ComfyUI.

Stem separation (Demucs), lyric-accuracy scoring (Whisper), optional vocal clean-up
(ClearVoice SE), stem remix + loudness master, and a save node with a per-stem quality report.
The measurements behind every default are in the README.
"""
import csv
import datetime
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio

import comfy.model_management
import comfy.utils
import folder_paths

HERE = os.path.dirname(os.path.abspath(__file__))
STEMS = ["vocals", "drums", "bass", "other", "guitar", "piano"]
_model_cache = {}


def _to_np(audio, i=0):
    """AUDIO dict -> (samples x channels float32 numpy, sample_rate)."""
    return audio["waveform"][i].float().cpu().numpy().T, audio["sample_rate"]


def _stats(x, sr):
    mono = x.mean(1)
    win = max(1, sr // 10)
    frames = mono[: len(mono) // win * win].reshape(-1, win)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9) if len(frames) else np.array([-120.0])
    return {
        "dur": len(x) / sr,
        "rms_db": float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)),
        "peak": float(np.abs(x).max()) if len(x) else 0.0,
        "clip_pct": float((np.abs(x) > 0.999).mean() * 100),
        "silent_pct": float((db < -50).mean() * 100),
        "tail_db": float(db[-20:].mean()),  # last 2 s
    }


class YuE2DemucsSeparate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "model": (["htdemucs_ft", "htdemucs_6s", "htdemucs"], {
                "tooltip": "htdemucs_ft: best 4-stem split (~0.8x song length on GPU). "
                           "htdemucs_6s: adds guitar + piano, ~5x faster, slightly weaker."}),
            "shifts": ("INT", {"default": 1, "min": 1, "max": 10,
                               "tooltip": "Random-shift averaging. Higher = marginally cleaner, linearly slower."}),
        }}

    RETURN_TYPES = ("AUDIO",) * 8
    RETURN_NAMES = ("vocals", "instrumental", "drums", "bass", "other", "guitar", "piano", "residual")
    FUNCTION = "separate"
    CATEGORY = "audio/yue2 tools"
    DESCRIPTION = ("Split a song into stems with Demucs. 'instrumental' = everything except vocals. "
                   "guitar/piano are silent unless model is htdemucs_6s. Stems are returned at the input "
                   "sample rate (Demucs itself runs at 44.1 kHz). 'residual' = input minus all stems: "
                   "feed it to Stem Remix + Master so an untouched remix equals the original exactly.")

    def separate(self, audio, model, shifts):
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        device = comfy.model_management.get_torch_device()
        if model not in _model_cache:
            _model_cache.clear()
            _model_cache[model] = get_model(model).eval()
        m = _model_cache[model].to(device)
        wav = audio["waveform"].float()
        if wav.shape[1] == 1:
            wav = wav.repeat(1, 2, 1)
        wav = wav[:, :2]
        in_sr, orig = audio["sample_rate"], wav.cpu()
        if in_sr != m.samplerate:
            wav = torchaudio.functional.resample(wav, in_sr, m.samplerate)
        outs = {s: [] for s in STEMS}
        pbar = comfy.utils.ProgressBar(wav.shape[0])
        for b in range(wav.shape[0]):
            x = wav[b]
            ref = x.mean(0)
            mean, std = ref.mean(), ref.std() + 1e-8
            with torch.no_grad():
                y = apply_model(m, ((x - mean) / std)[None], device=device, shifts=shifts,
                                split=True, overlap=0.25, progress=False)[0]
            y = (y * std + mean).cpu()
            for s in STEMS:
                outs[s].append(y[m.sources.index(s)] if s in m.sources else torch.zeros_like(x))
            pbar.update(1)
        m.cpu()
        comfy.model_management.soft_empty_cache()
        n = orig.shape[-1]
        stems = {}
        for s, v in outs.items():
            w = torch.stack(v)
            if in_sr != m.samplerate:
                w = torchaudio.functional.resample(w, m.samplerate, in_sr)
            w = w[..., :n]
            if w.shape[-1] < n:
                w = torch.nn.functional.pad(w, (0, n - w.shape[-1]))
            stems[s] = w
        residual = orig - sum(stems.values())
        pk = lambda w: {"waveform": w, "sample_rate": in_sr}  # noqa: E731
        inst = sum(stems[s] for s in STEMS if s != "vocals")
        return (pk(stems["vocals"]), pk(inst), pk(stems["drums"]), pk(stems["bass"]), pk(stems["other"]),
                pk(stems["guitar"]), pk(stems["piano"]), pk(residual))


class YuE2LyricScore:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vocals": ("AUDIO", {"tooltip": "Best fed the Demucs vocal stem; the full mix works too."}),
            "lyrics": ("STRING", {"multiline": True, "forceInput": True}),
            "filename_prefix": ("STRING", {"default": "audio/YuE2_song"}),
            "whisper_model": (["mobiuslabsgmbh/faster-whisper-large-v3-turbo",
                               "Systran/faster-whisper-small", "Systran/faster-whisper-tiny"],),
            "language": (["en", "auto", "ja", "zh", "ko", "es", "fr", "de"],),
        }, "optional": {
            "seed": ("INT", {"forceInput": True}),
            "style": ("STRING", {"forceInput": True}),
        }}

    RETURN_TYPES = ("FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("lyric_accuracy", "report", "filename_prefix")
    FUNCTION = "score"
    CATEGORY = "audio/yue2 tools"
    OUTPUT_NODE = True
    DESCRIPTION = ("Transcribes the vocal with Whisper (CPU, ~25 s per song) and compares it to the lyrics. "
                   "lyric_accuracy = 1 - WER (1.0 = every word sung clearly). Measures intelligibility, "
                   "NOT musicality. Appends every run to output/audio/yue2_scores.csv and returns a "
                   "filename prefix tagged with the score, so candidates sort by quality.")

    def score(self, vocals, lyrics, filename_prefix, whisper_model, language, seed=None, style=None):
        x, sr = _to_np(vocals)
        with tempfile.TemporaryDirectory() as tmp:
            wav, lyr = os.path.join(tmp, "v.wav"), os.path.join(tmp, "l.txt")
            sf.write(wav, x, sr, subtype="PCM_16")
            with open(lyr, "w", encoding="utf-8") as f:
                f.write(lyrics)
            env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", PYTHONIOENCODING="utf-8")
            r = subprocess.run([sys.executable, "-s", os.path.join(HERE, "whisper_score.py"),
                                wav, lyr, whisper_model, language],
                               capture_output=True, text=True, encoding="utf-8", env=env)
        line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("{")), None)
        if r.returncode or not line:
            raise RuntimeError(f"Whisper scoring failed:\n{r.stderr[-2000:]}")
        res = json.loads(line)
        acc = round(1.0 - res["wer"], 3)
        tag = f"{filename_prefix}_lyr{int(round(acc * 100)):03d}" + (f"_s{seed}" if seed is not None else "")
        report = (f"Lyric accuracy {acc:.1%}  (WER {res['wer']:.3f}; heard {res['heard_words']} of "
                  f"{res['ref_words']} words; language {res['language']})\n"
                  + ("  -> strong take" if acc >= 0.85 else "  -> words smeared - try another seed" if acc < 0.6
                     else "  -> usable") + f"\nHeard: {res['heard'][:400]}")
        out_dir = os.path.join(folder_paths.get_output_directory(), "audio")
        os.makedirs(out_dir, exist_ok=True)
        log = os.path.join(out_dir, "yue2_scores.csv")
        new = not os.path.exists(log)
        with open(log, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "lyric_accuracy", "wer", "seed", "file_prefix", "style", "heard"])
            w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), acc, round(res["wer"], 3),
                        seed, tag, (style or "")[:200], res["heard"][:300]])
        return {"ui": {"text": (report,)}, "result": (acc, report, tag)}


class YuE2SaveStems:
    @classmethod
    def INPUT_TYPES(cls):
        opt = {s: ("AUDIO",) for s in ["master", "mix", "vocals", "instrumental"] + STEMS[1:]}
        return {"required": {
            "filename_prefix": ("STRING", {"default": "audio/YuE2_song"}),
            "skip_below_db": ("FLOAT", {"default": -50.0, "min": -120.0, "max": 0.0, "step": 1.0,
                                        "tooltip": "Stems quieter than this (RMS) are reported but not saved."}),
        }, "optional": opt}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "audio/yue2 tools"
    DESCRIPTION = ("Saves the master, mix and every non-silent stem as 24-bit FLAC (32-bit float WAV if it "
                   "peaks above 0 dBFS, so nothing clips) in one folder per song, and "
                   "reports level / peak / clipping / silence / cut-off ending for each.")

    def save(self, filename_prefix, skip_below_db, **stems):
        out = folder_paths.get_output_directory()
        full, name, counter, sub, _ = folder_paths.get_save_image_path(filename_prefix, out)
        song_dir = os.path.join(full, f"{name}_{counter:05}")
        os.makedirs(song_dir, exist_ok=True)
        rel_sub = os.path.join(sub, f"{name}_{counter:05}")
        lines, ui = [f"{os.path.relpath(song_dir, out)}"], []
        for key in ["master", "mix", "vocals", "instrumental"] + STEMS[1:]:
            a = stems.get(key)
            if a is None:
                continue
            x, sr = _to_np(a)
            st = _stats(x, sr)
            flags = []
            if st["clip_pct"] > 0.01 and st["peak"] <= 1.0:  # >1.0 goes to float WAV, not clipped
                flags.append(f"CLIPPING {st['clip_pct']:.2f}%")
            if key in ("mix", "master") and st["tail_db"] > -40:
                flags.append("ABRUPT ENDING (loud last 2 s: the master fades it; or add an [Outro] / try "
                             "another seed; raise max_duration only if length == max_duration)")
            if st["peak"] > 1.0:
                flags.append(f"OVER 0 dBFS (peak {st['peak']:.2f}) - saved as float WAV so it does not clip")
            if key == "mix" and st["silent_pct"] > 10:
                flags.append(f"{st['silent_pct']:.0f}% silent")
            if st["rms_db"] < skip_below_db:
                lines.append(f"  {key:12s} {st['rms_db']:6.1f} dB  empty - not saved")
                continue
            if st["peak"] > 1.0:  # FLAC is integer PCM and would clip: keep the headroom in float WAV
                fn = f"{key}.wav"
                sf.write(os.path.join(song_dir, fn), x.astype(np.float32), sr, subtype="FLOAT")
            else:
                fn = f"{key}.flac"
                sf.write(os.path.join(song_dir, fn), x, sr, subtype="PCM_24")
            ui.append({"filename": fn, "subfolder": rel_sub, "type": "output"})
            lines.append(f"  {key:12s} {st['rms_db']:6.1f} dB  peak {st['peak']:.2f}  {sr // 1000}k  "
                         f"{st['dur']:.1f}s  " + ("  ".join(flags) if flags else "ok"))
        report = "\n".join(lines)
        return {"ui": {"audio": ui, "text": (report,)}, "result": (report,)}


def _match(a, sr, n, ch=2):
    """AUDIO -> [ch, n] tensor at sr (first batch item)."""
    w = a["waveform"][0].float().cpu()
    if a["sample_rate"] != sr:
        w = torchaudio.functional.resample(w, a["sample_rate"], sr)
    if w.shape[0] == 1:
        w = w.repeat(ch, 1)
    w = w[:ch, :n]
    return torch.nn.functional.pad(w, (0, n - w.shape[-1])) if w.shape[-1] < n else w


def _true_peak(x, sr):
    up = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(x.T)).float(), sr, sr * 4)
    return float(up.abs().max())


def _limit(x, sr, ceiling, lookahead_ms=5.0, release_ms=80.0):
    """Lookahead peak limiter, no make-up gain. x: [n, ch] float64."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    la = max(1, int(sr * lookahead_ms / 1000))
    env = np.abs(x).max(1)
    need = np.minimum(1.0, ceiling / np.maximum(env, 1e-12))
    g = minimum_filter1d(need, size=2 * la + 1)          # start reducing before the peak arrives
    rel = np.exp(-1.0 / (sr * release_ms / 1000))
    out = np.empty_like(g)
    cur = 1.0
    for i, t in enumerate(g):                            # instant attack, exponential release
        cur = t if t < cur else rel * cur + (1 - rel) * t
        out[i] = cur
    out = np.minimum(out, uniform_filter1d(out, size=la))  # smooth the attack corner
    return x * out[:, None]


class YuE2StemRemix:
    """Sum (optionally enhanced) stems + the Demucs residual, rebalance, then loudness-master."""

    TARGETS = {"-14 LUFS (Spotify/YouTube)": -14.0, "-11 LUFS (louder)": -11.0,
               "-9 LUFS (club/loud)": -9.0, "off (raw remix, untouched)": None}

    @classmethod
    def INPUT_TYPES(cls):
        db = lambda: ("FLOAT", {"default": 0.0, "min": -24.0, "max": 12.0, "step": 0.5})  # noqa: E731
        return {"required": {
            "vocals": ("AUDIO",), "drums": ("AUDIO",), "bass": ("AUDIO",), "other": ("AUDIO",),
            "vocals_db": db(), "drums_db": db(), "bass_db": db(), "other_db": db(),
            "guitar_db": db(), "piano_db": db(),
            "loudness": (list(cls.TARGETS), {"default": "-14 LUFS (Spotify/YouTube)"}),
            "true_peak_db": ("FLOAT", {"default": -1.0, "min": -6.0, "max": 0.0, "step": 0.1,
                                       "tooltip": "Ceiling after mastering (streaming spec is -1 dBTP)."}),
            "end_fade_s": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5,
                                     "tooltip": "Fade applied ONLY if the song ends loud (YuE2 sometimes stops "
                                                "mid-phrase). 0 = never. Not applied when loudness is off."}),
        }, "optional": {
            "guitar": ("AUDIO",), "piano": ("AUDIO",),
            "residual": ("AUDIO", {"tooltip": "Demucs residual. Connect it: without it the remix loses "
                                              "whatever Demucs failed to assign to any stem."}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("master", "report")
    FUNCTION = "remix"
    CATEGORY = "audio/yue2 tools"
    DESCRIPTION = ("Rebuilds the song from stems (all gains 0 dB + residual = the original mix, bit-for-bit "
                   "apart from float rounding), applies per-stem gain, then normalises integrated loudness "
                   "and limits true peak. Output at the vocal stem's sample rate.")

    def remix(self, vocals, drums, bass, other, vocals_db, drums_db, bass_db, other_db, guitar_db, piano_db,
              loudness, true_peak_db, end_fade_s=2.0, guitar=None, piano=None, residual=None):
        import pyloudnorm as pyln

        sr = vocals["sample_rate"]
        n = vocals["waveform"].shape[-1]
        parts = [(vocals, vocals_db), (drums, drums_db), (bass, bass_db), (other, other_db),
                 (guitar, guitar_db), (piano, piano_db), (residual, 0.0)]
        mix = sum(_match(a, sr, n) * (10 ** (g / 20)) for a, g in parts if a is not None)
        x = mix.numpy().T.astype(np.float64)
        meter = pyln.Meter(sr)
        before = meter.integrated_loudness(x)
        target = self.TARGETS[loudness]
        lines = [f"remix in: {before:.1f} LUFS, peak {20 * np.log10(np.abs(x).max() + 1e-12):.1f} dBFS"]
        limited, tp = False, None
        if target is not None and end_fade_s > 0:
            tail = x[-2 * sr:]
            if len(tail) and 20 * np.log10(np.sqrt((tail ** 2).mean()) + 1e-12) > -40:
                nf = min(len(x), int(end_fade_s * sr))
                x = x.copy()
                x[-nf:] *= (np.cos(np.linspace(0, np.pi / 2, nf)) ** 2)[:, None]
                lines.append(f"abrupt ending detected: {end_fade_s:.1f} s fade-out applied")
        if target is not None:
            # true-peak control, measured on a 4x-oversampled signal so inter-sample peaks count.
            # Own limiter: pedalboard.Limiter adds make-up gain (measured: +2.4 dB louder), which
            # is not what a ceiling should do.
            ceiling = 10 ** (true_peak_db / 20)
            for _ in range(5):  # gain -> limit -> trim; repeat because limiting lowers loudness
                now = meter.integrated_loudness(x)
                if not np.isfinite(now) or abs(now - target) < 0.1:
                    break
                x = x * 10 ** ((target - now) / 20)
                tp = _true_peak(x, sr)
                if tp > ceiling:
                    x = _limit(x, sr, ceiling * 10 ** (-0.2 / 20))
                    limited = True
            tp = _true_peak(x, sr)
            if tp > ceiling:  # residual inter-sample overshoot: trim it
                x *= ceiling / tp
                tp = ceiling
        after = meter.integrated_loudness(x)
        if tp is None:
            lines.append("master : untouched raw remix (loudness off)")
        else:
            lines.append(f"master : {after:.1f} LUFS, true peak {20 * np.log10(tp + 1e-12):.1f} dBTP"
                         + ("  (limiter engaged)" if limited else "  (no limiting needed)"))
        gains = ", ".join(f"{k} {g:+.1f}" for k, g in [("vox", vocals_db), ("drums", drums_db), ("bass", bass_db),
                                                          ("other", other_db), ("gtr", guitar_db), ("piano", piano_db)]
                          if g)
        lines.append("stem gains: " + (gains or "all 0 dB") + ("" if residual is not None else
                                                                "   WARNING: no residual connected"))
        report = "\n".join(lines)
        out = torch.from_numpy(x.T.astype(np.float32).copy()).unsqueeze(0)
        return {"ui": {"text": (report,)}, "result": ({"waveform": out, "sample_rate": sr}, report)}


def _node(name):
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f"Enhance Stems needs the '{name}' node, which is not installed/loaded.")
    return cls()


def _mask_to_stereo(orig, cleaned_mono, sr):
    """Apply ClearVoice's denoise to both channels: derive a T-F gain from cleaned/original mid,
    apply it to L and R. Keeps the stereo image ClearVoice's mono conversion throws away."""
    n_fft, hop = 2048, 512
    win = torch.hann_window(n_fft)
    x = orig["waveform"][0].float().cpu()                       # [2, n]
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    n = x.shape[-1]
    c = _match(cleaned_mono, orig["sample_rate"], n, ch=1)[0]    # [n]
    st = lambda s: torch.stft(s, n_fft, hop, window=win, return_complex=True)  # noqa: E731
    mid = st(x.mean(0))
    gain = (st(c).abs() / (mid.abs() + 1e-7)).clamp(0.0, 1.0)
    # smooth over 3 frames x 3 bins so the mask does not add musical noise
    gain = torch.nn.functional.avg_pool2d(gain[None, None], 3, stride=1, padding=1)[0, 0]
    out = torch.stack([torch.istft(st(x[ch]) * gain, n_fft, hop, window=win, length=n) for ch in range(2)])
    return {"waveform": out[None], "sample_rate": orig["sample_rate"]}


class YuE2EnhanceStems:
    """Optional vocal clean-up of separated stems (ClearVoice SE, stereo kept)."""

    MODES = ["off", "clean vocal (ClearVoice SE)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (cls.MODES, {"default": "off", "tooltip": (
                "clean vocal: ClearVoice MossFormer2 SE removes instrument bleed from the vocal stem "
                "(-8.6 dB in the gaps between lines, singing -0.2 dB), ~1.7 min per song. Stereo is kept "
                "by transferring its effect as a mask onto L/R. Instruments are never processed: "
                "super-resolution (AudioSR, ClearVoice SR) was measured to add artifacts to YuE2, whose "
                "output is already full-band.")}),
            "vocals": ("AUDIO",), "drums": ("AUDIO",), "bass": ("AUDIO",), "other": ("AUDIO",),
        }, "optional": {"guitar": ("AUDIO",), "piano": ("AUDIO",)}}

    RETURN_TYPES = ("AUDIO",) * 6 + ("STRING",)
    RETURN_NAMES = ("vocals", "drums", "bass", "other", "guitar", "piano", "report")
    FUNCTION = "enhance"
    CATEGORY = "audio/yue2 tools"
    DESCRIPTION = ("Optional vocal clean-up. 'off' passes every stem through untouched at no cost. "
                   "Instrument stems always pass through unchanged.")

    def enhance(self, mode, vocals, drums, bass, other, guitar=None, piano=None):
        out = [vocals, drums, bass, other, guitar, piano]
        if mode == "off":
            return (*out, "mode: off (stems untouched)")
        se_model = _node("FL_ClearVoice_ModelLoader").load_model(model="MossFormer2_SE_48K")[0]
        cleaned, msg = _node("FL_ClearVoice_Process").process_audio(model=se_model, audio=vocals)
        if str(msg).startswith("Error"):
            raise RuntimeError(f"ClearVoice failed: {msg}")
        out[0] = _mask_to_stereo(vocals, cleaned, vocals["sample_rate"])
        comfy.model_management.soft_empty_cache()
        return (*out, "mode: clean vocal - ClearVoice SE applied to vocals (stereo kept); instruments untouched")


NODE_CLASS_MAPPINGS = {
    "YuE2EnhanceStems": YuE2EnhanceStems,
    "YuE2StemRemix": YuE2StemRemix,
    "YuE2DemucsSeparate": YuE2DemucsSeparate,
    "YuE2LyricScore": YuE2LyricScore,
    "YuE2SaveStems": YuE2SaveStems,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2EnhanceStems": "Enhance Stems (optional)",
    "YuE2StemRemix": "Stem Remix + Master",
    "YuE2DemucsSeparate": "Stem Separate (Demucs)",
    "YuE2LyricScore": "Lyric Accuracy Score (Whisper)",
    "YuE2SaveStems": "Save Stems + Quality Report",
}
