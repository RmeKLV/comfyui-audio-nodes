"""
Batch audio nodes for one-file-per-run processing with original filenames.

LoadAudioBatch: point at a folder; each queued run loads the next file
(sequential or seeded-random) and also outputs its base filename.
SaveAudioWithName: saves using a filename passed in as a STRING, so the
output keeps the original name instead of a fixed prefix + counter.
"""

import os
import random

import numpy as np
import torch

try:
    import folder_paths
except ImportError:
    folder_paths = None

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff", ".m4a")


def _list_audio(directory):
    if folder_paths is not None and not os.path.isabs(directory):
        directory = os.path.join(folder_paths.get_input_directory(), directory)
    if not os.path.isdir(directory):
        return directory, []
    files = [f for f in os.listdir(directory)
             if f.lower().endswith(AUDIO_EXTS) and os.path.isfile(os.path.join(directory, f))]
    files.sort()
    return directory, files


class LoadAudioBatch:
    DESCRIPTION = ("Load audio files from a folder one-per-run. Set the Run batch "
                   "count to the number of files; 'index' auto-increments each run.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "tooltip": "Folder of audio files (absolute path, or relative to ComfyUI/input)"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99999, "control_after_generate": True,
                                   "tooltip": "Which file to load. Set Run batch count = file count and leave control on 'increment'."}),
                "order": (["sequential", "random"], {"default": "sequential"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "tooltip": "Shuffle seed for random order"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "INT")
    RETURN_NAMES = ("audio", "filename", "file_count")
    FUNCTION = "load"
    CATEGORY = "audio"

    def load(self, directory, index, order, seed):
        resolved, files = _list_audio(directory)
        if not files:
            raise ValueError(f"No audio files found in: {resolved}")
        if order == "random":
            rng = random.Random(seed)
            rng.shuffle(files)
        chosen = files[index % len(files)]
        path = os.path.join(resolved, chosen)

        data, sr = _read_audio(path)  # [C, T] float32
        waveform = torch.from_numpy(data).unsqueeze(0)  # [1, C, T]
        stem = os.path.splitext(chosen)[0]
        print(f"[LoadAudioBatch] [{index % len(files) + 1}/{len(files)}] {chosen}")
        return ({"waveform": waveform, "sample_rate": sr}, stem, len(files))

    @classmethod
    def IS_CHANGED(cls, directory, index, order, seed):
        return f"{directory}|{index}|{order}|{seed}"


def _read_audio(path):
    import soundfile as sf
    a, sr = sf.read(path, always_2d=True, dtype="float32")  # [T, C]
    return a.T.copy(), int(sr)  # [C, T]


class SaveAudioWithName:
    DESCRIPTION = "Save AUDIO to the output folder using a filename passed as STRING (keeps original name)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "filename": ("STRING", {"forceInput": True, "tooltip": "Base name (no extension), e.g. from LoadAudioBatch"}),
                "subfolder": ("STRING", {"default": "batch", "tooltip": "Subfolder inside ComfyUI/output"}),
                "suffix": ("STRING", {"default": "", "tooltip": "Optional text appended to the name, e.g. _enhanced"}),
                "format": (["flac", "wav"], {"default": "flac"}),
                "overwrite": ("BOOLEAN", {"default": False, "tooltip": "If off, adds _1, _2... instead of overwriting"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def save(self, audio, filename, subfolder, suffix, format, overwrite):
        import soundfile as sf

        out_root = folder_paths.get_output_directory() if folder_paths else "output"
        out_dir = os.path.join(out_root, subfolder) if subfolder else out_root
        os.makedirs(out_dir, exist_ok=True)

        wf = audio["waveform"]
        if isinstance(wf, torch.Tensor):
            wf = wf.cpu().numpy()
        wf = np.asarray(wf)
        if wf.ndim == 3:
            wf = wf[0]           # [C, T]
        if wf.ndim == 1:
            wf = wf[np.newaxis, :]
        data = wf.T              # [T, C]

        base = f"{filename}{suffix}"
        ext = "." + format
        path = os.path.join(out_dir, base + ext)
        if not overwrite:
            n = 1
            while os.path.exists(path):
                path = os.path.join(out_dir, f"{base}_{n}{ext}")
                n += 1

        subtype = "PCM_24" if format == "flac" else "FLOAT"
        # guard against clipping for integer subtypes
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if subtype != "FLOAT" and peak > 1.0:
            data = data / peak * 0.999
        sf.write(path, data, int(audio["sample_rate"]), subtype=subtype)
        print(f"[SaveAudioWithName] wrote {path}")
        return {"ui": {"text": [os.path.basename(path)]}}


class SaveAudioWAV:
    """
    ComfyUI's built-in Save Audio (Advanced) offers only flac / mp3 / opus, so
    there is no way to get a .wav out of a stock graph. This is the missing one:
    same filename_prefix convention as the built-in save nodes, writing WAV.
    """

    DESCRIPTION = "Save AUDIO as .wav (ComfyUI's built-in save nodes cannot). 24-bit by default; use 32-float to keep headroom above 0 dBFS."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "filename_prefix": ("STRING", {"default": "audio/ComfyUI",
                                    "tooltip": "Path/prefix under ComfyUI/output, e.g. audio/SUNO_vocals"}),
                "bit_depth": (["24", "32-float", "16"], {"default": "24",
                              "tooltip": "24 = safe for Premiere. 32-float preserves peaks over 0 dBFS unclipped."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def save(self, audio, filename_prefix="audio/ComfyUI", bit_depth="24"):
        import soundfile as sf

        wf = audio["waveform"]
        if isinstance(wf, torch.Tensor):
            wf = wf.cpu().numpy()
        wf = np.asarray(wf)
        if wf.ndim == 3:
            wf = wf[0]
        if wf.ndim == 1:
            wf = wf[np.newaxis, :]
        data = wf.T.astype(np.float32)

        out_root = folder_paths.get_output_directory() if folder_paths else "output"
        if folder_paths is not None:
            full_dir, base, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, out_root)
            fname = f"{base}_{counter:05}_.wav"
        else:
            full_dir = os.path.join(out_root, os.path.dirname(filename_prefix))
            base = os.path.basename(filename_prefix) or "audio"
            subfolder = os.path.dirname(filename_prefix)
            n = 1
            while os.path.exists(os.path.join(full_dir, f"{base}_{n:05}_.wav")):
                n += 1
            fname = f"{base}_{n:05}_.wav"
        os.makedirs(full_dir, exist_ok=True)
        path = os.path.join(full_dir, fname)

        subtype = {"24": "PCM_24", "16": "PCM_16", "32-float": "FLOAT"}[bit_depth]
        if subtype != "FLOAT":
            peak = float(np.max(np.abs(data))) if data.size else 0.0
            if peak > 1.0:
                data = data / peak * 0.999
                print(f"[SaveAudioWAV] peak {peak:.3f} > 1.0, scaled down for {bit_depth}-bit")
        sf.write(path, data, int(audio["sample_rate"]), subtype=subtype)
        print(f"[SaveAudioWAV] wrote {path}")
        return {"ui": {"audio": [{"filename": fname, "subfolder": subfolder, "type": "output"}]}}


NODE_CLASS_MAPPINGS = {
    "LoadAudioBatch": LoadAudioBatch,
    "SaveAudioWithName": SaveAudioWithName,
    "SaveAudioWAV": SaveAudioWAV,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudioBatch": "Load Audio Batch (folder)",
    "SaveAudioWithName": "Save Audio With Name",
    "SaveAudioWAV": "Save Audio (WAV)",
}
