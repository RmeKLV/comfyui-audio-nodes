"""ComfyUI audio nodes - HF restoration and batch I/O.

Independent groups:

  hybrid_hf_blend  restore high-frequency detail in generated audio
  batch_audio      process a folder one file per run, keeping filenames
  yue2_tools       stems, lyric score, remix + master for YuE2 songs (optional extra deps)
"""
from .hybrid_hf_blend import (
    NODE_CLASS_MAPPINGS as _HF_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _HF_NAMES,
)
from .batch_audio import (
    NODE_CLASS_MAPPINGS as _BA_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _BA_NAMES,
)

NODE_CLASS_MAPPINGS = {**_HF_CLASSES, **_BA_CLASSES}
NODE_DISPLAY_NAME_MAPPINGS = {**_HF_NAMES, **_BA_NAMES}

# YuE2 tools need ComfyUI internals (folder_paths, comfy.*) and, at run time, the packages in
# requirements-yue2.txt. Their heavy imports are lazy, so a missing extra only fails the node
# that needs it - but guard the import anyway so the HF nodes never go down with it.
try:
    from .yue2_tools import (
        NODE_CLASS_MAPPINGS as _Y2_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _Y2_NAMES,
    )
    NODE_CLASS_MAPPINGS.update(_Y2_CLASSES)
    NODE_DISPLAY_NAME_MAPPINGS.update(_Y2_NAMES)
except ImportError as e:  # pragma: no cover
    print(f"[comfyui-audio-nodes] YuE2 tools disabled: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
