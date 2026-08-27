"""ComfyUI audio nodes - HF restoration and batch I/O.

Two independent groups that happen to be used together:

  hybrid_hf_blend  restore high-frequency detail in generated audio
  batch_audio      process a folder one file per run, keeping filenames
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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
