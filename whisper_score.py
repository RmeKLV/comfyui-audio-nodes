"""Subprocess helper: transcribe a wav and score it against lyrics. Prints one JSON line.

Runs out-of-process because ctranslate2's OpenMP runtime clashes with torch's
(OMP Error #15) and would abort the ComfyUI server.
"""
import json
import os
import re
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import jiwer  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402


def norm(t):
    t = re.sub(r"\[[^\]]*\]", " ", t.lower()).replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^\w ]+", " ", t).split())


def main():
    wav, lyrics_file, model_name, language = sys.argv[1:5]
    ref = norm(open(lyrics_file, encoding="utf-8").read())
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    # vad_filter must stay off: Silero VAD treats singing over music as non-speech
    # (measured: 4/86 words heard with it, 85/86 without).
    segs, info = model.transcribe(wav, language=None if language == "auto" else language,
                                  beam_size=5, vad_filter=False, condition_on_previous_text=False)
    hyp = norm(" ".join(s.text for s in segs))
    wer = jiwer.wer(ref, hyp) if (ref and hyp) else 1.0
    print(json.dumps({"wer": min(wer, 1.0), "heard": hyp, "ref_words": len(ref.split()),
                      "heard_words": len(hyp.split()), "language": info.language}))


if __name__ == "__main__":
    main()
