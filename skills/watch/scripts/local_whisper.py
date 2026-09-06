#!/usr/bin/env python3
"""Local Whisper fallback via faster-whisper (CTranslate2) — no API key needed.

Kicks in when no cloud (Groq/OpenAI) key is configured. Reuses whisper.extract_audio
for the same ffmpeg audio extraction, then transcribes on-device with
faster-whisper large-v3. Returns {start, end, text} segments identical in shape to
the cloud path, so filter_range / format_transcript downstream don't care.

Requires `pip install faster-whisper` (imports lazily, only when actually used).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from whisper import extract_audio  # same ffmpeg mono-16k extraction as the cloud path

# ponytail: large-v3 default (the "v3" model); override with WATCH_FW_MODEL=small
# for speed on a slow CPU. int8 keeps it torch-free; bump to float16 if on GPU.
DEFAULT_MODEL = os.environ.get("WATCH_FW_MODEL", "large-v3")

_model_cache: dict[str, object] = {}


def _get_model(name: str):
    from faster_whisper import WhisperModel  # heavy import — only when transcribing

    if name not in _model_cache:
        # device="auto" uses CUDA if present else CPU; int8 needs neither torch nor GPU.
        _model_cache[name] = WhisperModel(name, device="auto", compute_type="int8")
    return _model_cache[name]


def transcribe_video_local(
    video_path: str,
    audio_out: Path,
    model_name: str | None = None,
) -> tuple[list[dict], str]:
    """Extract audio → transcribe locally. Returns (segments, backend_label)."""
    model_name = model_name or DEFAULT_MODEL
    print(f"[watch] extracting audio for local faster-whisper ({model_name})…", file=sys.stderr)
    audio_path = extract_audio(video_path, audio_out)

    print(
        f"[watch] transcribing locally with faster-whisper {model_name} "
        "(first run downloads the model)…",
        file=sys.stderr,
    )
    model = _get_model(model_name)
    # ponytail: temperature=0.0 alone + condition_on_previous_text (the default True)
    # makes whisper loop — it repeats one token or sentence from some point to the end
    # of the file and the tail of the transcript is silently destroyed. These three
    # settings are the fix: no conditioning on prior text, VAD to drop silence, and a
    # temperature ladder so a degenerate decode is retried instead of accepted.
    segments_iter, _info = model.transcribe(
        str(audio_path.resolve()),
        beam_size=5,
        condition_on_previous_text=False,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    out: list[dict] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if text:
            out.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})

    if not out:
        raise SystemExit("Local faster-whisper returned no transcript segments")

    print(f"[watch] transcribed {len(out)} segments via faster-whisper {model_name}", file=sys.stderr)
    return out, f"faster-whisper ({model_name})"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: local_whisper.py <video-path> [audio-out.mp3] [model]", file=sys.stderr)
        raise SystemExit(2)
    import json

    video = sys.argv[1]
    audio_out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("audio.mp3")
    model = sys.argv[3] if len(sys.argv) > 3 else None
    segs, backend = transcribe_video_local(video, audio_out, model)
    print(json.dumps({"backend": backend, "segments": segs}, indent=2))
