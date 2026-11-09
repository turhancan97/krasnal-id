"""Optional independent local transcription; this does not replace a human listening pass."""

import argparse
import json
from pathlib import Path

from build_review import HERE, load, sha

ASR_MODEL = "openai/whisper-tiny.en"
ASR_REVISION = "87c7102498dcde7456f24cfd30239ca606ed9063"


def transcribe(cache):
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    options = {"revision": ASR_REVISION, "cache_dir": str(cache / "asr")}
    processor = WhisperProcessor.from_pretrained(ASR_MODEL, **options)
    model = (
        WhisperForConditionalGeneration.from_pretrained(ASR_MODEL, use_safetensors=True, **options)
        .to("cpu")
        .eval()
    )
    timing = json.loads((HERE / "production/timing.json").read_text())
    assert sha(cache / "film/narration.wav") == timing["narration_sha256"]
    report = {
        "method": "Independent local automatic transcription; not a human listening pass",
        "model": {"repository": ASR_MODEL, "revision": ASR_REVISION},
        "narration_sha256": timing["narration_sha256"],
        "chapters": [],
    }
    for chapter, scene in zip(
        timing["chapters"][:-1], load("story.json")["scenes"][:-1], strict=True
    ):
        wav = cache / "film" / chapter["audio_file"]
        assert sha(wav) == chapter["audio_sha256"]
        audio, rate = sf.read(wav, dtype="float32")
        assert rate == 24000
        audio = resample_poly(audio, 2, 3).astype(np.float32)
        features = processor(
            audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        with torch.inference_mode():
            ids = model.generate(
                input_features=features.input_features,
                attention_mask=features.attention_mask,
                max_new_tokens=256,
                do_sample=False,
            )
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        entry = {"id": chapter["id"], "expected": scene["narration"], "transcribed": text}
        report["chapters"].append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    (HERE / "production/audio-transcription.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    transcribe(parser.parse_args().cache.resolve())
