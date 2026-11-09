"""Synthesize the approved film and derive frame-aligned chapter/caption timing."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

from build_review import HERE, load, sha, validate_sources
from narrate import RATE, cues_for, download

PRODUCTION = HERE / "production"


def validate_approval():
    approval = load("approval.json")
    assert approval["approval_text"] == "Approve with af_heart"
    assert approval["voice"] == "af_heart"
    assert approval["story_sha256"] == sha(HERE / "story.json"), "Approved story changed"
    validate_sources(allow_page_change=True)
    return approval


def spoken_form(text):
    pronunciations = load("pronunciations.json")
    for word, phonemes in pronunciations.items():
        text = text.replace(word, f"[{word}](/{phonemes}/)")
    return text


def word_timings(results):
    words = []
    elapsed = 0.0
    for result in results:
        for token in result.tokens or []:
            if token.start_ts is not None and token.end_ts is not None:
                words.append(
                    {
                        "text": token.text,
                        "start": elapsed + float(token.start_ts),
                        "end": elapsed + float(token.end_ts),
                    }
                )
        elapsed += len(result.audio) / RATE
    return words


def cues_from_words(text, words, seconds, offset=0):
    tokens = [
        SimpleNamespace(text=word["text"], start_ts=word["start"], end_ts=word["end"])
        for word in words
    ]
    result = SimpleNamespace(tokens=tokens, audio=range(round(seconds * RATE)))
    return cues_for(text, [result], offset)


def reflow_captions(cache):
    validate_approval()
    path = PRODUCTION / "timing.json"
    schedule = json.loads(path.read_text())
    assert sha(cache / "film/narration.wav") == schedule["narration_sha256"]
    cues = []
    for scene, chapter in zip(
        load("story.json")["scenes"][:-1], schedule["chapters"][:-1], strict=True
    ):
        offset = chapter["start"] + 0.15
        words = [
            {**word, "start": word["start"] - offset, "end": word["end"] - offset}
            for word in chapter["words"]
        ]
        part = cues_from_words(scene["narration"], words, chapter["speech_seconds"], offset)
        part[-1]["end"] = chapter["end"] - 0.12
        cues.extend(part)
    assert all(re.search(r"\w", cue["text"]) for cue in cues)
    schedule["cues"] = cues
    path.write_text(json.dumps(schedule, ensure_ascii=False, indent=2) + "\n")
    print(f"Reflowed {len(cues)} cues from existing word times; narration unchanged")


def allocate_frames(speech_seconds):
    """Preserve speech at natural speed, then distribute spare frames within 112 s."""
    story = load("story.json")
    target_frames = [round((s["end_s"] - s["start_s"]) * 30) for s in story["scenes"][:-1]]
    # 0.15 s lead + at least 0.2 s tail; no samples are truncated or time-stretched.
    minima = [math.ceil((duration + 0.35) * 30) for duration in speech_seconds]
    budget = 112 * 30
    if sum(minima) > budget:
        raise ValueError(
            f"Natural narration needs {sum(minima) / 30:.2f}s before credits; "
            "do not render or accelerate the approved voice"
        )
    allocation = minima[:]
    # Give each spare frame to the chapter furthest below its approved budget.
    while sum(allocation) < budget:
        index = max(range(7), key=lambda i: (target_frames[i] - allocation[i], -i))
        allocation[index] += 1
    return [*allocation, 8 * 30]


def synthesise_film(cache):
    import espeakng_loader
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KModel, KPipeline
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    approval = validate_approval()
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model_dir, lock = download(cache)
    model = (
        KModel(
            repo_id=lock["repository"],
            config=str(model_dir / "config.json"),
            model=str(model_dir / "kokoro-v1_0.pth"),
        )
        .to("cpu")
        .eval()
    )
    pipeline = KPipeline(lang_code="a", repo_id=lock["repository"], model=model, device="cpu")
    voice = torch.load(model_dir / "voices/af_heart.pt", map_location="cpu", weights_only=True)
    directory = cache / "film"
    directory.mkdir(parents=True, exist_ok=True)
    PRODUCTION.mkdir(exist_ok=True)
    story = load("story.json")
    generated = []
    for scene in story["scenes"][:-1]:
        text = scene["narration"]
        spoken = spoken_form(text)
        key = hashlib.sha256((lock["revision"] + approval["voice"] + spoken).encode()).hexdigest()
        wav = directory / f"{scene['id']}-{key[:12]}.wav"
        memo = wav.with_suffix(".json")
        if wav.exists() and memo.exists():
            info = json.loads(memo.read_text())
            assert info["sha256"] == sha(wav)
            samples, rate = sf.read(wav, dtype="float32")
            assert rate == RATE
        else:
            print("Synthesising approved chapter:", scene["id"], flush=True)
            with torch.inference_mode():
                results = list(pipeline(spoken, voice=voice, speed=1.0))
            samples = np.concatenate([result.audio.numpy() for result in results])
            assert np.isfinite(samples).all() and len(samples) > RATE
            samples *= 0.88 / float(np.abs(samples).max())
            edge = 120
            samples[:edge] *= np.linspace(0, 1, edge)
            samples[-edge:] *= np.linspace(1, 0, edge)
            sf.write(wav, samples, RATE, subtype="PCM_16")
            info = {
                "id": scene["id"],
                "text": text,
                "spoken_text": spoken,
                "speed": 1.0,
                "speech_seconds": len(samples) / RATE,
                "sha256": sha(wav),
                "cues": cues_for(text, results),
                "words": word_timings(results),
                "audio_file": wav.name,
            }
            memo.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
        info["cues"] = cues_from_words(text, info["words"], info["speech_seconds"])
        generated.append((samples, info))
        print(scene["id"], f"{info['speech_seconds']:.3f}s at speed 1.0", flush=True)
    frames = allocate_frames([info["speech_seconds"] for _, info in generated])
    mixed = np.zeros(120 * RATE, dtype=np.float32)
    timing = {
        "fps": 30,
        "frames": 3600,
        "duration_s": 120,
        "voice": "af_heart",
        "story_sha256": approval["story_sha256"],
        "model_revision": lock["revision"],
        "chapters": [],
        "cues": [],
    }
    cursor = 0
    for index, (samples, info) in enumerate(generated):
        start = cursor / 30
        end = (cursor + frames[index]) / 30
        offset = start + 0.15
        begin = round(offset * RATE)
        assert begin + len(samples) <= round(end * RATE)
        mixed[begin : begin + len(samples)] = samples
        cues = [
            {**cue, "start": cue["start"] + offset, "end": cue["end"] + offset}
            for cue in info["cues"]
        ]
        cues[-1]["end"] = end - 0.12
        words = [
            {**word, "start": word["start"] + offset, "end": word["end"] + offset}
            for word in info["words"]
        ]
        timing["chapters"].append(
            {
                "id": info["id"],
                "start": start,
                "end": end,
                "frames": frames[index],
                "speech_seconds": info["speech_seconds"],
                "speed": 1.0,
                "audio_file": info["audio_file"],
                "audio_sha256": info["sha256"],
                "words": words,
            }
        )
        timing["cues"].extend(cues)
        cursor += frames[index]
    assert cursor == 112 * 30
    timing["chapters"].append({"id": "credits", "start": 112, "end": 120, "frames": 240})

    def normalise(value):
        return re.sub(r"\W+", "", value).lower()

    caption_text = " ".join(cue["text"] for cue in timing["cues"])
    approved_text = " ".join(scene["narration"] for scene in story["scenes"])
    assert normalise(caption_text) == normalise(approved_text)
    sf.write(directory / "narration.wav", mixed, RATE, subtype="PCM_16")
    timing["narration_sha256"] = sha(directory / "narration.wav")
    timing["peak"] = float(np.abs(mixed).max())
    timing["rms"] = float(np.sqrt(np.mean(mixed**2)))
    (PRODUCTION / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2) + "\n")
    print("Natural narration fits; final film: 120 s, including 8 s credits", flush=True)
    print([(c["id"], round(c["end"] - c["start"], 3)) for c in timing["chapters"]], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reflow", action="store_true")
    args = parser.parse_args()
    if args.reflow:
        reflow_captions(args.cache.resolve())
    else:
        synthesise_film(args.cache.resolve())
