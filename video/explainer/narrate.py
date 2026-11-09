"""Produce only the short review audio, from hash-pinned local Kokoro assets."""

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RATE = 24000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(cache):
    lock = json.loads((HERE / "model-lock.json").read_text())
    target = cache / "model" / lock["revision"]
    target.mkdir(parents=True, exist_ok=True)
    for entry in lock["files"]:
        path = target / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and entry["sha256"] and digest(path) == entry["sha256"]:
            continue
        url = f"https://huggingface.co/{lock['repository']}/resolve/{lock['revision']}/{entry['path']}"
        temporary = path.with_suffix(path.suffix + ".part")
        print("Fetching pinned asset:", entry["path"], flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        checksum = digest(temporary)
        if entry["sha256"] and checksum != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {entry['path']}")
        if entry["bytes"] and temporary.stat().st_size != entry["bytes"]:
            raise ValueError(f"Length mismatch: {entry['path']}")
        temporary.replace(path)
        if entry["sha256"] is None:
            raise ValueError("All model assets must have a locked SHA-256 before synthesis")
    return target, lock


def cues_for(text, results, offset=0):
    """Use synthesis word times, grouping readable phrases; never estimate from word count."""
    tokens = []
    elapsed = offset
    for result in results:
        if not result.tokens:
            raise ValueError("Kokoro returned no word timestamps")
        for token in result.tokens:
            if token.start_ts is not None and token.end_ts is not None:
                tokens.append(
                    (token.text, elapsed + float(token.start_ts), elapsed + float(token.end_ts))
                )
        elapsed += len(result.audio) / RATE
    cues = []
    words = []
    start = None
    last_end = 0
    for word, begin, end in tokens:
        candidate = " ".join([*words, word])
        if words and re.search(r"\w", word) and (len(candidate) > 76 or end - start > 4.0):
            cues.append({"start": start, "end": last_end, "text": " ".join(words)})
            words = []
            start = None
        if start is None:
            start = begin
        words.append(word)
        last_end = end
        if re.search(r"[.!?;]$", word):
            cues.append({"start": start, "end": end, "text": " ".join(words)})
            words = []
            start = None
    if words:
        cues.append({"start": start, "end": last_end, "text": " ".join(words)})

    # Preserve canonical graphemes, including alias labels; normalise only token spacing.
    def normalise(value):
        return re.sub(r"\W+", "", value).lower()

    if normalise(" ".join(c["text"] for c in cues)) != normalise(text):
        raise ValueError("Caption words differ from the review narration")
    for cue in cues:
        cue["text"] = re.sub(r"\s+([,.!?;:])", r"\1", cue["text"])
    for i, cue in enumerate(cues):
        # Keep text on screen through short pauses, without overlapping the next phrase.
        cue["end"] = cues[i + 1]["start"] if i + 1 < len(cues) else elapsed
    return cues


def synthesise(cache):
    import espeakng_loader
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KModel, KPipeline
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

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
    story = json.loads((HERE / "story.json").read_text())
    review = HERE / "review"
    review.mkdir(exist_ok=True)
    report = {
        "model_revision": lock["revision"],
        "sample_rate": RATE,
        "synthetic": True,
        "approved_voice": None,
        "audio": [],
    }
    jobs = [
        (voice, story["voice_samples"]["text"], 15, f"voice-{voice}")
        for voice in story["voice_samples"]["voices"]
    ]
    jobs.append((story["preview"]["voice"], story["preview"]["text"], 10, "preview"))
    for voice, text, budget, name in jobs:
        print("Synthesising", name, flush=True)
        spoken = text.replace("Wrocław", "[Wrocław](/vɹ\u02c8ɔtsw\u0251f/)").replace(
            "Doppeldwarf-306", "Doppel dwarf three oh six"
        )
        pack = torch.load(model_dir / f"voices/{voice}.pt", map_location="cpu", weights_only=True)
        speed = 1.0
        with torch.inference_mode():
            for _attempt in range(5):
                results = list(pipeline(spoken, voice=pack, speed=speed))
                waveform = np.concatenate([r.audio.numpy() for r in results])
                speech_seconds = len(waveform) / RATE
                if speech_seconds <= budget - 0.35:
                    break
                speed *= speech_seconds / (budget - 0.55)
                if speed > 1.25:
                    raise ValueError("Review speech is too long; shorten text rather than rush")
            else:
                raise ValueError("Could not fit review audio without truncation")
        peak = float(np.max(np.abs(waveform)))
        if not np.isfinite(waveform).all() or peak == 0:
            raise ValueError("Invalid audio")
        waveform = waveform * (0.88 / peak)
        leading = 0.12
        pcm = np.zeros(round(budget * RATE), dtype=np.float32)
        begin = round(leading * RATE)
        if begin + len(waveform) > len(pcm):
            raise ValueError("Speech would be truncated")
        pcm[begin : begin + len(waveform)] = waveform
        # Short edge ramps prevent clicks; do not remove any spoken material.
        edge = 120
        pcm[begin : begin + edge] *= np.linspace(0, 1, edge)
        pcm[begin + len(waveform) - edge : begin + len(waveform)] *= np.linspace(1, 0, edge)
        destination = cache / f"{name}.wav" if name == "preview" else review / f"{name}.wav"
        sf.write(destination, pcm, RATE, subtype="PCM_16")
        if name == "preview":
            cues = cues_for(text, results, leading)
            (cache / "preview-cues.json").write_text(json.dumps(cues, indent=2) + "\n")
        entry = {
            "file": destination.name,
            "voice": voice,
            "speed": speed,
            "speech_seconds": speech_seconds,
            "duration_s": budget,
            "peak": float(np.abs(pcm).max()),
            "rms": float(np.sqrt(np.mean(pcm**2))),
            "sha256": digest(destination),
            "text": text,
            "spoken_text": spoken,
        }
        report["audio"].append(entry)
        print(json.dumps(entry), flush=True)
    (review / "audio-receipt.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    synthesise(args.cache.resolve())
