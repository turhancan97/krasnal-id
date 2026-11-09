"""Build the approved, self-hosted explainer and its accessible companion files."""

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

from build_review import HERE, REPO, load, run_manim, sha, timestamp
from film_audio import PRODUCTION, validate_approval
from narrate import CAPTION_MAX_CHARS

MEDIA = REPO / "docs/media"


def timing():
    validate_approval()
    result = json.loads((PRODUCTION / "timing.json").read_text())
    assert result["story_sha256"] == sha(HERE / "story.json")
    assert result["voice"] == "af_heart" and result["frames"] == 3600
    assert all(c.get("speed", 1) == 1 for c in result["chapters"])
    return result


def captions_and_pages():
    from PIL import Image

    schedule = timing()
    MEDIA.mkdir(exist_ok=True)
    for kind in ["srt", "vtt"]:
        lines = ["WEBVTT", ""] if kind == "vtt" else []
        for index, cue in enumerate(schedule["cues"], 1):
            assert "\n" not in cue["text"] and len(cue["text"]) <= CAPTION_MAX_CHARS
            if kind == "srt":
                lines.append(str(index))
            separator = "," if kind == "srt" else "."
            lines.extend(
                [
                    f"{timestamp(cue['start'], separator)} --> {timestamp(cue['end'], separator)}",
                    cue["text"],
                    "",
                ]
            )
        (MEDIA / f"explainer.en.{kind}").write_text("\n".join(lines).rstrip() + "\n")
    with Image.open(HERE / "review/poster.png") as image:
        image.save(MEDIA / "explainer-poster.webp", format="WEBP", quality=88, method=6)
    story = load("story.json")
    credits = load("credits.json")
    esc = html.escape
    sections = []
    for scene, chapter in zip(story["scenes"][:-1], schedule["chapters"][:-1], strict=True):
        start = timestamp(chapter["start"], ".")[:8]
        sections.append(
            f"<section><h2>{esc(scene['title'])} <small>{start}</small></h2>"
            f"<p>{esc(scene['narration'])}</p></section>"
        )
    body = (
        "<p>English narration, voiced by Kokoro-82M using the stock af_heart voice. "
        "This transcript matches the approved narration. Research location findings are "
        "simulations; the browser identifier reports its own measured accuracy.</p>"
        + "".join(sections)
        + '<p><a href="https://github.com/turhancan97/krasnal-id/blob/main/RESULTS.md">'
        "Research results and methods</a> · "
        '<a href="explainer-credits.html">Photograph and narration credits</a></p>'
    )
    (MEDIA / "explainer-transcript.html").write_text(companion_page("Video transcript", body))
    (MEDIA / "explainer-transcript.txt").write_text(
        "\n\n".join(s["narration"] for s in story["scenes"][:-1]) + "\n"
    )
    entries = []
    for photo in credits["photographs"]:
        entries.append(
            f"<li><strong>{esc(photo['label'])}</strong>: "
            f'<a href="{esc(photo["source_url"], quote=True)}">'
            f"{esc(photo['author'])}</a> · "
            f'<a href="{esc(photo["license_url"], quote=True)}">'
            f"{esc(photo['license'])}</a>. "
            f"{esc(photo['modifications'])}</li>"
        )
    body = (
        "<p>These four photographs appear in the film. Each retains its individual terms. "
        "The linked credit record accompanies the downloadable movie.</p><ul>"
        + "".join(entries)
        + "</ul>"
        + "<h2>Narration and animation</h2><p>Synthetic narration: "
        '<a href="https://huggingface.co/hexgrad/Kokoro-82M">Kokoro-82M</a>, '
        "Apache-licensed model and stock af_heart voice. Animation: Manim Community. "
        "Original animation code is MIT; this does not replace the photographs&#8217; terms.</p>"
        + f"<h2>Depicted sculptures</h2><p>{esc(credits['depicted_works'])}</p>"
        + "<p>Removal requests follow the dataset&#8217;s "
        '<a href="https://github.com/turhancan97/krasnal-id/blob/main/data/image-review.json">'
        "image-review record</a>; "
        '<a href="https://github.com/turhancan97/krasnal-id/issues">open an issue</a>.</p>'
        + '<p><a href="explainer-credits.json">Machine-readable credits</a> · '
        '<a href="explainer.mp4" download>Download the video</a></p>'
    )
    (MEDIA / "explainer-credits.html").write_text(companion_page("Video credits", body))
    public_credits = {
        "photographs": [
            {k: v for k, v in p.items() if k != "path"} for p in credits["photographs"]
        ],
        "narration": {
            "tool": "Kokoro-82M",
            "voice": "af_heart",
            "synthetic": True,
            "model_revision": schedule["model_revision"],
            "source": "https://huggingface.co/hexgrad/Kokoro-82M",
        },
        "depicted_works": credits["depicted_works"],
    }
    (MEDIA / "explainer-credits.json").write_text(
        json.dumps(public_credits, ensure_ascii=False, indent=2) + "\n"
    )


def companion_page(title, body):
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Doppeldwarf-306 · {html.escape(title)}</title>"
        '<link rel="icon" href="../brand/doppeldwarf-mark-light.svg">'
        '<link rel="stylesheet" href="../style.css">'
        "<style>h2 small{font-size:.7em;color:var(--ink-2)}"
        "section{gap:.5rem}li{margin:.7rem 0;overflow-wrap:anywhere}</style>"
        '</head><body><main class="page"><a href="../index.html#explainer">'
        f"← Project video</a><h1>{html.escape(title)}</h1>{body}</main></body></html>\n"
    )


def render(cache):
    schedule = timing()
    assert sha(cache / "film/narration.wav") == schedule["narration_sha256"]
    MEDIA.mkdir(exist_ok=True)
    source = run_manim(cache, "FullFilm", "doppeldwarf-film", scene_file=HERE / "film_scenes.py")
    encode(cache, source)


def encode(cache, source):
    """Encode picture with the lossless master, avoiding intermediate AAC priming."""
    assert sha(cache / "film/narration.wav") == timing()["narration_sha256"]
    ffmpeg = Path(sys.executable).parent / "ffmpeg"
    temporary = MEDIA / ".explainer.encode.mp4"
    subprocess.run(
        [
            str(ffmpeg),
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-i",
            str(cache / "film/narration.wav"),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            "120",
            "-c:v",
            "libx264",
            "-threads",
            "4",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "slow",
            "-crf",
            "22",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    temporary.replace(MEDIA / "explainer.mp4")


def verify_delivery_audio(cache):
    import numpy as np
    import soundfile as sf

    schedule = timing()
    original_path = cache / "film/narration.wav"
    assert sha(original_path) == schedule["narration_sha256"]
    decoded = cache / "film/delivered-audio.wav"
    ffmpeg = Path(sys.executable).parent / "ffmpeg"
    subprocess.run(
        [
            str(ffmpeg),
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(MEDIA / "explainer.mp4"),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "24000",
            str(decoded),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    original, rate = sf.read(original_path, dtype="float32")
    delivered, actual_rate = sf.read(decoded, dtype="float32")
    assert rate == actual_rate == 24000
    assert abs(len(original) - len(delivered)) < 2400
    assert np.isfinite(delivered).all()
    peak = float(np.max(np.abs(delivered)))
    credits_rms = float(np.sqrt(np.mean(delivered[112 * rate :] ** 2)))
    assert peak < 0.99, "Delivered audio clips"
    assert credits_rms < 0.0001, "Credits should be silent"
    report = {
        "delivered_peak": peak,
        "credits_rms": credits_rms,
        "video_sha256": sha(MEDIA / "explainer.mp4"),
        "master_audio_sha256": schedule["narration_sha256"],
        "chapters": [],
        "method": "Decoded delivered AAC compared to the approved natural-speed master",
    }
    for chapter in schedule["chapters"][:-1]:
        start = round((chapter["start"] + 0.15) * rate)
        end = start + round(chapter["speech_seconds"] * rate)
        a, b = original[start:end], delivered[start:end]
        correlation = float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))
        assert correlation > 0.99, (chapter["id"], correlation)
        report["chapters"].append({"id": chapter["id"], "correlation": correlation})
    (PRODUCTION / "delivery-audio.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def check():
    import av

    schedule = timing()
    path = MEDIA / "explainer.mp4"
    with av.open(path) as video:
        stream = video.streams.video[0]
        assert (stream.width, stream.height) == (1920, 1080)
        assert float(stream.average_rate) == 30
        assert stream.codec_context.name == "h264"
        assert video.streams.audio[0].codec_context.name == "aac"
        frames = sum(1 for _ in video.decode(video=0))
        assert frames == 3600, frames
        duration = float(stream.duration * stream.time_base)
        assert duration <= 120 and abs(duration - 120) < 1 / 30
    data = path.read_bytes()
    assert data.find(b"moov") < data.find(b"mdat")
    assert path.stat().st_size < 25 * 1024 * 1024

    def normal(value):
        return re.sub(r"\W+", "", value).lower()

    words = " ".join(c["text"] for c in schedule["cues"])
    original = " ".join(s["narration"] for s in load("story.json")["scenes"])
    assert normal(words) == normal(original)
    last = 0
    for cue in schedule["cues"]:
        assert cue["start"] >= last and cue["end"] > cue["start"]
        assert cue["end"] <= 112
        assert "\n" not in cue["text"] and len(cue["text"]) <= CAPTION_MAX_CHARS
        last = cue["end"]
    for kind in ["srt", "vtt"]:
        blocks = (MEDIA / f"explainer.en.{kind}").read_text().strip().split("\n\n")
        if kind == "vtt":
            assert blocks.pop(0) == "WEBVTT"
        assert len(blocks) == len(schedule["cues"])
        for index, (block, cue) in enumerate(zip(blocks, schedule["cues"], strict=True), 1):
            lines = block.splitlines()
            if kind == "srt":
                assert lines.pop(0) == str(index)
            assert len(lines) == 2, "Each exported cue needs one timing line and one text line"
            separator = "," if kind == "srt" else "."
            assert lines[0] == (
                f"{timestamp(cue['start'], separator)} --> {timestamp(cue['end'], separator)}"
            )
            assert lines[1] == cue["text"]
    assert schedule["chapters"][-1]["start"] == 112 and schedule["chapters"][-1]["end"] == 120
    files = {
        p.name: sha(p) for p in sorted(MEDIA.glob("explainer*")) if p.name != "explainer-build.json"
    }
    receipt = {
        "duration_s": duration,
        "frames": frames,
        "fps": 30,
        "width": 1920,
        "height": 1080,
        "video_codec": "h264",
        "audio_codec": "aac",
        "fast_start": True,
        "bytes": path.stat().st_size,
        "voice": "af_heart",
        "speed": 1.0,
        "burned_in_english_subtitles": True,
        "subtitle_cues": len(schedule["cues"]),
        "subtitle_lines": 1,
        "subtitle_max_chars": CAPTION_MAX_CHARS,
        "credits_start_s": 112,
        "approved_story_sha256": schedule["story_sha256"],
        "source_sha256": load("claims.json")["source_sha256"],
        "model_revision": schedule["model_revision"],
        "assets_sha256": files,
    }
    (MEDIA / "explainer-build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["assets", "render", "check", "delivery"])
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.stage == "assets":
        captions_and_pages()
    elif args.stage == "check":
        check()
    elif args.stage == "delivery":
        if args.cache is None:
            parser.error("--cache is required")
        verify_delivery_audio(args.cache.resolve())
    else:
        if args.cache is None:
            parser.error("--cache is required")
        render(args.cache.resolve())
