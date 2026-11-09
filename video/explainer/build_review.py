"""Build and validate a review package; deliberately has no full-video render mode."""

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from narrate import CAPTION_MAX_CHARS

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REVIEW = HERE / "review"


def load(name):
    return json.loads((HERE / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def esc(value):
    return html.escape(str(value), quote=True)


def validate_sources(allow_page_change=False):
    story, claims, credits = load("story.json"), load("claims.json"), load("credits.json")
    if allow_page_change:
        approval = load("approval.json")
        assert approval["story_sha256"] == sha(HERE / "story.json")
        assert approval["approval_text"] == "Approve with af_heart"
    assert story["status"] == "awaiting-owner-review", (
        "This build is only for the unapproved review package"
    )
    expected = [
        "#507AFF",
        "#D97FAE",
        "#55A398",
        "#8A6BDD",
        "#EAB34D",
        "#FF7F50",
        "#34C759",
        "#FFCC00",
    ]
    assert list(story["palette"].values()) == expected, "The supplied palette must stay exact"
    assert len(story["scenes"]) == 8
    previous = 0
    for scene in story["scenes"]:
        assert scene["start_s"] == previous and scene["end_s"] > previous
        previous = scene["end_s"]
        assert all(name in claims["claims"] for name in scene["claims"])
    assert previous <= 120 and previous == story["max_duration_s"]
    assert 210 <= sum(len(s["narration"].split()) for s in story["scenes"]) <= 240
    assert len(credits["photographs"]) <= 4
    for filename, checksum in claims["protected_sha256"].items():
        if allow_page_change and filename == "docs/index.html":
            continue
        assert sha(REPO / filename) == checksum, (
            f"Protected research/demo input changed: {filename}"
        )
    for filename, checksum in claims["source_sha256"].items():
        assert sha(REPO / filename) == checksum, (
            f"Source changed: {filename}; review facts deliberately"
        )
    images = {
        i["id"]: i for i in json.loads((REPO / "docs/assets/references.json").read_text())["images"]
    }
    assert len(images) == claims["claims"]["dataset"]["facts"]["photographs"]
    assert (
        len({i["dwarf"] for i in images.values()})
        == claims["claims"]["dataset"]["facts"]["statues"]
    )
    for photo in credits["photographs"]:
        original = images[photo["image_id"]]
        assert all(
            photo[k] == original[k] for k in ["author", "source_url", "license", "license_url"]
        )
        assert sha(REPO / photo["path"]) == photo["sha256"], photo["image_id"]
        assert all(
            photo.get(k)
            for k in ["author", "source_url", "license", "license_url", "modifications"]
        )
    table = (
        (REPO / "RESULTS.md")
        .read_text()
        .split("## 2. The headline result:", 1)[1]
        .split("### What", 1)[0]
    )
    rows = re.findall(r"\| (\d+)(?: \(full\))? \| ([\d.]+)% \| ([\d.]+)% \|", table)
    facts = claims["claims"]["pool"]["facts"]
    assert [int(r[0]) for r in rows] == facts["candidate_counts"]
    assert [float(r[1]) for r in rows] == facts["dinov2_percent"]
    assert [float(r[2]) for r in rows] == facts["clip_percent"]
    print(
        f"Sources, eight-scene budget, exact palette, "
        f"{sum(len(s['narration'].split()) for s in story['scenes'])}-word script "
        "and four photo credits verified"
    )


def write_docs(allow_page_change=False):
    validate_sources(allow_page_change=allow_page_change)
    REVIEW.mkdir(exist_ok=True)
    story, claims, credits = load("story.json"), load("claims.json"), load("credits.json")
    words = sum(len(s["narration"].split()) for s in story["scenes"])
    lines = [
        "# Doppeldwarf-306 · narration draft",
        "",
        f"{words} words · English · 120-second production budget including credits.",
        "",
        "Status: awaiting owner approval. Scene timings are budgets, "
        "not measured full-narration timings.",
        "",
    ]
    for i, scene in enumerate(story["scenes"], 1):
        lines.extend(
            [
                f"## {i}. {scene['title']} ({scene['start_s']}-{scene['end_s']} s)",
                "",
                scene["narration"] or "(No narration; hold credits.)",
                "",
            ]
        )
        for key in scene["claims"]:
            claim = claims["claims"][key]
            lines.extend(
                [
                    f"Source: [{claim['source']} - {claim['section']}](../../{claim['source']}).",
                    "",
                ]
            )
    if allow_page_change:
        lines = [
            line.replace(
                "Status: awaiting owner approval.",
                "Approved by the owner with af_heart narration.",
            )
            for line in lines
        ]
    (HERE / "script.md").write_text("\n".join(lines))
    (REVIEW / "transcript.txt").write_text(
        "\n\n".join(s["narration"] for s in story["scenes"] if s["narration"]) + "\n"
    )
    lines = [
        "# Photograph and narration credits",
        "",
        "Each photograph retains its own terms. Images are resized WebP copies "
        "and are fitted/cropped in the frames and animation.",
        "",
    ]
    for photo in credits["photographs"]:
        lines.extend(
            [
                f"- {photo['label']} ({photo['image_id']}): "
                f"[{photo['author']}]({photo['source_url']}) · "
                f"[{photo['license']}]({photo['license_url']}).",
                f"  Modifications: {photo['modifications']}",
            ]
        )
    lines.extend(
        [
            "",
            credits["depicted_works"],
            f"Removal path: `{credits['removal_path']}`.",
            "",
            "Narration: Kokoro-82M, Apache-2.0 model and stock voices. "
            "Two sample voices are included; the short preview uses af_heart provisionally. "
            "No voice has been approved.",
            "Model: https://huggingface.co/hexgrad/Kokoro-82M",
            "",
            "Original animation code is covered by the repository MIT licence. "
            "That does not replace the photographs' individual terms.",
        ]
    )
    if allow_page_change:
        lines = [
            line.replace(
                "Two sample voices are included; the short preview uses af_heart provisionally. "
                "No voice has been approved.",
                "The owner approved af_heart for the complete film. "
                "Both original voice auditions remain in the archived review package.",
            )
            for line in lines
        ]
    (HERE / "CREDITS.md").write_text("\n".join(lines) + "\n")
    cards = []
    for i, scene in enumerate(story["scenes"], 1):
        source_links = " · ".join(
            f"<a href='../../{esc(claims['claims'][key]['source'])}'>"
            f"{esc(claims['claims'][key]['section'])}</a>"
            for key in scene["claims"]
        )
        frame = f"review/frames/{i:02d}-{scene['id']}.png"
        narration = scene["narration"] or "No narration. Hold credits for eight seconds."
        cards.append(
            '<article class="scene"><div class="scene-top">'
            f"<span>{i:02d}</span><span>{scene['start_s']}-{scene['end_s']} s</span></div>"
            f'<h3>{esc(scene["title"])}</h3><a href="{frame}">'
            f'<img src="{frame}" width="1920" height="1080" '
            f'alt="Storyboard frame {i}: {esc(scene["visual"])}"></a>'
            f'<p class="narration">{esc(narration)}</p>'
            "<details><summary>Composition, transition and sources</summary>"
            f"<p>{esc(scene['visual'])}</p>"
            f"<p><b>Transition:</b> {esc(scene['transition'])}</p>"
            f"<p>{source_links}</p></details></article>"
        )
    photo_links = "".join(
        f"<li><b>{esc(photo['label'])}:</b> "
        f"<a href='{esc(photo['source_url'])}'>{esc(photo['author'])}</a> · "
        f"<a href='{esc(photo['license_url'])}'>{esc(photo['license'])}</a>. "
        f"{esc(photo['modifications'])}</li>"
        for photo in credits["photographs"]
    )
    palette = "".join(
        f'<span><i style="background:{value}"></i>{esc(key)} <code>{value}</code></span>'
        for key, value in story["palette"].items()
    )
    voices = []
    for voice in story["voice_samples"]["voices"]:
        description = "American English · " + ("feminine" if voice == "af_heart" else "masculine")
        voices.append(
            f'<article class="voice"><h3>{esc(voice)}</h3><p>{description}</p>'
            f'<audio controls preload="none" src="review/voice-{voice}.wav" '
            f'aria-label="Fifteen-second {voice} voice sample"></audio>'
            f'<a href="review/voice-{voice}.wav" download>Download 15-second WAV</a></article>'
        )
    page = (HERE / "review.template.html").read_text()
    page = (
        page.replace("VOICES", "".join(voices))
        .replace("SAMPLE", esc(story["voice_samples"]["text"]))
        .replace("CARDS", "\n".join(cards))
        .replace("WORD_COUNT", str(words))
        .replace("PALETTE", palette)
        .replace("PHOTO_LINKS", photo_links)
    )
    if allow_page_change:
        page = page.replace("Draft · awaiting your review", "Approved · af_heart selected")
        page = page.replace("What needs your decision", "Approved reference")
        page = page.replace(
            "Does the script tell the right story? Does the motion feel right? "
            "Which voice do you prefer: af_heart or am_michael?",
            "The owner approved the script, storyboard and motion style, with af_heart narration.",
        )
        page = page.replace(
            "The complete video and project-page player will be built after your approval. "
            "Nothing here is published.",
            '<a href="../../docs/index.html#explainer">'
            "Open the complete film on the project page</a>. "
            "The two short auditions and the motion study are retained as the approved reference.",
        )
    (HERE / "review.html").write_text(page.rstrip() + "\n")


def run_manim(cache, scene, output, static=False, index=None, scene_file=None):
    env = {**os.environ, "EXPLAINER_CACHE": str(cache)}
    if index is not None:
        env["EXPLAINER_FRAME"] = str(index)
    args = [
        sys.executable,
        "-m",
        "manim",
        "--config_file",
        str(HERE / "manim.cfg"),
        "--media_dir",
        str(cache / "manim"),
        "-o",
        output,
    ]
    if static:
        args.append("-s")
    args.extend([str(scene_file or HERE / "scenes.py"), scene])
    subprocess.run(args, check=True, env=env, cwd=HERE, stdin=subprocess.DEVNULL)
    suffix = "png" if static else "mp4"
    matches = list((cache / "manim").rglob(f"{output}.{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {output}.{suffix}, got {matches}")
    return matches[0]


def render_frames(cache):
    from PIL import Image, ImageOps

    frames = REVIEW / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, scene in enumerate(load("story.json")["scenes"]):
        name = f"{i + 1:02d}-{scene['id']}"
        path = run_manim(cache, "Storyboard", name, static=True, index=i)
        target = frames / f"{name}.png"
        shutil.copyfile(path, target)
        paths.append(target)
    sheet = Image.new("RGB", (1280, 1440), "#F7F8FA")
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            sheet.paste(ImageOps.contain(image, (640, 360)), ((i % 2) * 640, (i // 2) * 360))
    sheet.save(REVIEW / "storyboard-contact.png", optimize=True)
    poster = run_manim(cache, "Poster", "poster", static=True)
    shutil.copyfile(poster, REVIEW / "poster.png")


def timestamp(seconds, separator):
    value = round(seconds * 1000)
    hours, value = divmod(value, 3600000)
    minutes, value = divmod(value, 60000)
    secs, ms = divmod(value, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def render_preview(cache):
    cues = json.loads((cache / "preview-cues.json").read_text())
    for cue in cues:
        cue["text"] = re.sub(r"\s+([,.!?;:])", r"\1", cue["text"])
    cues[-1]["end"] = 10.0
    (cache / "preview-cues.json").write_text(json.dumps(cues, indent=2) + "\n")
    for kind in ["srt", "vtt"]:
        lines = ["WEBVTT", ""] if kind == "vtt" else []
        for i, cue in enumerate(cues, 1):
            assert "\n" not in cue["text"] and len(cue["text"]) <= CAPTION_MAX_CHARS
            if kind == "srt":
                lines.append(str(i))
            sep = "," if kind == "srt" else "."
            lines.extend(
                [
                    f"{timestamp(cue['start'], sep)} --> {timestamp(cue['end'], sep)}",
                    cue["text"],
                    "",
                ]
            )
        (REVIEW / f"preview.{kind}").write_text("\n".join(lines).rstrip() + "\n")
    source = run_manim(cache, "Preview", "motion-study")
    ffmpeg = Path(sys.executable).parent / "ffmpeg"
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
            str(cache / "preview.wav"),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            "10",
            "-c:v",
            "libx264",
            "-threads",
            "4",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "slow",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(REVIEW / "preview.mp4"),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )


def check_outputs(allow_page_change=False):
    import av
    import numpy as np
    import soundfile as sf
    from PIL import Image

    validate_sources(allow_page_change=allow_page_change)
    story = load("story.json")
    receipt = {
        "status": "approved-reference" if allow_page_change else "awaiting-owner-review",
        "narration_words": sum(len(s["narration"].split()) for s in story["scenes"]),
        "full_video_rendered": (REPO / "docs/media/explainer.mp4").is_file(),
        "project_page_modified": allow_page_change,
        "frames": [],
        "voices": [],
    }
    for i, scene in enumerate(story["scenes"], 1):
        path = REVIEW / "frames" / f"{i:02d}-{scene['id']}.png"
        with Image.open(path) as image:
            assert image.size == (1920, 1080)
        receipt["frames"].append({"file": str(path.relative_to(HERE)), "sha256": sha(path)})
    with Image.open(REVIEW / "poster.png") as poster:
        assert poster.size == (1920, 1080)
    receipt["poster_sha256"] = sha(REVIEW / "poster.png")
    for voice in story["voice_samples"]["voices"]:
        path = REVIEW / f"voice-{voice}.wav"
        samples, rate = sf.read(path)
        assert rate == 24000 and len(samples) == 15 * rate
        assert np.isfinite(samples).all() and 0.01 < float(np.abs(samples).max()) < 0.99
        receipt["voices"].append(
            {"file": path.name, "duration_s": 15, "sample_rate": rate, "sha256": sha(path)}
        )
    path = REVIEW / "preview.mp4"
    with av.open(path) as video:
        stream = video.streams.video[0]
        audio = video.streams.audio[0]
        assert (stream.width, stream.height) == (1920, 1080)
        assert float(stream.average_rate) == 30 and stream.codec_context.name == "h264"
        assert audio.codec_context.name == "aac"
        frames = sum(1 for _ in video.decode(video=0))
        assert frames == 300, frames
        duration = float(stream.duration * stream.time_base)
        assert abs(duration - 10) < 1 / 30
    blob = path.read_bytes()
    assert blob.find(b"moov") < blob.find(b"mdat"), "MP4 is not fast-start"
    assert path.stat().st_size < 5 * 1024 * 1024
    receipt["preview"] = {
        "duration_s": duration,
        "frames": frames,
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "video_codec": "h264",
        "audio_codec": "aac",
        "fast_start": True,
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }
    for name in ["preview.srt", "preview.vtt", "transcript.txt"]:
        assert (REVIEW / name).stat().st_size > 0
    (REVIEW / "checks.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["sources", "docs", "frames", "preview", "check"])
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--allow-page-change", action="store_true")
    args = parser.parse_args()
    if args.stage == "sources":
        validate_sources(allow_page_change=args.allow_page_change)
    elif args.stage == "docs":
        write_docs(allow_page_change=args.allow_page_change)
    elif args.stage == "check":
        check_outputs(allow_page_change=args.allow_page_change)
    else:
        if args.cache is None:
            parser.error("--cache is required for rendering")
        cache = args.cache.resolve()
        cache.mkdir(parents=True, exist_ok=True)
        if args.stage == "frames":
            render_frames(cache)
        else:
            render_preview(cache)
