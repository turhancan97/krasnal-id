"""Sample every burned-in cue and chapter transition for visual verification."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from build_review import HERE, REPO, sha


def inspect_frames(cache):
    from PIL import Image, ImageDraw, ImageOps

    timing = json.loads((HERE / "production/timing.json").read_text())
    movie = REPO / "docs/media/explainer.mp4"
    ffmpeg = Path(sys.executable).parent / "ffmpeg"
    qa = cache / "film/qa"
    qa.mkdir(parents=True, exist_ok=True)
    report = {
        "video_sha256": sha(movie),
        "cue_samples": [],
        "method": "Rendered cue samples, nonempty/changed text masks and visual contact sheets",
    }
    last_signature = None
    for index, cue in enumerate(timing["cues"], 1):
        t = (cue["start"] + cue["end"]) / 2
        path = qa / f"cue-{index:02d}.png"
        subprocess.run(
            [
                str(ffmpeg),
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-ss",
                str(t),
                "-i",
                str(movie),
                "-frames:v",
                "1",
                "-vf",
                "crop=1760:160:80:885",
                "-threads",
                "1",
                str(path),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
        )
        with Image.open(path) as image:
            mask = image.convert("L").resize((880, 80)).point(lambda p: 0 if p < 128 else 255)
            ink = list(mask.getdata()).count(0)
            signature = hashlib.sha256(mask.tobytes()).hexdigest()
        assert ink > 50, f"Blank burned-in cue {index}"
        assert signature != last_signature, f"Caption froze at cue {index}"
        last_signature = signature
        report["cue_samples"].append(
            {
                "cue": index,
                "time": t,
                "expected": cue["text"],
                "image": path.name,
                "text_mask_sha256": signature,
                "ink": ink,
            }
        )
    rows = (len(report["cue_samples"]) + 1) // 2
    sheet = Image.new("RGB", (1920, rows * 108), "#F7F8FA")
    for index, entry in enumerate(report["cue_samples"]):
        x = (index % 2) * 960
        y = (index // 2) * 108
        sheet.paste(ImageOps.contain(Image.open(qa / entry["image"]), (880, 80)), (x + 40, y + 26))
        ImageDraw.Draw(sheet).text(
            (x + 40, y + 7), f"Cue {entry['cue']:02d} at {entry['time']:.2f}s", fill="#1F2328"
        )
    sheet.save(qa / "caption-contact.png")
    for chapter in timing["chapters"][1:]:
        for delta, suffix in [(-0.3, "before"), (0.3, "during"), (1.2, "after")]:
            t = chapter["start"] + delta
            subprocess.run(
                [
                    str(ffmpeg),
                    "-nostdin",
                    "-y",
                    "-v",
                    "error",
                    "-ss",
                    str(t),
                    "-i",
                    str(movie),
                    "-frames:v",
                    "1",
                    "-threads",
                    "1",
                    "-q:v",
                    "2",
                    str(qa / f"transition-{chapter['id']}-{suffix}.jpg"),
                ],
                check=True,
                stdin=subprocess.DEVNULL,
            )
    files = sorted(qa.glob("transition-*.jpg"))
    sheet = Image.new("RGB", (1920, 7 * 385), "#F7F8FA")
    for index, path in enumerate(files):
        x = (index % 3) * 640
        y = (index // 3) * 385
        sheet.paste(ImageOps.contain(Image.open(path), (640, 360)), (x, y + 25))
        ImageDraw.Draw(sheet).text((x + 10, y + 5), path.stem, fill="#1F2328")
    sheet.save(qa / "transitions-contact.png")
    report["chapter_transitions"] = 7
    report["captions_nonempty_and_changing"] = True
    (HERE / "production/caption-frames.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(report['cue_samples'])} changing, nonempty rendered cues", flush=True)
    print(qa, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    inspect_frames(parser.parse_args().cache.resolve())
