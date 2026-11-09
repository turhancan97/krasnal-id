"""Review-only Manim scenes. Photographs and measured series are read, never rebuilt."""

import json
import os
import textwrap
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    Arc,
    Arrow,
    Circle,
    Create,
    DashedVMobject,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    GrowArrow,
    ImageMobject,
    LaggedStart,
    Line,
    Rectangle,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    Transform,
    VGroup,
    VMobject,
    config,
    register_font,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
STORY = json.loads((HERE / "story.json").read_text())
CLAIMS = json.loads((HERE / "claims.json").read_text())["claims"]
PHOTOS = json.loads((HERE / "credits.json").read_text())["photographs"]
P = STORY["palette"]
BG, INK, MUTED, RULE = STORY["background"], STORY["ink"], "#57606A", "#DDE1E3"
FONT = "TeX Gyre Heros"
config.frame_width, config.frame_height = 16, 9
config.background_color = BG


def label(words, size=30, color=INK, width=None):
    obj = Text(words, font=FONT, font_size=size, color=color, line_spacing=0.8)
    if width and obj.width > width:
        obj.scale_to_fit_width(width)
    return obj


def at(obj, x, y):
    return obj.move_to([x, y, 0])


def pill(words, color=P["blue"], width=3.4):
    frame = RoundedRectangle(
        width=width,
        height=0.48,
        corner_radius=0.13,
        stroke_width=0,
        fill_color=color,
        fill_opacity=0.13,
    )
    return VGroup(frame, label(words, 21, color, width - 0.24))


def card(index, width=2.35):
    photo = PHOTOS[index]
    frame = RoundedRectangle(
        width=width,
        height=3.12,
        corner_radius=0.13,
        stroke_color=RULE,
        stroke_width=1.6,
        fill_color=WHITE,
        fill_opacity=1,
    )
    image = ImageMobject(str(REPO / photo["path"]))
    image.height = 2.22
    if image.width > width - 0.18:
        image.scale_to_fit_width(width - 0.18)
    image.shift(UP * 0.26)
    title = at(label(photo["label"], 23, width=width - 0.18), 0, -1.03)
    short = [
        "Adalbertus · BY-SA 4.0",
        "Pnapora · BY-SA 3.0",
        "Krasnoludek · public domain",
        "Pnapora · BY-SA 3.0",
    ][index]
    credit = at(label(short, 16, MUTED, width - 0.18), 0, -1.35)
    return Group(frame, image, title, credit)


def header(index, title=None):
    scene = STORY["scenes"][index]
    eyebrow = label(f"DOPPELDWARF-306   /   {index + 1:02d}", 20, MUTED)
    eyebrow.to_edge(LEFT, buff=0.72).set_y(3.95)
    heading = label(title or scene["title"], 48, width=14.5)
    heading.to_edge(LEFT, buff=0.72).set_y(3.12)
    group = VGroup(eyebrow, heading)
    if index == 3:
        simulation = label("IN SIMULATION", 24, P["purple"]).to_edge(RIGHT, buff=0.72).set_y(3.95)
        group.add(simulation)
    return group


CAPTIONS = [
    "Given a photograph, which individual statue are we looking at?",
    "Cosine similarity compares these vectors.",
    "DINOv2's accuracy falls slowly; CLIP's falls faster.",
    "In simulation, a soft distance bonus works better.",
    "Learned geometric matching checks whether local details line up,",
    "Comparing the best two candidates exposes ambiguity.",
    "Explore the dataset, and try the identifier.",
    "",
]


def caption(words):
    lines = textwrap.wrap(words, width=44, break_long_words=False, break_on_hyphens=False)
    if len(lines) > 2:
        raise ValueError("Subtitle exceeds two lines")
    plate = at(
        RoundedRectangle(
            width=14.6,
            height=1.34,
            corner_radius=0.12,
            fill_color=WHITE,
            fill_opacity=1,
            stroke_width=1,
            stroke_color=RULE,
        ),
        0,
        -3.53,
    )
    if not words:
        return VGroup(plate)
    text = at(label("\n".join(lines), 46, width=13.9), 0, -3.53)
    if text.height > 1.03:
        text.scale_to_fit_height(1.03)
    return VGroup(plate, text)


def hook():
    query = at(card(0), -5.3, 0.35)
    refs = Group(
        *[at(card(i), x, 0.35) for i, x in zip([1, 2, 3], [-0.75, 2.3, 5.35], strict=True)]
    )
    arrow = Arrow([-3.9, 0.55, 0], [-2.25, 0.55, 0], color=P["mustard"], buff=0.12, stroke_width=7)
    facts = at(label("306 statues   ·   1,691 openly licensed photographs", 29, MUTED), 0, -1.93)
    return query, refs, arrow, facts


def retrieval():
    query = at(card(0, 1.9).scale(0.88), -6.0, 0.25)
    encoder = VGroup(
        RoundedRectangle(
            width=2.45,
            height=1.35,
            corner_radius=0.17,
            stroke_color=P["blue"],
            fill_color=P["blue"],
            fill_opacity=0.1,
        ),
        label("Frozen\nvision model", 29, width=2.15),
    )
    at(encoder, -3.15, 0.5)
    into = Arrow(query.get_right() + RIGHT * 0.04, encoder.get_left(), color=P["blue"], buff=0.15)
    out = Arrow(encoder.get_right(), [-0.7, 0.5, 0], color=P["blue"], buff=0.16)
    origin = np.array([-0.25, -1.05, 0])
    axes = VGroup(
        Line(origin + DOWN * 0.1, origin + UP * 3, color=RULE),
        Line(origin + LEFT * 0.1, origin + RIGHT * 3.0, color=RULE),
    )
    vectors = VGroup(
        Arrow(origin, origin + np.array([1.95, 2.1, 0]), color=P["blue"], buff=0, stroke_width=7),
        Arrow(
            origin, origin + np.array([2.17, 1.82, 0]), color=P["mustard"], buff=0, stroke_width=7
        ),
        Arrow(
            origin, origin + np.array([2.40, 0.55, 0]), color=P["orange"], buff=0, stroke_width=7
        ),
    )
    arc = Arc(
        radius=1.0,
        start_angle=np.arctan2(1.82, 2.17),
        angle=0.13,
        arc_center=origin,
        color=P["purple"],
        stroke_width=5,
    )
    info = at(label("Illustrative vectors", 23, MUTED), 1.2, -1.65)
    ranking = VGroup()
    for i, (word, color) in enumerate(
        [("Solne", P["blue"]), ("Oławskie", P["orange"]), ("Świdnickie", P["purple"])]
    ):
        row = VGroup(
            RoundedRectangle(
                width=3.1,
                height=0.64,
                corner_radius=0.09,
                fill_color=color,
                fill_opacity=0.11,
                stroke_color=color,
                stroke_width=2,
            ),
            label(f"{i + 1}   {word}", 28, width=2.8),
        )
        ranking.add(at(row, 5.45, 1.22 - i * 0.85))
    rank_title = at(label("Best reference\nper statue", 24, MUTED), 5.45, 2.05)
    foot = at(label("Cosine similarity   ·   query photograph held out", 23, MUTED), 0, -2.25)
    return query, encoder, into, out, axes, vectors, arc, info, ranking, rank_title, foot


def pool_chart():
    facts = CLAIMS["pool"]["facts"]
    x0, y0, width, height = -5.65, -1.25, 10.6, 3.45

    def pos(n, acc):
        return np.array(
            [x0 + width * np.log2(n / 2) / np.log2(306 / 2), y0 + height * (acc - 75) / 25, 0]
        )

    chart = VGroup()
    for acc in [80, 90, 100]:
        chart.add(Line(pos(2, acc), pos(306, acc), color=RULE, stroke_width=1))
        chart.add(at(label(f"{acc}%", 24, MUTED), x0 - 0.65, pos(2, acc)[1]))
    chart.add(Line(pos(2, 75), pos(306, 75), color=MUTED, stroke_width=2))
    for n in [2, 10, 50, 306]:
        chart.add(at(label(str(n), 24, MUTED), pos(n, 75)[0], y0 - 0.35))
    chart.add(at(label("Candidate statues · logarithmic scale", 25, MUTED), -0.4, -2.1))
    chart.add(at(label("Top-1 accuracy", 24, MUTED), -4.3, 2.5))
    for key, color, title in [
        ("dinov2_percent", P["blue"], "DINOv2"),
        ("clip_percent", P["orange"], "CLIP"),
    ]:
        points = [pos(n, acc) for n, acc in zip(facts["candidate_counts"], facts[key], strict=True)]
        curve = VMobject(color=color, stroke_width=7).set_points_as_corners(points)
        chart.add(curve, *[Dot(point, radius=0.055, color=color) for point in points])
        chart.add(at(label(f"{title}\n{facts[key][-1]}%", 27, color), 6.0, points[-1][1]))
    return chart


def location_map():
    panel = VGroup()
    # Positions are illustrative: this is not a city map or a new experiment.
    for center, soft in [(np.array([-4.0, 0.35, 0]), False), (np.array([3.25, 0.35, 0]), True)]:
        if soft:
            for radius in np.linspace(1.8, 0.15, 18):
                panel.add(
                    Circle(
                        radius=radius,
                        color=P["teal"],
                        fill_color=P["teal"],
                        fill_opacity=0.018,
                        stroke_width=0,
                    ).move_to(center)
                )
        else:
            panel.add(
                DashedVMobject(Circle(radius=1.6, color=MUTED, stroke_width=3).move_to(center))
            )
        for x, y, inside in [
            (-0.8, 0.65, True),
            (0.55, 0.8, True),
            (0.9, -0.65, True),
            (-1.25, -0.9, True),
            (1.95, 0.55, False),
        ]:
            panel.add(
                Dot(
                    center + np.array([x, y, 0]),
                    radius=0.11,
                    color=P["mustard"] if inside or soft else "#B8BEC4",
                )
            )
        panel.add(Dot(center, radius=0.10, color=P["blue"]))
    panel.add(
        at(label("Hard filter", 31), -4, 2.3), at(label("Soft distance bonus", 31), 3.25, 2.3)
    )
    panel.add(at(label("Nearby lookalikes stay together", 23, MUTED), -4, -1.85))
    panel.add(at(label("Photo only → with location", 22, MUTED), 3.25, -0.91))
    panel.add(at(label("93.4%  →  98.7%", 40, P["teal"]), 3.25, -1.48))
    panel.add(at(label("294 placed statues · user 25 m away", 23, MUTED), 3.25, -2.03))
    panel.add(at(label("Schematic positions · held-out research weights", 21, MUTED), 0, -2.55))
    return panel


def verification():
    left, right = at(card(0, 2.5), -4.8, 0.2), at(card(1, 2.5), 0.3, 0.2)
    connections = VGroup()
    for a, b in [
        ((-0.3, 0.8), (0.0, 0.48)),
        ((0.25, 0.1), (0.30, 0.1)),
        ((-0.15, -0.4), (-0.1, -0.45)),
    ]:
        start = np.array([-4.8 + a[0], 0.6 + a[1], 0])
        end = np.array([0.3 + b[0], 0.6 + b[1], 0])
        connections.add(
            Line(start, end, color=P["purple"], stroke_width=2.5),
            Dot(start, radius=0.055, color=P["purple"]),
            Dot(end, radius=0.055, color=P["purple"]),
        )
    notes = VGroup(
        at(label("Learned matching", 29, P["purple"]), 4.75, 1.15),
        at(label("checks local details", 24, MUTED), 4.75, 0.65),
        at(label("SIFT", 29, P["pink"]), 4.75, -0.3),
        at(label("no measurable gain", 24, MUTED), 4.75, -0.8),
    )
    foot = at(label("Photographer withheld · illustrative connections", 23, MUTED), 0, -2.05)
    return Group(left, right, connections, notes, foot)


def uncertainty():
    group = VGroup()
    for i, (name, width, color) in enumerate(
        [("Candidate A", 8.4, P["blue"]), ("Candidate B", 7.9, P["orange"])]
    ):
        y = 1.4 - i * 1.15
        bar = Rectangle(
            width=width, height=0.58, stroke_width=0, fill_color=color, fill_opacity=0.83
        )
        bar.move_to([-4.4 + width / 2, y, 0])
        group.add(at(label(name, 25, MUTED), -5.9, y), bar)
    group.add(
        Line([3.5, 2.1, 0], [4, 2.1, 0], color=P["purple"], stroke_width=4),
        Line([3.5, 1.9, 0], [3.5, 2.25, 0], color=P["purple"], stroke_width=3),
        Line([4, 1.9, 0], [4, 2.25, 0], color=P["purple"], stroke_width=3),
        at(label("Small margin", 28, P["purple"]), 5.4, 2.1),
    )
    group.add(at(label("Illustrative similarity · no calibrated probability", 23, MUTED), 0, -0.55))
    for x, title, sub, color in [
        (-3.6, "Research", "Unknown-statue evaluation", P["purple"]),
        (3.6, "Browser demo", "Separate close-call hint", P["teal"]),
    ]:
        box = RoundedRectangle(
            width=6.2,
            height=1.2,
            corner_radius=0.15,
            fill_color=color,
            fill_opacity=0.07,
            stroke_color=color,
            stroke_width=2,
        )
        group.add(
            at(box, x, -1.75),
            at(label(title, 28, color), x, -1.48),
            at(label(sub, 24, MUTED), x, -1.98),
        )
    return group


def takeaway():
    group = VGroup()
    for x, word, color in [
        (-4.7, "Candidate pool", P["blue"]),
        (0, "Location", P["teal"]),
        (4.7, "Photographer", P["purple"]),
    ]:
        circle = Circle(
            radius=0.7, color=color, fill_color=color, fill_opacity=0.12, stroke_width=4
        )
        group.add(at(circle, x, 1.5), at(label(word, 28, color), x, 0.15))
        if word == "Candidate pool":
            for dx, dy in [(-0.19, -0.19), (-0.19, 0.19), (0.19, -0.19), (0.19, 0.19)]:
                group.add(
                    at(
                        Square(
                            side_length=0.23,
                            stroke_color=color,
                            stroke_width=2,
                            fill_color=color,
                            fill_opacity=0.35,
                        ),
                        x + dx,
                        1.5 + dy,
                    )
                )
        elif word == "Location":
            group.add(
                at(Circle(radius=0.28, color=color, stroke_width=3), x, 1.55),
                Dot([x, 1.55, 0], radius=0.07, color=color),
                Line([x - 0.18, 1.34, 0], [x, 1.12, 0], color=color, stroke_width=3),
                Line([x, 1.12, 0], [x + 0.18, 1.34, 0], color=color, stroke_width=3),
            )
        else:
            group.add(
                at(
                    RoundedRectangle(
                        width=0.83,
                        height=0.53,
                        corner_radius=0.06,
                        stroke_color=color,
                        stroke_width=3,
                    ),
                    x,
                    1.48,
                ),
                at(Circle(radius=0.16, color=color, stroke_width=3), x, 1.48),
                at(
                    Rectangle(
                        width=0.24, height=0.09, stroke_width=0, fill_color=color, fill_opacity=1
                    ),
                    x - 0.18,
                    1.8,
                ),
            )
    group.add(
        Line([-4, 1.5, 0], [-0.7, 1.5, 0], color=RULE, stroke_width=4),
        Line([0.7, 1.5, 0], [4, 1.5, 0], color=RULE, stroke_width=4),
        at(label("Doppeldwarf-306", 52), 0, -1.0),
        at(label("turhancan97.github.io/krasnal-id/demo.html", 28, P["blue"], 13.9), 0, -2.0),
    )
    return group


def credits():
    group = VGroup()
    rows = [
        ("Query · Solne", "Adalbertus", "CC BY-SA 4.0"),
        ("Solne reference", "Pnapora", "CC BY-SA 3.0"),
        ("Oławskie reference", "Krasnoludek (WT-shared)", "Public domain"),
        ("Świdnickie reference", "Pnapora", "CC BY-SA 3.0"),
    ]
    for i, (role, author, licence) in enumerate(rows):
        y = 2.1 - i * 0.78
        group.add(
            at(label(role, 25, MUTED), -5.35, y),
            at(label(author, 27), -0.25, y),
            at(label(licence, 25, P["teal"]), 5.35, y),
        )
    group.add(
        at(label("Kokoro-82M · synthetic narration · stock voice", 27, MUTED), 0, -1.7),
        at(
            label(
                "Photographs fitted/cropped · full linked credits accompany the video", 22, MUTED
            ),
            0,
            -2.35,
        ),
        at(label("turhancan97.github.io/krasnal-id", 30, P["blue"]), 0, -3.3),
    )
    return group


def registered_fonts():
    stack = ExitStack()
    for filename in [
        "texgyreheros-regular.otf",
        "texgyreheros-bold.otf",
        "texgyreheros-italic.otf",
    ]:
        stack.enter_context(register_font(str(REPO / "src/krasnal_id/viz/fonts" / filename)))
    return stack


class Storyboard(Scene):
    def construct(self):
        with registered_fonts():
            index = int(os.environ.get("EXPLAINER_FRAME", "0"))
            self.add(header(index))
            factories = [
                lambda: Group(*hook()),
                lambda: Group(*retrieval()),
                pool_chart,
                location_map,
                verification,
                uncertainty,
                takeaway,
                credits,
            ]
            self.add(factories[index]())
            if index < 7:
                self.add(caption(CAPTIONS[index]))


class Poster(Scene):
    def construct(self):
        with registered_fonts():
            self.add(header(0), Group(*hook()))


class Preview(Scene):
    """A ten-second motion/voice study, not the complete explainer."""

    def construct(self):
        with registered_fonts():
            cache = Path(os.environ["EXPLAINER_CACHE"])
            cues = json.loads((cache / "preview-cues.json").read_text())
            active = [-1]
            subtitles = caption("")

            def update_subtitles(mob, dt):
                chosen = next(
                    (i for i, c in enumerate(cues) if c["start"] <= self.time < c["end"]), -1
                )
                if chosen != active[0]:
                    mob.become(caption(cues[chosen]["text"] if chosen >= 0 else ""))
                    active[0] = chosen

            subtitles.add_updater(update_subtitles)
            top = header(0)
            query, refs, arrow, facts = hook()
            self.add(top, query, subtitles)
            self.add_sound(str(cache / "preview.wav"))
            self.play(
                FadeIn(refs, shift=LEFT * 0.16), GrowArrow(arrow), FadeIn(facts), run_time=1.1
            )
            self.wait(max(0, cues[1]["start"] - self.time))
            rquery, encoder, into, out, axes, vectors, arc, info, ranking, rank_title, foot = (
                retrieval()
            )
            self.play(
                Transform(query, rquery),
                FadeOut(refs),
                FadeOut(arrow),
                FadeOut(facts),
                Transform(top, header(1)),
                run_time=1.0,
            )
            self.play(FadeIn(encoder), GrowArrow(into), GrowArrow(out), Create(axes), run_time=1.0)
            self.play(
                LaggedStart(*[GrowArrow(v) for v in vectors], lag_ratio=0.2),
                FadeIn(info),
                Create(arc),
                run_time=1.1,
            )
            self.wait(max(0, cues[2]["start"] + 1.15 - self.time))
            self.play(
                LaggedStart(*[FadeIn(row, shift=RIGHT * 0.15) for row in ranking], lag_ratio=0.15),
                FadeIn(rank_title),
                FadeIn(foot),
                run_time=1.1,
            )
            self.wait(10 - self.time)
            subtitles.remove_updater(update_subtitles)
