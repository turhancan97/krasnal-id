"""The approved eight-chapter film, timed against synthesized word positions."""

import json
import os
from pathlib import Path

import numpy as np
from manim import (
    LEFT,
    RIGHT,
    UP,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    GrowArrow,
    GrowFromCenter,
    Indicate,
    LaggedStart,
    Line,
    ScaleInPlace,
    Scene,
    Square,
    Succession,
    Transform,
    VGroup,
    VMobject,
)
from scenes import (
    CLAIMS,
    HERE,
    MUTED,
    RULE,
    P,
    at,
    caption,
    credits,
    header,
    hook,
    label,
    location_map,
    registered_fonts,
    retrieval,
    takeaway,
    uncertainty,
    verification,
)


class FullFilm(Scene):
    def wait_to(self, time):
        remaining = round((time - self.time) * 30) / 30
        if remaining > 0:
            self.wait(remaining)
        elif remaining < -1 / 30:
            raise ValueError(f"Animation overran narration beat by {-remaining:.3f}s")

    def mark(self, phrase, fallback):
        """Locate a phrase in the actual chapter word timing, with a bounded visual default."""
        words = self.chapter.get("words", [])

        def normal(value):
            return "".join(c for c in value.lower() if c.isalnum())

        target = normal(phrase)
        joined = "".join(normal(word["text"]) for word in words)
        position = joined.find(target)
        if position >= 0:
            cursor = 0
            for word in words:
                cursor += len(normal(word["text"]))
                if cursor > position:
                    return word["start"]
        return self.chapter["start"] + fallback

    def beat(self, phrase, fallback, *animations, seconds=0.9):
        time = self.mark(phrase, fallback)
        self.wait_to(max(self.time, time))
        remaining = self.chapter["end"] - self.time - 0.15
        runtime = min(seconds, remaining)
        if runtime < 0.1:
            raise ValueError("No time left for a narration-synchronised animation")
        self.play(*animations, run_time=round(runtime * 30) / 30)

    def consolidate(self, *objects):
        self.body = Group(*objects)
        self.add(self.body)

    def enter(self, index):
        self.chapter = self.timing["chapters"][index]
        self.wait_to(self.chapter["start"])
        new = header(index)
        if self.body is not None:
            self.play(
                FadeOut(self.body, scale=0.96),
                Succession(FadeOut(self.heading), FadeIn(new)),
                run_time=0.6,
            )
            self.heading = new
        else:
            self.heading = new
            self.add(new)

    def intro(self):
        self.enter(0)
        query, refs, arrow, facts = hook()
        self.play(FadeIn(query, shift=RIGHT * 0.15), run_time=0.8)
        self.beat(
            "bronze dwarves",
            1.0,
            LaggedStart(*[FadeIn(card, shift=LEFT * 0.15) for card in refs], lag_ratio=0.18),
            seconds=1.3,
        )
        self.beat("Given a photograph", 4.0, GrowArrow(arrow))
        self.beat("Meet Doppeldwarf-306", 8.0, FadeIn(facts, shift=UP * 0.12))
        self.consolidate(query, refs, arrow, facts)
        self.intro_objects = (query, refs, arrow, facts)
        self.wait_to(self.chapter["end"])

    def vectors(self):
        self.chapter = self.timing["chapters"][1]
        self.wait_to(self.chapter["start"])
        query, refs, old_arrow, facts = self.intro_objects
        (new_query, encoder, into, out, axes, vectors, arc, info, ranking, rank_title, foot) = (
            retrieval()
        )
        new_heading = header(1)
        self.play(
            Transform(query, new_query),
            FadeOut(refs),
            FadeOut(old_arrow),
            FadeOut(facts),
            Succession(FadeOut(self.heading), FadeIn(new_heading)),
            FadeIn(encoder),
            run_time=1.0,
        )
        self.heading = new_heading
        self.beat("turns each image", 1.5, GrowArrow(into), GrowArrow(out), seconds=1)
        self.beat(
            "into a vector",
            3.0,
            Create(axes),
            LaggedStart(*[GrowArrow(v) for v in vectors], lag_ratio=0.2),
            FadeIn(info),
            seconds=1.4,
        )
        self.beat(
            "Cosine similarity",
            6.0,
            Create(arc),
            Indicate(vectors[0], color=P["blue"]),
            seconds=1.2,
        )
        self.beat("Similar directions", 8.0, Indicate(vectors[1], color=P["mustard"]))
        self.beat(
            "We rank each statue",
            11.0,
            LaggedStart(*[FadeIn(row, shift=LEFT * 0.15) for row in ranking], lag_ratio=0.2),
            FadeIn(rank_title),
            FadeIn(foot),
            seconds=1.3,
        )
        self.consolidate(
            query, encoder, into, out, axes, vectors, arc, info, ranking, rank_title, foot
        )
        self.wait_to(self.chapter["end"])

    def pool(self):
        self.enter(2)
        grid = VGroup(
            *[
                Square(
                    side_length=0.37,
                    stroke_width=2,
                    stroke_color=RULE,
                    fill_color=P["mustard"],
                    fill_opacity=0.23,
                )
                for _ in range(40)
            ]
        ).arrange_in_grid(rows=5, buff=0.18)
        at(grid, 0, 0.25)
        truth = grid[0]
        truth.set_stroke(P["green"], width=4).set_fill(P["green"], opacity=0.5)
        note = at(label("Correct statue always retained", 28, MUTED), 0, -1.9)
        self.play(FadeIn(VGroup(*grid[:5])), FadeIn(note), run_time=0.8)
        self.beat(
            "more candidates",
            1.0,
            LaggedStart(*[GrowFromCenter(cell) for cell in grid[5:]], lag_ratio=0.025),
            seconds=1.4,
        )
        self.beat("repeatedly sample larger", 5.0, Indicate(truth, color=P["green"]))
        axes, dino, clip, dots_dino, dots_clip, names = self.pool_plot()
        self.wait_to(max(self.time, self.mark("candidate pools", 7.5)))
        self.play(FadeOut(grid, scale=0.8), FadeOut(note), FadeIn(axes), run_time=0.7)
        self.beat(
            "DINOv2's accuracy",
            10.0,
            Create(dino),
            FadeIn(dots_dino),
            FadeIn(names[0]),
            seconds=1.5,
        )
        self.beat(
            "CLIP's falls", 12.0, Create(clip), FadeIn(dots_clip), FadeIn(names[1]), seconds=1.3
        )
        self.consolidate(axes, dino, clip, dots_dino, dots_clip, names)
        self.wait_to(self.chapter["end"])

    @staticmethod
    def pool_plot():
        facts = CLAIMS["pool"]["facts"]

        def pos(n, accuracy):
            return np.array(
                [
                    -5.65 + 10.6 * np.log2(n / 2) / np.log2(306 / 2),
                    -1.25 + 3.45 * (accuracy - 75) / 25,
                    0,
                ]
            )

        axes = VGroup()
        for value in [80, 90, 100]:
            axes.add(
                Line(pos(2, value), pos(306, value), color=RULE, stroke_width=1),
                at(label(f"{value}%", 24, MUTED), -6.3, pos(2, value)[1]),
            )
        axes.add(Line(pos(2, 75), pos(306, 75), color=MUTED, stroke_width=2))
        for n in [2, 10, 50, 306]:
            axes.add(at(label(str(n), 24, MUTED), pos(n, 75)[0], -1.6))
        axes.add(
            at(label("Candidate statues · logarithmic scale", 25, MUTED), -0.4, -2.1),
            at(label("Top-1 accuracy", 24, MUTED), -4.3, 2.5),
            at(label("Research pipeline · leave-one-out", 21, MUTED), 0, -2.55),
        )
        curves, dots, names = [], [], VGroup()
        for key, color, name in [
            ("dinov2_percent", P["blue"], "DINOv2"),
            ("clip_percent", P["orange"], "CLIP"),
        ]:
            points = [pos(n, a) for n, a in zip(facts["candidate_counts"], facts[key], strict=True)]
            curves.append(VMobject(color=color, stroke_width=7).set_points_as_corners(points))
            dots.append(VGroup(*[Dot(point, radius=0.055, color=color) for point in points]))
            names.add(at(label(f"{name}\n{facts[key][-1]}%", 27, color), 6.0, points[-1][1]))
        return axes, *curves, *dots, names

    def location(self):
        self.enter(3)
        panel = location_map()
        left = VGroup(*[obj for obj in panel if obj.get_center()[0] < -1])
        right = VGroup(*[obj for obj in panel if obj.get_center()[0] > 1])
        shared = VGroup(*[obj for obj in panel if abs(obj.get_center()[0]) <= 1])
        self.play(FadeIn(left), FadeIn(shared), run_time=0.9)
        self.beat("hard geographic filter", 4.0, Indicate(left[0], color=P["mustard"]))
        self.beat("a soft distance bonus", 7.0, FadeIn(right, shift=LEFT * 0.25), seconds=1.3)
        self.beat("DINOv2 rises", 12.0, Indicate(right, scale_factor=1.02, color=P["teal"]))
        self.consolidate(panel)
        self.wait_to(self.chapter["end"])

    def geometry(self):
        self.enter(4)
        panel = verification()
        left, right, lines, notes, foot = panel
        self.play(
            FadeIn(left, shift=RIGHT * 0.18),
            FadeIn(right, shift=LEFT * 0.18),
            FadeIn(foot),
            run_time=1.0,
        )
        self.beat(
            "different viewpoints",
            1.8,
            Succession(ScaleInPlace(right, 1.025), ScaleInPlace(right, 1 / 1.025)),
        )
        scarcity = at(label("Fewer references also matter", 25, MUTED), -2.3, 2.0)
        self.beat("Fewer remaining", 6.0, FadeIn(scarcity))
        self.beat(
            "Learned geometric matching",
            10.0,
            LaggedStart(*[Create(line) for line in lines], lag_ratio=0.1),
            FadeIn(VGroup(*notes[:2])),
            seconds=1.5,
        )
        self.beat("Classical SIFT", 15.0, FadeIn(VGroup(*notes[2:])), seconds=1)
        self.consolidate(panel, scarcity)
        self.wait_to(self.chapter["end"])

    def margin(self):
        self.enter(5)
        panel = uncertainty()
        # Labels and bars precede the bracket and the two explicitly separated uses.
        first = VGroup(*panel[:4])
        bracket = VGroup(*panel[4:8])
        rest = VGroup(*panel[8:])
        second_bar = panel[3]
        target = second_bar.copy()
        left = second_bar.get_left()
        second_bar.stretch_to_fit_width(5.8).move_to(left + RIGHT * 2.9)
        self.play(FadeIn(first), run_time=0.9)
        self.beat(
            "Comparing the best two",
            3.0,
            Transform(second_bar, target),
            FadeIn(bracket),
            seconds=1.3,
        )
        self.beat("This margin reduces", 7.0, FadeIn(rest), seconds=1)
        self.consolidate(panel)
        self.wait_to(self.chapter["end"])

    def ending(self):
        self.enter(6)
        panel = takeaway()
        self.play(FadeIn(panel, shift=UP * 0.08), run_time=1.0)
        self.consolidate(panel)
        self.wait_to(self.chapter["end"])
        # All four credits stay fully visible for the last eight seconds.
        self.remove(self.body, self.heading, self.subtitles)
        self.subtitles.clear_updaters()
        self.heading = header(7)
        self.body = credits()
        self.add(self.heading, self.body)
        self.wait_to(120)

    def construct(self):
        with registered_fonts():
            cache = Path(os.environ["EXPLAINER_CACHE"])
            self.timing = json.loads((HERE / "production/timing.json").read_text())
            assert self.timing["voice"] == "af_heart"
            self.body = None
            self.heading = None
            self.subtitles = caption("")
            active = [-1]
            cues = self.timing["cues"]

            def update_subtitles(mob, dt):
                index = next(
                    (i for i, c in enumerate(cues) if c["start"] <= self.time < c["end"]), -1
                )
                if index != active[0]:
                    mob.become(caption(cues[index]["text"] if index >= 0 else ""))
                    mob.set_z_index(100, family=True)
                    active[0] = index

            self.subtitles.add_updater(update_subtitles)
            assert self.subtitles.has_time_based_updater(), "Captions must update during waits"
            self.add(self.subtitles)
            self.add_sound(str(cache / "film/narration.wav"))
            self.intro()
            self.vectors()
            self.pool()
            self.location()
            self.geometry()
            self.margin()
            self.ending()
