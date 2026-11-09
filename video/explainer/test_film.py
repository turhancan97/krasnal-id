"""Offline protocol checks; no model, image decoder or network is needed."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from film_audio import allocate_frames, validate_approval
from narrate import CAPTION_MAX_CHARS, RATE, cues_for


class FilmProtocolTests(unittest.TestCase):
    def test_natural_speech_rebalances_chapters_and_preserves_full_credits(self):
        speech = [9.725, 19.125, 18.25, 20.325, 19.025, 15.025, 6.775]
        frames = allocate_frames(speech)
        self.assertEqual(sum(frames), 3600)
        self.assertEqual(frames[-1], 240)
        self.assertGreater(frames[1], 18 * 30)
        for count, duration in zip(frames[:-1], speech, strict=True):
            self.assertGreaterEqual(count / 30, duration + 0.35)

    def test_long_speech_is_refused_instead_of_cut_or_accelerated(self):
        with self.assertRaisesRegex(ValueError, "do not render or accelerate"):
            allocate_frames([20] * 7)

    def test_subtitles_preserve_decimal_values_and_remove_token_spacing(self):
        tokens = [
            SimpleNamespace(text=word, start_ts=i * 0.4, end_ts=i * 0.4 + 0.3)
            for i, word in enumerate(["DINOv2", "reaches", "93.4", "percent", "."])
        ]
        result = SimpleNamespace(tokens=tokens, audio=[None] * (3 * RATE))
        cues = cues_for("DINOv2 reaches 93.4 percent.", [result], offset=0.15)
        self.assertEqual(cues[0]["text"], "DINOv2 reaches 93.4 percent.")
        self.assertEqual(cues[0]["start"], 0.15)
        self.assertLessEqual(cues[-1]["end"], 3.15)

    def test_late_punctuation_stays_with_its_sentence(self):
        tokens = [
            SimpleNamespace(text=word, start_ts=start, end_ts=end)
            for word, start, end in [
                ("A", 0, 0.5),
                ("caption", 0.5, 1),
                ("ends", 1, 2),
                ("here", 2, 2.99),
                (".", 3.21, 3.25),
            ]
        ]
        result = SimpleNamespace(tokens=tokens, audio=range(round(4.5 * RATE)))
        cues = cues_for("A caption ends here.", [result])
        self.assertEqual([cue["text"] for cue in cues], ["A caption ends here."])

    def test_long_narration_is_split_into_single_line_cues(self):
        text = (
            "LoFTR matches local details across photographs. Geometric consistency adds "
            "evidence to visual similarity, moving a better match up the shortlist."
        )
        tokens = [
            SimpleNamespace(text=word, start_ts=i * 0.35, end_ts=i * 0.35 + 0.3)
            for i, word in enumerate(text.split())
        ]
        result = SimpleNamespace(tokens=tokens, audio=range(12 * RATE))
        cues = cues_for(text, [result])
        self.assertGreater(len(cues), 1)
        self.assertEqual(" ".join(c["text"] for c in cues), text)
        for cue in cues:
            self.assertNotIn("\n", cue["text"])
            self.assertLessEqual(len(cue["text"]), CAPTION_MAX_CHARS)

    def test_changed_script_invalidates_approval_before_any_render(self):
        approval = {
            "approval_text": "Approve with af_heart",
            "voice": "af_heart",
            "story_sha256": "approved-script",
        }
        with (
            patch("film_audio.load", return_value=approval),
            patch("film_audio.sha", return_value="changed-script"),
            self.assertRaisesRegex(AssertionError, "Approved story changed"),
        ):
            validate_approval()


if __name__ == "__main__":
    unittest.main()
