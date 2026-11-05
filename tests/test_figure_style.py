"""The shared publication style: layouts, palette roles, fonts and saving."""

from pathlib import Path

import pytest

from krasnal_id.viz import figure_style as style


def test_cvww_widths_match_the_kit() -> None:
    # 17 cm text block with a 0.8 cm gutter: figures drawn at these widths are
    # placed at \linewidth and never scaled.
    assert style.CVWW.full * 2.54 == pytest.approx(17.0)
    assert style.CVWW.column * 2.54 == pytest.approx(8.1)
    assert style.CVWW.backbones == ("dinov2", "clip", "dinov3")


def test_journal_layout_keeps_its_two_backbones() -> None:
    assert style.layout_named("journal").backbones == ("dinov2", "clip")


def test_unknown_layout_is_refused() -> None:
    with pytest.raises(style.FigureStyleError, match="unknown figure layout"):
        style.layout_named("poster")


def test_every_backbone_has_a_colour_a_label_and_a_marker() -> None:
    for backbone in style.CVWW.backbones:
        assert backbone in style.BACKBONE_COLOUR
        assert backbone in style.BACKBONE_LABEL
        assert backbone in style.BACKBONE_MARKER
    # Identity rests on more than hue: no two backbones share a marker.
    assert len(set(style.BACKBONE_MARKER.values())) == len(style.BACKBONE_MARKER)
    assert len(set(style.BACKBONE_COLOUR.values())) == len(style.BACKBONE_COLOUR)


def test_correctness_colours_are_not_backbone_colours() -> None:
    assert style.CORRECT not in style.BACKBONE_COLOUR.values()
    assert style.WRONG not in style.BACKBONE_COLOUR.values()


def test_minus_uses_a_true_minus_sign() -> None:
    assert style.minus("-0.79") == "\N{MINUS SIGN}0.79"


def test_missing_font_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(style, "FONT_DIRECTORY", tmp_path)
    with pytest.raises(style.FigureStyleError, match="figure font is missing"):
        style.register_fonts()


def test_vendored_font_registers_and_resolves() -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import font_manager

    style.register_fonts()
    style.register_fonts()  # idempotent
    path = font_manager.findfont(
        font_manager.FontProperties(family=style.FONT_FAMILY), fallback_to_default=False
    )
    assert Path(path).parent == style.FONT_DIRECTORY


def test_save_keeps_the_drawn_size(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    style.apply(plt)
    figure, axes = plt.subplots(figsize=(style.CVWW.column, 2.0), layout="constrained")
    axes.plot([0, 1], [0, 1])
    style.panel_tag(axes, "a")
    path = style.save(figure, tmp_path / "figure.png", dpi=100)
    plt.close(figure)
    with Image.open(path) as image:
        # No tight bounding box: the file is the size it was drawn at (matplotlib
        # truncates the pixel count rather than rounding it).
        assert image.size == (int(style.CVWW.column * 100), 200)
