"""Visual identity: palette, typography, and text constructors.

Components never specify colours or font sizes directly -- they ask the theme
for a role ("heading", "accent"). That is what lets a channel's visual identity
be swapped by config, and it is why `theme` is a cache-key input (D-004).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from manim import MathTex, Mobject, Tex, Text, VMobject

from chalkdust.scenes.regions import tag_font_size


@dataclass(frozen=True)
class Palette:
    bg: str
    fg: str          # primary text
    muted: str       # secondary text, labels
    accent: str      # the one colour that means "look here"
    accent_alt: str  # second emphasis, used sparingly
    success: str
    danger: str


@dataclass(frozen=True)
class Typography:
    """Font families plus sizes in Manim font_size units.

    If a font is not installed, Pango substitutes silently -- your render will
    succeed and look wrong. `check_fonts()` below surfaces that early.
    """

    heading_font: str
    body_font: str
    mono_font: str

    title: float = 60.0
    heading: float = 44.0
    body: float = 32.0
    caption: float = 24.0
    mono: float = 28.0


@dataclass(frozen=True)
class Theme:
    name: str
    palette: Palette
    type: Typography
    # Motion language: how fast things move in this channel's videos.
    fade_time: float = 0.4
    write_time: float = 0.8


DEFAULT = Theme(
    name="default",
    palette=Palette(
        bg="#0E1116",
        fg="#E6EDF3",
        muted="#8B949E",
        accent="#58A6FF",
        accent_alt="#F0883E",
        success="#3FB950",
        danger="#F85149",
    ),
    type=Typography(
        heading_font="Archivo",
        body_font="Inter",
        mono_font="JetBrains Mono",
    ),
)

THEMES: dict[str, Theme] = {"default": DEFAULT}


def get_theme(name: str) -> Theme:
    if name not in THEMES:
        raise KeyError(f"unknown theme {name!r}; available: {sorted(THEMES)}")
    return THEMES[name]


def check_fonts(theme: Theme) -> list[str]:
    """Return theme fonts that are not installed.

    Call this once at startup. A missing font does not raise -- Pango falls
    back -- so this is the only way to notice before the render looks wrong.
    """
    try:
        import manimpango

        available = set(manimpango.list_fonts())
    except Exception:
        return []  # can't check; don't block the render
    wanted = {theme.type.heading_font, theme.type.body_font, theme.type.mono_font}
    return sorted(f for f in wanted if f not in available)


# --- text constructors ------------------------------------------------------
# Every text mobject in the system should come from one of these. They apply
# the theme and tag the font size so the legibility check in regions.py works.


def title_text(s: str, theme: Theme, color: str | None = None) -> Text:
    t = Text(s, font=theme.type.heading_font, font_size=theme.type.title,
             color=color or theme.palette.fg, weight="BOLD")
    return tag_font_size(t, theme.type.title)


def heading_text(s: str, theme: Theme, color: str | None = None) -> Text:
    t = Text(s, font=theme.type.heading_font, font_size=theme.type.heading,
             color=color or theme.palette.fg, weight="SEMIBOLD")
    return tag_font_size(t, theme.type.heading)


def body_text(s: str, theme: Theme, color: str | None = None) -> Text:
    t = Text(s, font=theme.type.body_font, font_size=theme.type.body,
             color=color or theme.palette.fg)
    return tag_font_size(t, theme.type.body)


def caption_text(s: str, theme: Theme, color: str | None = None) -> Text:
    t = Text(s, font=theme.type.body_font, font_size=theme.type.caption,
             color=color or theme.palette.muted)
    return tag_font_size(t, theme.type.caption)


def mono_text(s: str, theme: Theme, color: str | None = None) -> Text:
    t = Text(s, font=theme.type.mono_font, font_size=theme.type.mono,
             color=color or theme.palette.fg)
    return tag_font_size(t, theme.type.mono)


def math(s: str, theme: Theme, size: float | None = None,
         color: str | None = None) -> MathTex:
    """LaTeX maths. Note MathTex font_size behaves differently from Text --
    the same numeric value renders visually smaller, so we bump it."""
    size = size or theme.type.heading
    m = MathTex(s, font_size=size * 1.2, color=color or theme.palette.fg)
    return tag_font_size(m, size)


def emphasize(mob: VMobject, theme: Theme) -> VMobject:
    """Recolour to the accent. The one gesture that means 'this matters'."""
    return mob.set_color(theme.palette.accent)

# --- font resolution --------------------------------------------------------
# Pango substitutes missing fonts silently, so a render succeeds and looks
# wrong. Resolving explicitly at startup makes the substitution loud.

FALLBACKS = {
    "heading": ["Archivo", "Helvetica Neue", "Arial"],
    "body": ["Inter", "Helvetica Neue", "Arial"],
    "mono": ["JetBrains Mono", "Menlo", "Courier New"],
}


def _installed_fonts() -> set[str]:
    try:
        import manimpango

        return set(manimpango.list_fonts())
    except Exception:
        return set()


def resolve_fonts(theme: Theme, warn: bool = True) -> Theme:
    """Return a copy of `theme` with any missing font swapped for a fallback.

    Called once per scene construction. If nothing in the fallback chain is
    installed we leave the original name and let Pango decide -- but by then
    the warning has already been printed.
    """
    available = _installed_fonts()
    if not available:
        return theme

    def pick(role: str, wanted: str) -> str:
        if wanted in available:
            return wanted
        for candidate in FALLBACKS[role]:
            if candidate in available:
                if warn:
                    print(f"[theme] {wanted!r} missing, using {candidate!r} for {role}")
                return candidate
        return wanted

    t = theme.type
    return Theme(
        name=theme.name,
        palette=theme.palette,
        type=Typography(
            heading_font=pick("heading", t.heading_font),
            body_font=pick("body", t.body_font),
            mono_font=pick("mono", t.mono_font),
            title=t.title,
            heading=t.heading,
            body=t.body,
            caption=t.caption,
            mono=t.mono,
        ),
        fade_time=theme.fade_time,
        write_time=theme.write_time,
    )
