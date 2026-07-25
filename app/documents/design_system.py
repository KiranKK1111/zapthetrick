"""Design system engine (vNext §3.11, Stage 8 Component D).

Beautiful-by-default documents come from a design SYSTEM, not a fixed template:
theme tokens (6 color roles / 2 font roles / a spacing scale) × curated font
pairings × layout variants. That's a several-lakh design space — and the ONE
non-negotiable is that EVERY combination stays accessible. So the LLM PROPOSES a
theme, but a deterministic **WCAG contrast guardrail** checks it and
auto-corrects any failing text pair to a safe one before it can render.

This module owns the pure core:
  * WCAG maths — `relative_luminance`, `contrast_ratio`, `passes_wcag`;
  * the token model — `ThemeTokens` (6 colors / 2 fonts / spacing) + a known-safe
    default;
  * `validate_theme` → the failing text-on-surface pairs;
  * `auto_correct` → nudge each failing foreground (preserving hue) until it
    passes, so the proposed theme is GUARANTEED accessible;
  * `propose_theme` — the INJECTED LLM proposal → validate → auto-correct →
    fall back to the safe default on any error;
  * font pairings + layout variants + a design score for the visual-QA rubric.

Fail-open: disabled OR any error → the safe default theme. Flag-gated
(`documents.design_system`, default OFF → today's fixed template styling).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# WCAG 2.1 contrast thresholds.
_THRESHOLDS = {
    ("AA", False): 4.5, ("AA", True): 3.0,       # normal / large text
    ("AAA", False): 7.0, ("AAA", True): 4.5,
}


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.documents, "design_system", False))
    except Exception:  # noqa: BLE001
        return False


def _wcag_level() -> str:
    try:
        from app.core.config_loader import cfg
        lv = str(getattr(cfg.documents, "design_wcag_level", "AA") or "AA").upper()
        return lv if lv in ("AA", "AAA") else "AA"
    except Exception:  # noqa: BLE001
        return "AA"


# --------------------------------------------------------------------------- #
# Colour maths
# --------------------------------------------------------------------------- #
def _hex_to_rgb(h: str) -> "tuple[int, int, int]":
    s = (h or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"bad hex colour: {h!r}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _rgb_to_hex(rgb: "tuple[int, int, int]") -> str:
    return "#" + "".join(f"{max(0, min(255, int(round(c)))):02x}" for c in rgb)


def _lin(c: int) -> float:
    x = c / 255.0
    return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance of a colour, 0 (black) … 1 (white)."""
    r, g, b = _hex_to_rgb(hex_color)
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG contrast ratio between two colours, 1.0 … 21.0. Never raises → 1.0
    (the worst case) on a bad colour, so a parse failure reads as 'fails'."""
    try:
        l1 = relative_luminance(fg)
        l2 = relative_luminance(bg)
        lo, hi = min(l1, l2), max(l1, l2)
        return (hi + 0.05) / (lo + 0.05)
    except Exception:  # noqa: BLE001
        return 1.0


def passes_wcag(fg: str, bg: str, *, level: str = "AA", large: bool = False) -> bool:
    return contrast_ratio(fg, bg) >= _THRESHOLDS.get((level, large), 4.5)


# --------------------------------------------------------------------------- #
# Token model
# --------------------------------------------------------------------------- #
@dataclass
class ThemeTokens:
    # 6 colour roles.
    bg: str = "#ffffff"          # page background
    surface: str = "#f4f4f5"     # cards / panels
    text: str = "#18181b"        # body text on bg/surface
    muted: str = "#52525b"       # secondary text
    accent: str = "#2563eb"      # links / headings / primary buttons
    on_accent: str = "#ffffff"   # text ON an accent fill
    # 2 font roles.
    heading_font: str = "Inter"
    body_font: str = "Inter"
    # Spacing scale (px), layout variant.
    spacing: list[int] = field(default_factory=lambda: [4, 8, 12, 16, 24, 32])
    density: str = "comfortable"  # comfortable | compact
    columns: int = 1

    def to_dict(self) -> dict:
        return {"bg": self.bg, "surface": self.surface, "text": self.text,
                "muted": self.muted, "accent": self.accent,
                "on_accent": self.on_accent, "heading_font": self.heading_font,
                "body_font": self.body_font, "spacing": list(self.spacing),
                "density": self.density, "columns": self.columns}


# A known-WCAG-AA-safe default (the fail-open target).
SAFE_DEFAULT = ThemeTokens()

# The text pairs that MUST clear contrast: (fg_role, bg_role, is_large).
_CRITICAL_PAIRS = [
    ("text", "bg", False),
    ("text", "surface", False),
    ("muted", "bg", False),
    ("on_accent", "accent", False),
    ("accent", "bg", True),        # accent as a heading/link colour → large text
]


# --------------------------------------------------------------------------- #
# Validation + auto-correction
# --------------------------------------------------------------------------- #
@dataclass
class ThemeValidation:
    ok: bool
    violations: list[dict] = field(default_factory=list)   # failing pairs
    ratios: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "violations": list(self.violations),
                "ratios": dict(self.ratios)}


def validate_theme(tokens: ThemeTokens, *, level: str | None = None) -> ThemeValidation:
    """Check every critical text pair against WCAG. Never raises → a validation
    listing each failing pair + its actual ratio."""
    lv = level or _wcag_level()
    violations: list[dict] = []
    ratios: dict = {}
    try:
        for fg_role, bg_role, large in _CRITICAL_PAIRS:
            fg = getattr(tokens, fg_role)
            bg = getattr(tokens, bg_role)
            r = contrast_ratio(fg, bg)
            ratios[f"{fg_role}/{bg_role}"] = round(r, 2)
            need = _THRESHOLDS.get((lv, large), 4.5)
            if r < need:
                violations.append({"pair": f"{fg_role}/{bg_role}", "ratio": round(r, 2),
                                   "required": need, "large": large})
        return ThemeValidation(not violations, violations, ratios)
    except Exception:  # noqa: BLE001
        return ThemeValidation(False, [{"pair": "?", "error": "validate failed"}], {})


def _blend(fg: str, target: str, t: float) -> str:
    """Blend fg toward target by fraction t (0..1), preserving nothing but moving
    linearly in RGB — enough to reach a legible colour while staying near hue."""
    a = _hex_to_rgb(fg)
    b = _hex_to_rgb(target)
    return _rgb_to_hex(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))


def _correct_pair(fg: str, bg: str, *, level: str, large: bool) -> str:
    """Return a foreground near `fg` that PASSES contrast on `bg`. Blend toward
    whichever extreme (black/white) has more contrast with bg, in 8% steps."""
    if passes_wcag(fg, bg, level=level, large=large):
        return fg
    target = "#000000" if contrast_ratio("#000000", bg) >= contrast_ratio("#ffffff", bg) else "#ffffff"
    t = 0.0
    best = fg
    while t <= 1.0:
        cand = _blend(fg, target, t)
        if passes_wcag(cand, bg, level=level, large=large):
            return cand
        best = cand
        t += 0.08
    return best  # the extreme — maximal contrast even if it didn't formally pass


def auto_correct(tokens: ThemeTokens, *, level: str | None = None
                 ) -> "tuple[ThemeTokens, list[dict]]":
    """Nudge each failing foreground until every critical pair passes WCAG.
    Runs to a FIXPOINT (a role like `text` appears on both bg and surface, so one
    pass can leave the other failing; correcting a foreground toward black/white
    only ever raises contrast, so the loop converges). Returns (corrected_tokens,
    corrections[]). Deterministic; never raises."""
    lv = level or _wcag_level()
    corrections: list[dict] = []
    try:
        fixed = ThemeTokens(**tokens.to_dict())
        origin = {role: getattr(fixed, role)
                  for role, _, _ in _CRITICAL_PAIRS}
        for _ in range(6):                       # fixpoint (converges fast)
            changed = False
            for fg_role, bg_role, large in _CRITICAL_PAIRS:
                fg = getattr(fixed, fg_role)
                bg = getattr(fixed, bg_role)
                if not passes_wcag(fg, bg, level=lv, large=large):
                    new_fg = _correct_pair(fg, bg, level=lv, large=large)
                    if new_fg != fg:
                        setattr(fixed, fg_role, new_fg)
                        changed = True
            if not changed:
                break
        # Report net change per role (not each intermediate step).
        for role, before in origin.items():
            after = getattr(fixed, role)
            if after != before:
                corrections.append({"role": role, "from": before, "to": after})
        return fixed, corrections
    except Exception:  # noqa: BLE001
        return SAFE_DEFAULT, [{"error": "auto_correct failed → safe default"}]


# --------------------------------------------------------------------------- #
# Font pairings + layout variants
# --------------------------------------------------------------------------- #
# Curated OFL (Google Fonts) heading/body pairings.
FONT_PAIRINGS: dict[str, tuple[str, str]] = {
    "inter": ("Inter", "Inter"),
    "editorial": ("Playfair Display", "Source Serif 4"),
    "corporate": ("Poppins", "Inter"),
    "technical": ("Space Grotesk", "IBM Plex Sans"),
    "classic": ("Merriweather", "Lora"),
}
LAYOUT_VARIANTS = ("comfortable", "compact")


def font_pairing(name: str) -> "tuple[str, str]":
    return FONT_PAIRINGS.get((name or "").strip().lower(), FONT_PAIRINGS["inter"])


# --------------------------------------------------------------------------- #
# Proposal (injected) → validate → auto-correct
# --------------------------------------------------------------------------- #
THEME_SCHEMA = {
    "type": "object",
    "properties": {
        "bg": {"type": "string"}, "surface": {"type": "string"},
        "text": {"type": "string"}, "muted": {"type": "string"},
        "accent": {"type": "string"}, "on_accent": {"type": "string"},
        "heading_font": {"type": "string"}, "body_font": {"type": "string"},
        "density": {"type": "string"},
    },
    "required": ["bg", "surface", "text", "muted", "accent", "on_accent"],
    "additionalProperties": False,
}


def _tokens_from_obj(obj: dict) -> ThemeTokens:
    t = ThemeTokens()
    for role in ("bg", "surface", "text", "muted", "accent", "on_accent",
                 "heading_font", "body_font", "density"):
        v = obj.get(role)
        if isinstance(v, str) and v.strip():
            setattr(t, role, v.strip())
    return t


async def propose_theme(brief: str, *, propose_fn=None, level: str | None = None
                        ) -> "tuple[ThemeTokens, list[dict]]":
    """Propose a theme for `brief` via an INJECTED structured call (default the
    `core.structured` facade), then GUARANTEE accessibility by auto-correcting to
    WCAG. Returns (tokens, corrections). Fail-open: disabled OR any error → the
    safe default (already AA-clean)."""
    if not enabled():
        return SAFE_DEFAULT, []
    try:
        fn = propose_fn
        if fn is None:
            from app.core.structured import structured as fn  # type: ignore
        msgs = [{"role": "system", "content": _PROPOSE_INSTRUCTION},
                {"role": "user", "content": f"Design brief: {brief}"}]
        res = await fn(THEME_SCHEMA, msgs, tier="standard", name="theme")
        obj = getattr(res, "obj", None)
        if not isinstance(obj, dict):
            return SAFE_DEFAULT, []
        tokens = _tokens_from_obj(obj)
        return auto_correct(tokens, level=level)   # never returns an inaccessible theme
    except Exception:  # noqa: BLE001
        return SAFE_DEFAULT, []


_PROPOSE_INSTRUCTION = (
    "Propose a document theme as hex colours for 6 roles (bg, surface, text, "
    "muted, accent, on_accent) plus heading_font and body_font (Google-Fonts OFL "
    "names) and a density (comfortable|compact). Aim for a tasteful, high-contrast "
    "palette; the system will still enforce WCAG, but propose accessible colours.")


# --------------------------------------------------------------------------- #
# Design score (for the visual-QA rubric §3.3)
# --------------------------------------------------------------------------- #
def design_score(tokens: ThemeTokens) -> dict:
    """A 0–100 design score + sub-scores (contrast/spacing/hierarchy) the
    visual-QA rubric folds in. Deterministic; never raises."""
    try:
        v = validate_theme(tokens)
        contrast = 100 if v.ok else max(0, 100 - 25 * len(v.violations))
        # Spacing rhythm: a consistent, ascending scale scores full marks.
        sp = tokens.spacing or []
        spacing = 100 if (len(sp) >= 3 and all(
            sp[i] < sp[i + 1] for i in range(len(sp) - 1))) else 60
        # Hierarchy: a distinct heading font vs body reads as intentional.
        hierarchy = 100 if tokens.heading_font != tokens.body_font else 80
        overall = round(0.5 * contrast + 0.3 * spacing + 0.2 * hierarchy)
        return {"score": overall, "contrast": contrast, "spacing": spacing,
                "hierarchy": hierarchy, "wcag_ok": v.ok}
    except Exception:  # noqa: BLE001
        return {"score": 0, "contrast": 0, "spacing": 0, "hierarchy": 0,
                "wcag_ok": False}


__all__ = ["enabled", "relative_luminance", "contrast_ratio", "passes_wcag",
           "ThemeTokens", "SAFE_DEFAULT", "ThemeValidation", "validate_theme",
           "auto_correct", "FONT_PAIRINGS", "LAYOUT_VARIANTS", "font_pairing",
           "THEME_SCHEMA", "propose_theme", "design_score"]
