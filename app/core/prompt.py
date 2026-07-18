"""Reliable prompt templating.

`str.format()` brace-parses the whole template, so a prompt that embeds a
literal JSON example (`{"x": 1}`) raises `KeyError` unless every brace is
escaped `{{ }}` — a fragile, easy-to-forget rule that silently broke several
classifier prompts.

`fill()` substitutes only the named `{placeholder}` tokens, in ONE pass, and
leaves every other brace untouched. So prompts can contain natural JSON, no
escaping is needed, and a forgotten escape can't break anything. Substitution is
single-pass, so a value that happens to contain `{another_key}` is never
re-substituted.
"""
from __future__ import annotations

import re


def fill(template: str, /, **values: object) -> str:
    """Replace each ``{name}`` in `template` with ``values[name]``; all other
    braces (e.g. a JSON example) pass through unchanged. Unknown placeholders are
    left as-is — never raises."""
    if not values:
        return template
    pattern = re.compile("|".join(re.escape("{" + k + "}") for k in values))
    return pattern.sub(lambda m: str(values[m.group()[1:-1]]), template)


__all__ = ["fill"]
