"""Multi-interpretation disambiguation (advanced-intent-reasoning R5).

When a request reads multiple ways, the gate may return a set of candidate
interpretations with probabilities (from the SAME single LLM call — never a
second blocking one). If one dominates, answer it; otherwise ask one precise
disambiguation question whose options are the candidate readings.
"""
from __future__ import annotations

_DEFAULT_DOMINANCE = 0.6
_MAX_INTERP = 4


def parse_interpretations(raw) -> list[dict]:
    """Sanitize the model's `interpretations` into [{reading, probability}].
    Tolerates missing/garbled probabilities (defaults to uniform). Returns []
    for anything unusable (R5.4)."""
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw[:_MAX_INTERP]:
        if not isinstance(item, dict):
            continue
        reading = str(item.get("reading") or item.get("label") or "").strip()
        if not reading:
            continue
        try:
            p = float(item.get("probability"))
        except (TypeError, ValueError):
            p = 0.0
        out.append({"reading": reading[:120], "probability": max(0.0, p)})
    if len(out) < 2:
        return []  # not actually ambiguous → caller uses the normal path
    # Normalise probabilities to sum ~1.0 (uniform if all zero).
    total = sum(i["probability"] for i in out)
    if total <= 0:
        for i in out:
            i["probability"] = 1.0 / len(out)
    else:
        for i in out:
            i["probability"] = i["probability"] / total
    return out


def pick_interpretation(interps: list[dict],
                        dominance: float = _DEFAULT_DOMINANCE) -> tuple[str, object]:
    """Decide between answering the dominant reading and asking (R5.2/5.3).

    Returns:
      ("answer", reading_str)  when exactly one probability > `dominance`;
      ("ask", options_list)    otherwise — one option per interpretation.
    Empty/single interpretation → ("none", None) (caller uses the normal path).
    """
    clean = parse_interpretations(interps)
    if not clean:
        return ("none", None)
    top = max(clean, key=lambda i: i["probability"])
    if top["probability"] > dominance:
        return ("answer", top["reading"])
    options = [
        {
            "id": f"interp{n + 1}",
            "label": i["reading"][:60],
            "description": "",
            "recommended": (i is top),
        }
        for n, i in enumerate(clean)
    ]
    # At most one recommended.
    seen = False
    for o in options:
        if o["recommended"] and not seen:
            seen = True
        else:
            o["recommended"] = False
    return ("ask", options)


__all__ = ["parse_interpretations", "pick_interpretation"]
