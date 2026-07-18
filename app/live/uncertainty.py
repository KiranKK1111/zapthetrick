"""
Uncertainty propagation (live-conversational-intelligence R14).

Low upstream confidence (STT / topic / speaker attribution) drags the surfaced
answer confidence down so a poorly-heard or ambiguous turn is not presented as
high-confidence. Deterministic + fail-open.
"""
from __future__ import annotations


def propagate(
    answer_conf: float,
    *,
    stt_conf: float | None = None,
    topic_conf: float | None = None,
    speaker_conf: float | None = None,
) -> float:
    """Reduce `answer_conf` by any low upstream confidences. Each provided
    signal caps the result at (0.5 + 0.5*signal), so a very low signal pulls the
    final confidence toward 0.5*signal. Returns a value in [0,1]."""
    try:
        conf = max(0.0, min(1.0, float(answer_conf)))
        for c in (stt_conf, topic_conf, speaker_conf):
            if c is None:
                continue
            c = max(0.0, min(1.0, float(c)))
            conf = min(conf, 0.5 + 0.5 * c)
        return max(0.0, min(1.0, conf))
    except Exception:  # noqa: BLE001
        return max(0.0, min(1.0, answer_conf if isinstance(answer_conf, (int, float)) else 0.5))
