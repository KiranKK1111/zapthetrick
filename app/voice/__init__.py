"""Chat voice surface — the `/ws/voice` conversation stack.

This package owns everything the realtime-voice-mode design adds. The Live
interview module (`app/live/`, `/ws/live`) must never import from here: the
dependency direction is one-way and enforced by `tests/test_voice_isolation.py`.

Layout (design §L3):

    protocol.py   wire frame encode/decode — single source of truth
    engine.py     VoiceEngine / VoiceSession contract + registry
    staged.py     StagedEngine — the local cascade behind the contract
    realtime.py   RealtimeEngine — speech-native cloud session
    tools.py      agent-stack tool definitions + dispatch
    policy.py     engine selection, degradation
    budget.py     spend accounting + ceilings
    transcript.py turn ledger -> chat bubbles
    turn_gate.py  first-real-word gate (moved out of the Live file)
"""
