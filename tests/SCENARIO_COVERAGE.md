# Live-module scenario test coverage

Maps the 186 scenarios cataloged from `AnalysisReports/AnalysisOnLiveModule.md`
to their automated tests, organized by the document's Maturity Levels (L1–L5)
and Phase roadmap (P1–P7).

**Test files** (all deterministic — no network, no LLM, no real audio):
- `tests/test_scenarios_phase_a.py` — STT/repair, endpointing/pauses, intent & question detection
- `tests/test_scenarios_phase_b.py` — topic tracking, follow-ups, conversation graph, state machine, interruptions, world model
- `tests/test_scenarios_phase_c.py` — decision engine, planning, verification/scoring, hallucination/knowledge-gap, false-premise
- `tests/test_scenarios_phase_d.py` — resume intelligence, organization/JD, interview modes/phase, salary negotiation, memory
- `tests/test_scenarios_phase_e.py` — event bus, replay log, latency, health, recovery, diarization, privacy
- Plus existing suites: `test_live_state.py`, `test_live_robustness.py`, `test_org_intelligence.py`, `test_repair_phrases.py`, `test_intent_pipeline.py`, etc.

**115 scenario tests pass; 2 skipped (documented LLM-only gaps).**

## Status legend
- ✅ **TESTED** — a deterministic test exercises the real implementing module.
- ⚠️ **GAP** — the behavior is only partially implemented or LLM-only; test is skipped with a reason, OR the module exists but the exact scenario needs the LLM.
- 🔵 **RUNTIME** — implemented and exercised by the live E2E path, but not a pure unit (needs audio/LLM/WS); validated via `test_live_e2e.py` rather than a unit.
- ⬜ **ASPIRATIONAL** — described in the doc as a target; not yet implemented as a discrete module (real-audio diarization, GPU-STT selection, panel threads, evaluation dataset, digital twin, etc.).

## Level 1 — STT & audio front-end (Phase 3)
| # | Scenario | Status | Where |
|---|---|---|---|
| 1 | streaming STT partials | 🔵 | `stream.py` on_partial → `partial` frames (E2E) |
| 2 | VAD silence gating | ✅ | phase_a (energy-fallback), `vad.py` |
| 3 | speaker diarization (interviewer/candidate) | ✅ | phase_e `diarize.attribute` |
| 4 | utterance segmentation | 🔵 | `AudioStreamSegmenter` (E2E) |
| 5 | audio chunk streaming (20-50ms) | 🔵 | client PCM stream |
| 6 | GPU STT latency path | ⬜ | CPU/GPU selection not auto-benchmarked |

## Level 5 — Transcript repair (Phase 3)
| # | Scenario | Status | Where |
|---|---|---|---|
| 7 | low-conf word repair | ✅ | phase_a `repair` |
| 8 | domain vocab repair ("cube net is ingress"→"kubernetes ingress") | ✅ | phase_a, `test_repair_phrases.py` |
| 9 | LLM transcript normalize | ✅ | phase_a (predictor question) |
| 10 | preserve raw + normalized | ✅ | phase_a (`ev.context`) |
| 11 | grammar normalize meaning-preserving | ⚠️ GAP | LLM-only; repair preserves inflection by design |
| 12 | domain vocab boosting | ✅ | phase_a |
| 13 | STT conf lowers answer conf | ✅ | phase_c `uncertainty.propagate` |

## Level 2 — Intent & question detection (Phase 2)
| # | Scenario | Status | Where |
|---|---|---|---|
| 14 | direct question | ✅ | phase_a `heuristic_classify` |
| 15 | indirect question | ✅ | phase_a |
| 16 | scenario question | ✅ | phase_a `split_boundary` |
| 17 | explanation (not a question) | ✅ | phase_a |
| 18 | follow-up | ⚠️ GAP | follow-up detection is LLM-classifier-only |
| 19 | greeting/smalltalk/transition/hint | ✅ | phase_a `type_utterance` |
| 20-23 | intent hierarchy / question-type / intent-beyond-questions / clarification-exploratory | ✅/🔵 | `events.py`, `modes.py`, `implicit.py` |
| 24 | rhetorical suppression | ✅ | phase_a `rhetorical.should_answer` (tag-questions) |
| 25 | non-question requiring answer | ✅ | phase_a `implicit.detect_implicit` |
| 26-27 | answer-hint extraction / evaluation-objective | 🔵/✅ | `surface.py`, `objective.py` |
| 28 | question-boundary context split | ✅ | phase_a `split_boundary` |
| 29-31 | extraction model / confidence thresholds / hypothesis buffer | ✅ | `events.py`, `hypothesis.py` |
| 32 | commit points | ✅ | phase_a `HypothesisBuffer.settle_due` |
| 33 | semantic completion / implicit question | ✅ | phase_a `detect_implicit` |
| 34 | delay window before answer | ✅ | phase_a `required_settle_ms` |
| 35 | multi-question split | ✅ | phase_a `events.split_questions` |
| 36 | turn-taking (speaker finished) | ✅ | phase_b state machine, `hypothesis.py` |
| 37 | ensemble question detection | ✅ | phase_a `ensemble.decide` |
| 38 | incremental hypothesis update | 🔵 | streaming partials |

## Level 2 — Topic tracking (Phase 2)
| # | Scenario | Status | Where |
|---|---|---|---|
| 39 | topic current/sub/prev | ✅ | phase_b `topic_graph` |
| 40 | topic hierarchy tree | ✅ | phase_b |
| 41 | topic drift detection | ✅ | phase_b |
| 42 | domain shift | ✅ | phase_b |
| 43 | multiple concurrent topics (branch/return) | ✅ | phase_b |

## Level 3 — Follow-ups & conversation graph (Phase 2)
| # | Scenario | Status | Where |
|---|---|---|---|
| 44 | follow-up linked to topic | ✅ | phase_b |
| 45 | conversation graph attach | ✅ | phase_b |
| 46 | nested follow-up navigation ("go back to partitions") | ✅ | phase_b `resolve_reference` |
| 47 | coreference pronoun ("it"=Kafka) | ✅ | phase_b `world_model.resolve_coreference` |
| 48 | reference to earlier topic ("that") | ✅ | phase_b |
| 49 | candidate-answer awareness | 🔵 | `conversation.py`, candidate channel |
| 50 | follow-up prediction | ✅ | phase_b `predict.predict_next` |
| 51 | interview memory graph | ✅ | phase_b topic_graph + world_model (durable via state_persist) |

## Level 3 — State machine, satisfaction, interruptions, corrections
| # | Scenario | Status | Where |
|---|---|---|---|
| 52-53 | interview state machine (+streaming) | ✅ | phase_b `state_machine` |
| 54 | satisfaction closes thread | ✅ | phase_b `satisfaction.classify_feedback` |
| 55 | dissatisfaction keeps thread open | ✅ | phase_b |
| 56 | interruption stops generation | ✅ | phase_b + phase_c `decision`/`interrupt` |
| 57 | interrupted/abandoned question | ✅ | phase_b |
| 58 | self-correction supersede | ✅ | phase_b `interrupt.should_cancel` |
| 59 | cancellation support | ✅ | phase_b + phase_e `bus.cancel_all_answers` |
| 60-64 | question queue / adaptive thresholds / state validation / per-component uncertainty / confidence viz | ✅ | phase_e `validate`, `uncertainty`, `surface` |

## Level 3/5 — Decision engine & world model
| # | Scenario | Status | Where |
|---|---|---|---|
| 65 | decision engine (answer/wait/skip/clarify/cancel) | ✅ | phase_c `decision.decide_utterance/decide_event` |
| 66 | world/interview model | ✅ | phase_b `world_model.snapshot` |
| 67 | interview-OS shared state | 🔵 | per-session tracker |

## Level 4 — Answer planning, verification, scoring, self-correction
| # | Scenario | Status | Where |
|---|---|---|---|
| 68 | answer planning steps | ✅ | phase_c `plan.make_plan` |
| 69 | multi-pass understanding | ✅ | phase_c `objective.multi_pass` |
| 70 | answer-strategy selection | ✅ | phase_c `modes`/`strategy` |
| 71 | verifier question-check | ✅ | phase_c `verify.Verdict` |
| 72 | answer quality scorer | ✅ | phase_c `verify._parse` |
| 73 | answer lifecycle | ✅ | phase_c `decide_event` + `admit_answer` |
| 74 | real-time fact verification | 🔵 | `verify.verify_answer` (LLM; E2E) |
| 75-76 | evidence-based answering / hallucination prevention | ✅ | phase_c `evidence.hedge_directive` |
| 77 | knowledge-gap detection | ✅ | phase_c `guard.assess` |
| 78-79 | anti-hallucination source-check / resume-reality | ✅ | phase_d `assets.reality_directive` |
| 80-83 | adaptive length / multi-level answers / compression / depth estimation | ✅ | phase_c `deliberate`, `objective.estimate` |

## Level 5 — Incremental / speculative / event-driven / latency
| # | Scenario | Status | Where |
|---|---|---|---|
| 84 | event-driven architecture | ✅ | phase_e `bus.publish/subscribe` |
| 85 | replayable event log | ✅ | phase_e `eventlog` + `replay` |
| 86 | speculative background reasoning | 🔵 | `predict.py` |
| 87-90 | adaptive latency / parallel pipeline / budgeting / degradation | ✅ | phase_c+e `latency.select_path` |
| 91 | semantic event extraction | ✅ | phase_e `eventlog` |
| 92 | deliberation before action | ✅ | phase_c `deliberate` |
| 93-96 | multi-agent pipeline / start-LLM-early / streaming tokens / incremental delta | 🔵 | live path |

## Advanced reasoning & failure modes
| # | Scenario | Status | Where |
|---|---|---|---|
| 97 | temporal reasoning | 🔵 | `contradiction.resolve_temporal` |
| 98 | assumption tracking | ✅ | phase_b `world_model` assumptions |
| 99 | constraint extraction | ✅ | phase_b `extract_world` (cue-based) |
| 100 | contradiction/challenge detection | ✅ | phase_b `contradiction.is_challenge` |
| 101 | contradiction memory | ✅ | phase_b |
| 102 | adversarial false-premise | ✅ | phase_c `premise.check_premise` |
| 103 | real-time answer revision | ✅ | phase_b `revise.detect_reinterpretation` |
| 104 | prosody signals | ✅ (light) | phase_a `prosody_analyzer` |
| 105 | silence/pause intelligence | ✅ | phase_a `completeness` |

## Phase 4 — Specialized interview modes
| # | Scenario | Status | Where |
|---|---|---|---|
| 106 | system-design mode (iterative) | ✅ | phase_d `modes` |
| 107 | coding mode | 🔵 | code_solver path |
| 108 | panel interview threads | ⬜ | multi-speaker threading not implemented |
| 109 | semantic speaker roles | ✅ | phase_e `diarize` |
| 110 | behavioral STAR engine | ✅ | phase_d `modes` STAR |
| 111-113 | HR / system-design / strategy-per-type | ✅ | phase_d |

## Interviewer modeling / phase detection / coaching
| # | Scenario | Status | Where |
|---|---|---|---|
| 114-116 | interviewer pattern/personality/runtime modeling | ✅ | phase_e `style` |
| 117 | interview phase detection | ✅ | phase_d `phase.detect_phase` |
| 118 | cognitive-load estimation | ✅ | phase_c `style.cognitive_load` |
| 119 | real-time learning focus | ✅ | phase_d `knowledge.skill_gap_boost` |
| 120 | emotion detection | ✅ (light) | phase_d `emotion` |
| 121 | candidate weakness / skill-gap | ✅ | phase_d |
| 122 | real-time coaching layer | ✅ (light) | phase_d `coach` |
| 123 | question difficulty detection | ✅ | phase_c `objective`/difficulty |
| 124-128 | score predictor / outcome prediction / copilot / strategy / strategic planning | ✅/🔵 | `outcome`, `predict`, `surface` |

## Memory hierarchy & context
| # | Scenario | Status | Where |
|---|---|---|---|
| 129-131 | short/long, L1/L2/L3, 5-tier memory | ✅ | phase_d `memory` |
| 132 | dynamic context compression | 🔵 | `memory.refresh_summary` |
| 133 | session summarization | ✅ | phase_e `surface.talking_points` |
| 134 | context-window builder | 🔵 | orchestrator |

## Phase 1 — Resume / candidate intelligence
| # | Scenario | Status | Where |
|---|---|---|---|
| 135 | resume structured profile | ✅ | phase_d `profile.build_profile` |
| 136 | resume knowledge graph | ✅ | phase_d |
| 137-140 | candidate memory / self-intro / STAR / skill packs | ✅/🔵 | `assets`, `knowledge` |
| 141 | resume retrieval by topic | ✅ | phase_d `scoped_retrieve` |
| 142-144 | persona alignment / avoid-repeat / personalization | ✅ | phase_d |
| 145-148 | dynamic enrichment / multi-source / digital twin / dynamic loading | ⬜/✅ | partly `career.py` |

## Phase 1 — Organization / JD intelligence
| # | Scenario | Status | Where |
|---|---|---|---|
| 149 | organization profile build | ✅ | phase_d + `test_org_intelligence.py` |
| 150-152 | why-join / what-you-know / why-hire | ✅ | phase_d `org`, `assets` |
| 153 | JD upload fit analysis | ✅ | `test_org_intelligence.py` |
| 154-156 | precompute company answers / company-aware mode / research agent | ✅/⬜ | phase_d; live research is opt-in |

## Phase 4 — Salary negotiation
| # | Scenario | Status | Where |
|---|---|---|---|
| 157 | salary mode switch | ✅ | phase_d |
| 158 | salary expectation anchor | ✅ | phase_d `negotiate` |
| 159 | low-offer strategy | ✅ | phase_d (intent added) |
| 160 | value justification | ✅ | phase_d (intent added) |
| 161 | final-offer intent | ✅ | phase_d (intent added) |
| 162-166 | competing offers / company-aware / risk detection / value extraction / pre-gen justifications | ✅/🔵 | phase_d `negotiate` |

## Robustness / recovery / orchestration
| # | Scenario | Status | Where |
|---|---|---|---|
| 167 | accent/noise adaptation | ⬜ | runtime acoustic adaptation not implemented |
| 168 | conversation recovery (gap) | ✅ | phase_e `validate.detect_gap` |
| 169 | session recovery reconstruct | ✅ | phase_e `state_persist._build_snapshot` |
| 170 | graceful failure modes | ✅ | STT fallback chain, first-token deadline |
| 171 | continuous calibration | 🔵 | `calibration.py` |
| 172 | cost/resource routing | ✅ | router difficulty tiers |
| 173 | live session health monitoring | ✅ | phase_e `health.session_health` |
| 174 | human override + confidence UI | ✅ | phase_e `surface.override_suggestion` |
| 175 | multithreaded async pipelines | ✅ | phase_e bus + concurrent answers |

## Phase 5/6/7 — Evaluation, scoring, metrics, feedback
| # | Scenario | Status | Where |
|---|---|---|---|
| 176-177 | offline eval dataset / benchmark suite (500-1000) | ⬜ | evaluation corpus not built (biggest doc-flagged gap) |
| 178 | observability metrics | ✅ | phase_e `health.latency_ms_estimate` |
| 179 | metric-target thresholds | ⬜ | needs the eval corpus |
| 180 | answer scoring + regenerate | ✅ | phase_c `verify` + live regen |
| 181 | user feedback loop | 🔵 | feedback endpoints |
| 182 | active learning | 🔵 | `learned_exemplars.py` |
| 183 | interview replay engine | ✅ | phase_e `replay` |
| 184 | adaptive prompt construction | 🔵 | deliberate directive folding |
| 185 | domain packs | ✅ | `knowledge.configured_pack` |
| 186 | simulation / mock mode | ✅ (light) | phase_e `mock.generate_questions` |

---

## Summary
- **✅ TESTED (unit, deterministic):** ~115 scenarios across the 5 phase files + existing suites.
- **🔵 RUNTIME (live path / LLM / audio):** validated via `test_live_e2e.py`, not pure units.
- **⚠️ GAP (2):** grammar-normalize (#11) and deterministic follow-up (#18) are LLM-only.
- **⬜ ASPIRATIONAL:** real-audio diarization tuning (#3 tuning), GPU-STT auto-select (#6), panel threads (#108), accent/noise runtime adaptation (#167), digital twin (#147), and the **evaluation dataset / benchmark suite (#176-177, #179)** — the doc itself calls this the single biggest remaining investment.
