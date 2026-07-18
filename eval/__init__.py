"""Evaluation harness — Architecture.md §"Eval harness".

A repeatable, graded test corpus that catches regressions in:
  - question detection
  - intent classification
  - retrieval
  - persona answer quality
  - DSA pipeline correctness

The harness is a CLI runner (`python -m eval.runner`) plus a small
library other tests can call. Each case is a single YAML file under
`eval/datasets/` carrying:

    case_id: 0001
    category: dsa | behavioral | system_design | concept | meta
    input: { question, context_snippets? }
    expected:
        intent: coding
        contains: ["binary search", "O(log n)"]
        omits:   ["O(n^2)"]
        rubric: 0.7   # LLM-as-judge grade threshold

The full architecture commits to 200+ cases; this scaffold ships
with a seed of ~10 covering the most-likely-to-regress paths.
Adding cases is just dropping more YAML files in.
"""
