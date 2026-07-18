"""Query rewriter — multi-query expansion + HyDE.

Three transformations (Architecture.md §3):
  1. Resolve pronouns ("that project" → "Payments-V2")
  2. Expand to N paraphrases of the same intent
  3. HyDE: generate a hypothetical answer; embed *that* alongside the query

TODO: small LLM call. Returns the query unchanged until then.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewrittenQuery:
    original: str
    paraphrases: list[str]
    hyde: str


class QueryRewriter:
    def __init__(self, *, n_paraphrases: int = 3, use_hyde: bool = True) -> None:
        self.n_paraphrases = n_paraphrases
        self.use_hyde = use_hyde

    async def rewrite(self, query: str, *, context: str | None = None) -> RewrittenQuery:
        # TODO: LLM call.
        return RewrittenQuery(
            original=query,
            paraphrases=[query] * self.n_paraphrases,
            hyde=query if self.use_hyde else "",
        )
