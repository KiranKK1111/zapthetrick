"""Technical-domain pipelines — Architecture.md §"Beyond DSA".

The DSA pipeline is one of many. The doc commits to 25+ specialised
sub-pipelines (system design, databases, devops, cloud, frontend,
backend, security, ML, distributed systems, …). Each follows the
same pattern as the DSA pipeline:

    extract → classify-pattern → generate-approach → verify → polish

But the verifier and pattern library differ per domain — a "system
design" answer's "verifier" is a checklist (load balancing, caching,
sharding, fault tolerance addressed?), not a Python subprocess.

This module is the dispatcher. The CoderAgent / Supervisor calls
`dispatch(question, intent)` and gets back an async iterator of
events shaped the same way as `app.dsa.solve()`.

The scaffold ships 5 domain stubs (system_design, databases,
devops, cloud, frontend) and a generic fallback. The remaining 20+
are TODO and currently route to the generic fallback.
"""
from .dispatcher import dispatch, DOMAINS


__all__ = ["dispatch", "DOMAINS"]
