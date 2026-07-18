"""
Tools package — importing this triggers each tool module to call
`registry.register(...)` at import time, so by the time any other code
asks `tool_registry.all_tools()` they're all there.
"""
from app.tools import code_solver as _code_solver  # noqa: F401
from app.tools import persona_answer as _persona_answer  # noqa: F401
from app.tools import resume_lookup as _resume_lookup  # noqa: F401
from app.tools import web_search as _web_search  # noqa: F401

# Code knowledge graph query tools (code_search / code_callers / code_callees /
# code_impact / code_file_structure) — registered on import.
try:
    from app.codegraph import tools as _codegraph_tools  # noqa: F401
except Exception:  # noqa: BLE001 — never block startup if codegraph deps absent
    pass
