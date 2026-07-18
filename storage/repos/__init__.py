"""Repository layer — every route reads/writes through one of these.

Each repo wraps an [AsyncSession] and exposes intent-named methods
(`create_session`, `append_message`, `mark_resume_active`, …) rather
than raw ORM calls. Callers never see SQLAlchemy.

Imports go through [app.storage] — never [app.database]. The legacy
`database.py` module is a shim and is being retired.
"""
from .session_repo import SessionRepo
from .message_repo import MessageRepo
from .resume_repo import ResumeRepo
from .feedback_repo import FeedbackRepo
from .agent_run_repo import AgentRunRepo
from .agent_step_repo import AgentStepRepo
from .solve_repo import SolveRepo
from .usage_repo import UsageRepo

__all__ = [
    "SessionRepo",
    "MessageRepo",
    "ResumeRepo",
    "FeedbackRepo",
    "AgentRunRepo",
    "AgentStepRepo",
    "SolveRepo",
    "UsageRepo",
]
