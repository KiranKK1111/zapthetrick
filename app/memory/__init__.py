"""Three-tier memory: working / episodic / semantic.

Architecture.md §4:

  WORKING MEMORY  — current conversation, blackboard, recent Q&As.
                    Cleared at session end.
  EPISODIC MEMORY — every Q&A with feedback, vector-indexed.
                    Used by [MemoryAgent] to recall similar interactions.
  SEMANTIC MEMORY — distilled skills, preferences, lessons.
                    Curated by the [ReflectorAgent].
"""
from .working import WorkingMemory
from .episodic import (
    Episode,
    EpisodicMemory,
    attach_feedback_db,
    record_episode,
    search_episodes_similar,
)
from .semantic import (
    SemanticMemory,
    Skill,
    delete_skill,
    list_skills_for_session,
    record_skill,
    relevant_skills_for_question,
)
from .skills_extractor import extract_skills

__all__ = [
    "WorkingMemory",
    "EpisodicMemory",
    "Episode",
    "record_episode",
    "attach_feedback_db",
    "search_episodes_similar",
    "SemanticMemory",
    "Skill",
    "record_skill",
    "list_skills_for_session",
    "relevant_skills_for_question",
    "delete_skill",
    "extract_skills",
]
