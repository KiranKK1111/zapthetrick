"""End-to-end project verification loop (plan → build → verify → test)."""
from .project_verify import ProjectVerification, files_from_zip, verify_project_files

__all__ = ["ProjectVerification", "verify_project_files", "files_from_zip"]
