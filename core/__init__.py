"""Core package for shared components across the YouTube Content Suite.

This package provides stable import paths for cross-cutting concerns such as
configuration, LLM utilities, shared data models, and translators. It is
introduced to reduce coupling between individual pipelines and to keep their
directories focused on domain-specific logic.

Phase 1: this module primarily re-exports existing implementations from
yt_transcriber and yt_script_generator to avoid breaking changes. Downstream
code can begin importing from `core` while legacy imports continue to work.
"""

# Re-export convenience imports for users of `core`
from core.settings import AppSettings, settings  # noqa: F401
