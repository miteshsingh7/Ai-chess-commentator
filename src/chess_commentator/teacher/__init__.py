"""Teacher LLM commentary generation module."""

from chess_commentator.teacher.cache import TeacherCache
from chess_commentator.teacher.prompt_builder import (
    build_teacher_prompt,
    TEACHER_SYSTEM_PROMPT,
)
from chess_commentator.teacher.client import ClaudeTeacherClient
from chess_commentator.teacher.generator import TeacherCommentaryGenerator

__all__ = [
    "TeacherCache",
    "build_teacher_prompt",
    "TEACHER_SYSTEM_PROMPT",
    "ClaudeTeacherClient",
    "TeacherCommentaryGenerator",
]
