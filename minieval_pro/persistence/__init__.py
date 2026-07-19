"""Audit persistence — append-only records of gate decisions."""

from .audit import AuditLog, DEFAULT_LOG_NAME

__all__ = ["AuditLog", "DEFAULT_LOG_NAME"]