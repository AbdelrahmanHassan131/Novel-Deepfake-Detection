"""
Evaluation Reports subpackage.

Generates structured evaluation reports in Markdown and JSON formats.

Public API::

    from evaluation.reports import EvaluationReportGenerator
"""

from .generator import EvaluationReportGenerator

__all__ = [
    'EvaluationReportGenerator',
]
