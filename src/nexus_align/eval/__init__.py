"""Evaluation: checkpoint sampling and FID/IS metrics."""

from nexus_align.registry import registry
from nexus_align.eval.evaluator import FidEvaluator

registry.register("evaluator", "fid", FidEvaluator)
