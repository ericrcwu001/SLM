"""Typed pipeline errors, including the guard exceptions for blocked stages.

These mirror the eval harness's ``RequiresDecoder``/``RequiresModel`` pattern: a blocked
capability raises a specific, catchable error so the orchestrator records ``pending``
instead of fabricating a result.
"""

from __future__ import annotations


class PipelineError(RuntimeError):
    """Base class for data-pipeline errors."""


class AcquisitionError(PipelineError):
    """A source could not be acquired (network, auth, verification, extraction)."""


class StagingError(PipelineError):
    """Corpus staging failed (pack/stage/push): manifest, shard verify, or transfer error."""


class RequiresTokenizer(PipelineError):
    """Token materialization needs a frozen VQ tokenizer, which is not available."""


class RequiresTeacher(PipelineError):
    """Instruction generation needs a pinned teacher profile in configs/model_clients.yaml.

    Also raised at call time when the profile is pinned but the referenced credential env
    vars (endpoint_env / api_key_env) are not set in this environment — the profile is
    declared but not runnable here.
    """


class TeacherGenerationError(PipelineError):
    """The teacher API call ran but failed or returned an unparseable / invalid instruction."""


class RequiresManualOptIn(PipelineError):
    """A source needs a one-time manual step (credentials / concrete URLs) before it runs."""
