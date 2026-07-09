"""SLM data-generation pipeline (master-plan Stages 2-9 + Stage 11 spine).

Acquisition-first: real, autonomous source acquisition (Stage 2) feeds a runnable spine
(registry -> canonicalize -> filter -> split/leakage -> select -> active/eval/warmup
manifests). Token materialization (frozen VQ tokenizer) and instruction generation
(teacher model) are typed, guarded interfaces that refuse rather than fabricate.

Depends one-way on the :mod:`eval` package (``cube_io``, ``vocab``, ``schemas``,
``color_pipeline``).
"""

__version__ = "0.1.0"
