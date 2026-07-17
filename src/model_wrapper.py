"""Model wrapper - provided by the shared dllm package.

Kept as a thin re-export so existing imports and saved checkpoints
(config.json + `model.` prefixed weights) keep working unchanged.
"""

from dllm.models.hf import (  # noqa: F401
    DiffusionConfig,
    DiffusionLMOutput,
    DiffusionTransformerLM,
)

LLaDAOutput = DiffusionLMOutput  # old name
