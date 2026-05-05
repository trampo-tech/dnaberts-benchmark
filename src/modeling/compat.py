from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def block_triton_imports() -> None:
    """Clear stale Triton placeholders so remote-code loaders can fall back."""
    for key in ("triton", "triton.language", "triton.compiler", "triton.runtime"):
        if sys.modules.get(key) is None:
            sys.modules.pop(key, None)


def disable_remote_flash_attention(model: Any) -> None:
    """Force DNABERT-style remote modules onto their PyTorch attention path."""
    candidate_modules: list[str] = []

    for module_owner in (model, getattr(model, "backbone", None)):
        if module_owner is None:
            continue

        module_name = module_owner.__class__.__module__
        candidate_modules.append(module_name)
        if "." in module_name:
            package_prefix = module_name.rsplit(".", 1)[0]
            candidate_modules.extend(
                name for name in sys.modules if name.startswith(f"{package_prefix}.")
            )

    seen: set[str] = set()
    for module_name in candidate_modules:
        if module_name in seen:
            continue
        seen.add(module_name)

        module = sys.modules.get(module_name)
        if module is None or not hasattr(module, "flash_attn_qkvpacked_func"):
            continue

        setattr(module, "flash_attn_qkvpacked_func", None)


def fix_pad_token_id(config: Any, tokenizer: Any | None) -> Any:
    if config is None or tokenizer is None:
        return config

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return config

    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = pad_token_id
    return config


def resolve_hidden_size(config: Any) -> int:
    for attr in ("hidden_size", "d_model", "dim", "n_embd"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Unable to infer hidden size from backbone config.")


def _requires_legacy_remote_init_patch(model_name: str, trust_remote_code: bool) -> bool:
    if not trust_remote_code:
        return False
    return "dnabert-2" in model_name.lower()


@contextmanager
def legacy_remote_meta_init_disabled(
    model_name: str,
    trust_remote_code: bool,
) -> Iterator[None]:
    """Disable Transformers 5.x meta init for legacy DNABERT-2 remote code."""
    if not _requires_legacy_remote_init_patch(model_name, trust_remote_code):
        yield
        return

    try:
        from transformers import PreTrainedModel
        from transformers import __version__ as transformers_version
    except Exception:
        yield
        return

    version_prefix = transformers_version.split(".", 1)[0]
    if not version_prefix.isdigit() or int(version_prefix) < 5:
        yield
        return

    original = getattr(PreTrainedModel, "get_init_context", None)
    if original is None:
        yield
        return

    def _get_init_context(self, *args: Any, **kwargs: Any):
        return []

    setattr(PreTrainedModel, "get_init_context", _get_init_context)
    try:
        yield
    finally:
        setattr(PreTrainedModel, "get_init_context", original)