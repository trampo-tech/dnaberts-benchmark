"""Centralised parameter resolution for Hydra/OmegaConf configs.

Every runner used to implement its own ad-hoc ``getattr`` /
``_resolve_train_override`` cascade, making it hard to reason about priority
and impossible to override consistently from the CLI.  This module provides a
single ``ParamResolver`` that:

* encapsulates the full priority chain as an **ordered list of path
  templates**;
* supports per-key overrides (e.g. ``token_max_length`` has a different
  fallback chain than ``train_bs``);
* resolves ``{task}`` placeholders from ``cfg.data.task`` so that per-task
  registries in the YAML files are consulted automatically;
* is fully compatible with Hydra CLI overrides – anything set via
  ``key=value`` on the command line is already reflected in the composed
  ``cfg``, so the resolver simply reads it at the right priority level.

Typical usage::

    from config.resolver import ParamResolver, TRANSFORMER_PRIORITY

    resolver = ParamResolver(cfg, priority=TRANSFORMER_PRIORITY)
    train_bs = resolver.resolve("train_bs", type_fn=int, default=cfg.train.train_bs)
    token_max_length = resolver.resolve("token_max_length", priority=TOKEN_MAX_LENGTH_PRIORITY, ...)
"""

from __future__ import annotations

from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Priority templates
# ---------------------------------------------------------------------------
# Each entry is a dot-path template.  ``{key}`` is replaced by the key being
# resolved; ``{task}`` is replaced by ``cfg.data.task``.
# ---------------------------------------------------------------------------

TRANSFORMER_PRIORITY: list[str] = [
    "data.{key}",
    "data.tasks.{task}.{key}",
    "train.{key}",
]

EVO2_PRIORITY: list[str] = [
    "model.tasks.{task}.{key}",
    "model.{key}",
    "data.{key}",
    "data.tasks.{task}.{key}",
    "train.{key}",
]

VARIANT_ZEROSHOT_PRIORITY: list[str] = [
    "data.{key}",
    "train.{key}",
]

KMER_LOGREG_PRIORITY: list[str] = [
    "data.{key}",
    "data.tasks.{task}.{key}",
    "train.{key}",
]

# ``token_max_length`` has a different cascade because it falls back to
# ``max_length`` (a different key name) at both the data and model levels.

TOKEN_MAX_LENGTH_PRIORITY: list[str] = [
    "model.token_max_length",
    "data.token_max_length",
    "data.max_length",
    "model.max_length",
]

EVO2_VARIANT_ZEROSHOT_PRIORITY: list[str] = [
    "model.{key}",
    "data.{key}",
    "train.{key}",
]

EVO2_TOKEN_MAX_LENGTH_PRIORITY: list[str] = [
    "model.tasks.{task}.token_max_length",
    "model.token_max_length",
    "data.token_max_length",
    "data.tasks.{task}.token_max_length",
    "data.max_length",
    "train.token_max_length",
    "model.max_length",
]

# ``eval_bs`` in the variant-effect zero-shot runner falls back from data to
# train, which is covered by VARIANT_ZEROSHOT_PRIORITY.  No special override
# needed.

# ---------------------------------------------------------------------------
# Sentinel for "value was not found in any source"
# ---------------------------------------------------------------------------

_MISSING = object()


class ParamResolver:
    """Resolve a config key through an ordered list of source paths.

    Parameters
    ----------
    cfg:
        The composed Hydra ``DictConfig``.
    priority:
        Ordered list of dot-path templates (highest priority first).
        ``{key}`` and ``{task}`` are expanded at resolution time.
    param_overrides:
        Optional mapping from key name to an alternative priority list.
        When resolving that key the override list is used instead of
        *priority*.
    """

    def __init__(
        self,
        cfg: DictConfig,
        priority: list[str],
        param_overrides: dict[str, list[str]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.priority = priority
        self.param_overrides = param_overrides or {}
        self._task: str | None = getattr(
            getattr(cfg, "data", None), "task", None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        key: str,
        type_fn: Callable | None = None,
        default: Any = _MISSING,
        *,
        priority: list[str] | None = None,
    ) -> Any:
        """Look up *key* through the priority chain and return the first
        non-``None`` value.

        Parameters
        ----------
        key:
            Config key to resolve (e.g. ``"train_bs"``).
        type_fn:
            Optional cast (e.g. ``int``, ``float``).  Applied only when a
            value was found – *default* is returned uncast.
        default:
            Fallback if *no* source contains the key.  ``_MISSING``
            (the sentinel) means raise ``KeyError`` instead of returning.
        priority:
            Override the instance-level priority list for this one call.
        """
        templates = priority or self.param_overrides.get(key, self.priority)
        for template in templates:
            path = template.format(key=key, task=self._task)
            value = self._lookup(path)
            if value is not None:
                return type_fn(value) if type_fn is not None else value

        if default is not _MISSING:
            return default

        raise KeyError(
            f"Config key {key!r} not found in any source.  "
            f"Priority chain: {templates}"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _lookup(self, dot_path: str) -> Any:
        """Traverse *dot_path* on ``self.cfg`` using ``OmegaConf.select``."""
        return OmegaConf.select(self.cfg, dot_path, default=None)


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------


def make_transformer_resolver(cfg: DictConfig) -> ParamResolver:
    return ParamResolver(
        cfg,
        priority=TRANSFORMER_PRIORITY,
        param_overrides={"token_max_length": TOKEN_MAX_LENGTH_PRIORITY},
    )


def make_evo2_resolver(cfg: DictConfig) -> ParamResolver:
    return ParamResolver(
        cfg,
        priority=EVO2_PRIORITY,
        param_overrides={
            "token_max_length": EVO2_TOKEN_MAX_LENGTH_PRIORITY,
        },
    )


def make_variant_zeroshot_resolver(cfg: DictConfig) -> ParamResolver:
    return ParamResolver(
        cfg,
        priority=VARIANT_ZEROSHOT_PRIORITY,
        param_overrides={
            "token_max_length": [
                "data.token_max_length",
                "data.max_length",
                "model.max_length",
            ],
        },
    )


def make_evo2_variant_zeroshot_resolver(cfg: DictConfig) -> ParamResolver:
    return ParamResolver(
        cfg,
        priority=EVO2_VARIANT_ZEROSHOT_PRIORITY,
        param_overrides={
            "token_max_length": [
                "model.token_max_length",
                "data.token_max_length",
                "data.max_length",
                "model.max_length",
            ],
        },
    )