import inspect
from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from modeling.compat import (
	fix_pad_token_id,
	legacy_remote_meta_init_disabled,
	resolve_hidden_size,
)


@dataclass
class ModelLoadInfo:
	used_fallback: bool
	fallback_reason: str | None = None


def _filter_forward_kwargs(module: nn.Module, kwargs: dict[str, Any]) -> dict[str, Any]:
	signature = inspect.signature(module.forward)
	accepts_var_kwargs = any(
		p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
	)
	if accepts_var_kwargs:
		return kwargs
	accepted = set(signature.parameters.keys())
	return {k: v for k, v in kwargs.items() if k in accepted}


def _resolve_hidden_size(config: Any) -> int:
	return resolve_hidden_size(config)


class BackboneSequenceClassifier(nn.Module):
	def __init__(self, backbone: nn.Module, num_labels: int):
		super().__init__()
		self.backbone = backbone
		self.num_labels = num_labels
		hidden_size = _resolve_hidden_size(backbone.config)
		dropout_prob = float(getattr(backbone.config, "hidden_dropout_prob", 0.1))
		self.dropout = nn.Dropout(dropout_prob)
		self.classifier = nn.Linear(hidden_size, num_labels)

	def _pool_sequence(
		self,
		last_hidden_state: torch.Tensor,
		attention_mask: torch.Tensor | None,
	) -> torch.Tensor:
		if attention_mask is None:
			return last_hidden_state[:, 0]

		mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
		denom = mask.sum(dim=1).clamp(min=1e-9)
		return (last_hidden_state * mask).sum(dim=1) / denom

	def forward(
		self,
		input_ids: torch.Tensor | None = None,
		attention_mask: torch.Tensor | None = None,
		token_type_ids: torch.Tensor | None = None,
		labels: torch.Tensor | None = None,
		**kwargs: Any,
	) -> SequenceClassifierOutput:
		model_inputs: dict[str, Any] = {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"token_type_ids": token_type_ids,
			**kwargs,
		}

		if (
			attention_mask is not None
			and "encoder_attention_mask"
			in inspect.signature(self.backbone.forward).parameters
			and "encoder_attention_mask" not in model_inputs
		):
			model_inputs["encoder_attention_mask"] = attention_mask

		model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
		model_inputs = _filter_forward_kwargs(self.backbone, model_inputs)

		outputs = self.backbone(**model_inputs)
		last_hidden_state = getattr(outputs, "last_hidden_state", None)
		if last_hidden_state is None and isinstance(outputs, (tuple, list)) and outputs:
			last_hidden_state = outputs[0]
		if last_hidden_state is None:
			raise ValueError("Backbone output does not contain last_hidden_state.")

		pooled_output = getattr(outputs, "pooler_output", None)
		if pooled_output is None:
			pooled_output = self._pool_sequence(last_hidden_state, attention_mask)

		logits = self.classifier(self.dropout(pooled_output))
		loss = None
		if labels is not None:
			loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

		return SequenceClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=getattr(outputs, "hidden_states", None),
			attentions=getattr(outputs, "attentions", None),
		)


def load_model_for_sequence_classification(
	model_name: str,
	num_labels: int,
	trust_remote_code: bool,
	tokenizer: Any | None = None,
	revision: str | None = None,
) -> tuple[nn.Module, ModelLoadInfo]:
	config = AutoConfig.from_pretrained(
		model_name,
		trust_remote_code=trust_remote_code,
		revision=revision,
	)
	config.num_labels = num_labels
	fix_pad_token_id(config, tokenizer)

	with legacy_remote_meta_init_disabled(model_name, trust_remote_code):
		try:
			model = AutoModelForSequenceClassification.from_pretrained(
				model_name,
				config=config,
				trust_remote_code=trust_remote_code,
				revision=revision,
			)
			return model, ModelLoadInfo(used_fallback=False)
		except Exception as exc:
			backbone = AutoModel.from_pretrained(
				model_name,
				config=config,
				trust_remote_code=trust_remote_code,
				revision=revision,
			)
			model = BackboneSequenceClassifier(backbone=backbone, num_labels=num_labels)
			return model, ModelLoadInfo(used_fallback=True, fallback_reason=str(exc))


def apply_lora(
	model: nn.Module,
	r: int,
	lora_alpha: int,
	lora_dropout: float,
	target_modules: list[str],
) -> nn.Module:
	config = LoraConfig(
		r=r,
		lora_alpha=lora_alpha,
		target_modules=target_modules,
		lora_dropout=lora_dropout,
		bias="none",
		task_type="SEQ_CLS",
		inference_mode=False,
	)
	model = get_peft_model(model, config)
	model.print_trainable_parameters()
	return model
