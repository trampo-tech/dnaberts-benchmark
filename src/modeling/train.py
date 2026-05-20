from typing import Any

from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification

from modeling.compat import (
	fix_pad_token_id,
	legacy_remote_meta_init_disabled,
)


def load_model_for_sequence_classification(
	model_name: str,
	num_labels: int,
	trust_remote_code: bool,
	tokenizer: Any | None = None,
	revision: str | None = None,
) -> nn.Module:
	config = AutoConfig.from_pretrained(
		model_name,
		trust_remote_code=trust_remote_code,
		revision=revision,
	)
	config.num_labels = num_labels
	fix_pad_token_id(config, tokenizer)

	with legacy_remote_meta_init_disabled(model_name, trust_remote_code):
		return AutoModelForSequenceClassification.from_pretrained(
			model_name,
			config=config,
			trust_remote_code=trust_remote_code,
			revision=revision,
		)


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
