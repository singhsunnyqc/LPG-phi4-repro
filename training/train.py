"""
Training pipeline for SafetyLatentGuard with cached teacher summaries.
"""

from dataclasses import dataclass
from math import ceil
import json
import logging
import os
import re
import time
from typing import Dict, List, Sequence

import torch
import transformers
from peft import LoraConfig, TaskType
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Trainer, TrainerCallback

from src.model import (
    IGNORE_INDEX,
    DataArguments,
    ModelArguments,
    SafetyLatentGuard,
    TrainingArguments,
    build_teacher_reasoning_text,
    extract_json_verdict,
    extract_simple_verdict,
    extract_tagged_block,
    normalize_tagged_summary,
    parse_safety_stages,
)


def _to_scalar(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().float().mean().item()
    return float(x)


def read_jsonl(file_path: str) -> List[dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_lora_config(model_name: str, model_args: ModelArguments) -> LoraConfig:
    task_type = TaskType.CAUSAL_LM
    model_name_lower = model_name.lower()
    if any(name in model_name_lower for name in ["llama", "mistral", "falcon", "qwen"]):
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
    elif "phi" in model_name_lower:
        target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    elif any(name in model_name_lower for name in ["gpt2", "gsm-cot"]):
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        raise ValueError(
            f"Unsupported model: {model_args.model_name_or_path}. "
            "Add target_modules for this architecture."
        )

    return LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        init_lora_weights=True,
    )


def _tokenize_text(
    text: str,
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int = None,
) -> List[int]:
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=max_length is not None,
        max_length=max_length,
    )
    return list(tokens)


def _get_answer_token_position(
    tokens: torch.Tensor,
    answer_prompts: Sequence[torch.Tensor],
) -> int:
    for prompt in answer_prompts:
        if prompt.numel() == 0 or tokens.numel() < prompt.numel():
            continue
        try:
            match_index = (
                (tokens.unfold(0, prompt.numel(), 1) == prompt)
                .all(dim=1)
                .nonzero(as_tuple=True)[0]
                .item()
            )
            return match_index + prompt.numel()
        except Exception:
            continue
    return max(0, len(tokens) - 2)


def _find_subsequence_last_index(tokens: Sequence[int], pattern: Sequence[int]) -> int:
    if not pattern or len(tokens) < len(pattern):
        return -1
    for start_idx in range(len(tokens) - len(pattern) + 1):
        if list(tokens[start_idx : start_idx + len(pattern)]) == list(pattern):
            return start_idx + len(pattern) - 1
    return -1


def _build_stage_boundary_positions(
    question_ids: Sequence[int],
    teacher_ids: Sequence[int],
    stage_names: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> List[int]:
    positions = []
    question_offset = len(question_ids)
    for stage_name in stage_names:
        stage_tag = "".join(part.capitalize() for part in stage_name.split("_"))
        closing_tag_ids = _tokenize_text(f"</{stage_tag}>", tokenizer)
        boundary_idx = _find_subsequence_last_index(teacher_ids, closing_tag_ids)
        if boundary_idx < 0:
            # BPE may merge the final '>' with following whitespace (e.g. '>\n\n').
            # Fall back to matching the prefix '</Tag' and then the next token whose
            # decoded text starts with '>'.
            prefix_ids = _tokenize_text(f"</{stage_tag}", tokenizer)
            prefix_idx = _find_subsequence_last_index(teacher_ids, prefix_ids)
            if prefix_idx >= 0 and prefix_idx + 1 < len(teacher_ids):
                next_tok_text = tokenizer.decode([teacher_ids[prefix_idx + 1]])
                if next_tok_text.startswith(">"):
                    boundary_idx = prefix_idx + 1
        if boundary_idx < 0:
            raise ValueError(
                f"Could not locate closing tag </{stage_tag}> in teacher reasoning tokens."
            )
        positions.append(question_offset + boundary_idx)
    return positions


def _build_answer_text(example: dict, generated_reasoning: str) -> str:
    """Build a compact verdict string from the generated reasoning.

    Returns one of:
        ``safe``
        ``unsafe, policy N``             (single violation)
        ``unsafe, policy N1, N2, N3``    (multi-violation)
        ``unsafe``
    """
    verdict = extract_json_verdict(generated_reasoning)
    if verdict is None:
        body = extract_tagged_block(generated_reasoning, "Output")
        if body:
            verdict = extract_json_verdict(body)
            if verdict is None:
                verdict = extract_simple_verdict(body)

    if verdict is None:
        if example.get("safe", True):
            return "safe"
        # Prefer a multi-index JSON field, else the single-index field.
        match_many = re.search(
            r'"policy_indices"\s*:\s*\[([^\]]+)\]', generated_reasoning
        )
        if match_many:
            indices = [s.strip() for s in match_many.group(1).split(",") if s.strip().isdigit()]
            if indices:
                return f"unsafe, policy {', '.join(indices)}"
        match = re.search(r'"policy_index"\s*:\s*(\d+)', generated_reasoning)
        if match:
            return f"unsafe, policy {match.group(1)}"
        return "unsafe"

    if verdict.get("safe", True):
        return "safe"
    # Multi-index takes precedence; fall back to single-index field.
    indices = verdict.get("policy_indices")
    if indices:
        return f"unsafe, policy {', '.join(str(i) for i in indices)}"
    policy_idx = verdict.get("policy_index")
    if policy_idx is not None:
        return f"unsafe, policy {policy_idx}"
    return "unsafe"


def _load_teacher_summaries(
    example: dict,
    stage_names: List[str],
    require_teacher_summaries: bool,
) -> List[str]:
    summaries = example.get("teacher_summaries") or {}
    explicit_stages = parse_safety_stages(example.get("generated_reasoning", ""), stage_names)
    result = []
    for stage_name in stage_names:
        summary_key = f"{stage_name}_summary"
        raw_text = summaries.get(summary_key)
        if raw_text is None:
            if require_teacher_summaries:
                raise ValueError(
                    f"Missing `{summary_key}` for example with annotation_input prefix "
                    f"{example.get('annotation_input', '')[:80]!r}."
                )
            raw_text = explicit_stages.get(stage_name, "")
        result.append(normalize_tagged_summary(raw_text, stage_name))
    return result


class PerStepJSONLCallback(TrainerCallback):
    """Write one structured line per optimizer step to {output_dir}/per_step_log.jsonl.

    The CustomTrainer emits the five component losses via self.log() in compute_loss,
    while HF's Trainer emits a separate row carrying grad_norm/learning_rate/epoch for
    the same step. on_log caches the component losses, then flushes a merged line once
    it sees the grad_norm/learning_rate row — emitting at most one line per step.
    """

    _LOSS_KEYS = (
        "ce_loss",
        "distill_loss",
        "ref_ce_loss",
        "stage_align_loss",
        "explain_loss",
    )

    def __init__(self):
        self._cached_losses = {}
        self._last_logged_step = -1
        self._last_wall = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            if logs is None:
                return
            # Cache any component losses we see for the current step.
            for key in self._LOSS_KEYS:
                if key in logs:
                    self._cached_losses[key] = _to_scalar(logs[key])

            # The grad_norm / learning_rate row is the trigger to flush a merged line.
            if "grad_norm" not in logs and "learning_rate" not in logs:
                return

            step = state.global_step
            # Emit only once per optimizer step.
            if step == self._last_logged_step:
                return

            now = time.time()
            secs_since_last = None if self._last_wall is None else now - self._last_wall

            record = {
                "step": step,
                "epoch": logs.get("epoch", state.epoch),
                "ce_loss": self._cached_losses.get("ce_loss"),
                "distill_loss": self._cached_losses.get("distill_loss"),
                "ref_ce_loss": self._cached_losses.get("ref_ce_loss"),
                "stage_align_loss": self._cached_losses.get("stage_align_loss"),
                "explain_loss": self._cached_losses.get("explain_loss"),
                "lr": logs.get("learning_rate"),
                "grad_norm": _to_scalar(logs.get("grad_norm")),
                "timestamp": now,
                "secs_since_last": secs_since_last,
            }

            log_path = os.path.join(args.output_dir, "per_step_log.jsonl")
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            self._last_logged_step = step
            self._last_wall = now
            self._cached_losses = {}
        except Exception as e:  # never let logging crash training
            logging.warning(f"PerStepJSONLCallback.on_log failed: {e}")


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        step = self.state.global_step

        batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        num_epochs = self.args.num_train_epochs
        dataset_size = len(self.train_dataset)
        effective_batch_size = batch_size * self.args.world_size * gradient_accumulation_steps
        total_steps = ceil(dataset_size / effective_batch_size) * num_epochs

        inputs["step_ratio"] = step / max(1, total_steps)
        inputs["step"] = step

        outputs = model(**inputs)
        loss = outputs["loss"]

        if step % self.args.logging_steps == 0:
            logs = {
                "loss": _to_scalar(loss),
                "ce_loss": _to_scalar(outputs.get("ce_loss")),
                "distill_loss": _to_scalar(outputs.get("distill_loss")),
                "ref_ce_loss": _to_scalar(outputs.get("ref_ce_loss")),
                "explain_loss": _to_scalar(outputs.get("explain_loss")),
                "stage_align_loss": _to_scalar(outputs.get("stage_align_loss")),
            }
            self.log(logs)

        if return_outputs:
            return loss, outputs
        return loss


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    lora_config = build_lora_config(model_args.model_name_or_path, model_args)
    model = SafetyLatentGuard(model_args, training_args, lora_config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    stage_names = training_args.get_stage_names()

    def preprocess(
        questions: Sequence[str],
        teacher_reasonings: Sequence[str],
        answers: Sequence[str],
        stage_summaries: Sequence[Sequence[str]],
        tokenizer: transformers.PreTrainedTokenizer,
        bot_id: int,
        eot_id: int,
    ) -> Dict[str, List[torch.Tensor]]:
        eos_token_id = tokenizer.eos_token_id

        encoder_input_ids = []
        ref_input_ids = []
        ref_labels = []
        decoder_input_ids = []
        labels = []
        ref_answer_position = []
        model_answer_position = []
        stage_summary_ids = []
        stage_boundary_positions = []

        for question, teacher_reasoning, answer, summaries in zip(
            questions,
            teacher_reasonings,
            answers,
            stage_summaries,
        ):
            question_ids = _tokenize_text(question, tokenizer)
            teacher_ids = _tokenize_text(teacher_reasoning, tokenizer)
            answer_ids = _tokenize_text(answer, tokenizer)

            if not training_args.remove_eos and eos_token_id is not None:
                question_ids = question_ids + [eos_token_id]
                teacher_ids = teacher_ids + [eos_token_id]
            if eos_token_id is not None:
                answer_ids = answer_ids + [eos_token_id]

            summary_ids_for_example = []
            for summary_text in summaries:
                summary_ids = _tokenize_text(
                    summary_text,
                    tokenizer,
                    max_length=training_args.summary_max_target_length - 1,
                )
                if eos_token_id is not None:
                    summary_ids = summary_ids + [eos_token_id]
                summary_ids_for_example.append(torch.tensor(summary_ids, dtype=torch.long))

            question_tensor = torch.tensor(question_ids, dtype=torch.long)
            teacher_tensor = torch.tensor(teacher_ids, dtype=torch.long)
            answer_tensor = torch.tensor(answer_ids, dtype=torch.long)

            encoder_tensor = torch.tensor(question_ids + [bot_id], dtype=torch.long)
            decoder_prefix = [eot_id]
            if not training_args.remove_eos and eos_token_id is not None:
                decoder_prefix.append(eos_token_id)
            decoder_tensor = torch.tensor(decoder_prefix + answer_ids, dtype=torch.long)

            ref_tensor = torch.cat([question_tensor, teacher_tensor, answer_tensor])
            ref_label_tensor = ref_tensor.clone()
            ref_label_tensor[: len(question_tensor)] = IGNORE_INDEX
            boundary_positions = _build_stage_boundary_positions(
                question_ids=question_ids,
                teacher_ids=teacher_ids,
                stage_names=stage_names,
                tokenizer=tokenizer,
            )

            encoder_input_ids.append(encoder_tensor)
            ref_input_ids.append(ref_tensor)
            ref_labels.append(ref_label_tensor)
            decoder_input_ids.append(decoder_tensor)
            labels.append(decoder_tensor.clone())
            stage_summary_ids.append(summary_ids_for_example)
            stage_boundary_positions.append(torch.tensor(boundary_positions, dtype=torch.long))

            # Compute answer positions directly from known lengths
            # (avoids fragile token-pattern matching that can mis-fire on
            # common words like "safe"/"unsafe" appearing in teacher text).
            ref_answer_position.append(len(question_ids) + len(teacher_ids))
            model_answer_position.append(len(decoder_prefix))

        return {
            "encoder_input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "ref_input_ids": ref_input_ids,
            "labels": labels,
            "ref_answer_position": ref_answer_position,
            "model_answer_position": model_answer_position,
            "ref_labels": ref_labels,
            "stage_summary_ids": stage_summary_ids,
            "stage_boundary_positions": stage_boundary_positions,
        }

    class SafetyDataset(Dataset):
        def __init__(self, data_args, tokenizer, bot, eot):
            super().__init__()
            data_path = data_args.summary_cache_path or data_args.data_path
            if not data_path or not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"Training data not found. data_path={data_args.data_path}, "
                    f"summary_cache_path={data_args.summary_cache_path}"
                )

            logging.warning("Loading safety dataset from %s", data_path)
            raw_data = read_jsonl(data_path)
            if training_args.exp_mode:
                raw_data = raw_data[: training_args.exp_data_num]

            questions = []
            teacher_reasonings = []
            answers = []
            stage_summaries = []

            for example in tqdm(raw_data, desc="Processing examples"):
                annotation_input = example.get("annotation_input", "")
                generated_reasoning = example.get("generated_reasoning", "")
                if not annotation_input or not generated_reasoning:
                    continue

                token_count = len(
                    tokenizer.encode(
                        annotation_input + "\n" + generated_reasoning,
                        add_special_tokens=False,
                    )
                )
                if token_count > training_args.max_token_num:
                    continue

                summary_texts = _load_teacher_summaries(
                    example,
                    stage_names=stage_names,
                    require_teacher_summaries=training_args.require_teacher_summaries,
                )
                questions.append(annotation_input)
                teacher_reasonings.append(
                    build_teacher_reasoning_text(generated_reasoning, stage_names)
                )
                answers.append(_build_answer_text(example, generated_reasoning))
                stage_summaries.append(summary_texts)

            logging.warning("%s valid examples after filtering", len(questions))
            self.data_dict = preprocess(
                questions,
                teacher_reasonings,
                answers,
                stage_summaries,
                tokenizer,
                bot,
                eot,
            )
            self.keys = list(self.data_dict.keys())

        def __len__(self):
            return len(self.data_dict["encoder_input_ids"])

        def __getitem__(self, index):
            return {key: self.data_dict[key][index] for key in self.keys}

    @dataclass
    class SafetyDataCollator:
        tokenizer: transformers.PreTrainedTokenizer
        num_stages: int

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            encoder_input_ids = [instance["encoder_input_ids"] for instance in instances]
            decoder_input_ids = [instance["decoder_input_ids"] for instance in instances]
            ref_input_ids = [instance["ref_input_ids"] for instance in instances]
            labels = [instance["labels"] for instance in instances]
            ref_labels = [instance["ref_labels"] for instance in instances]
            ref_answer_position = [instance["ref_answer_position"] for instance in instances]
            model_answer_position = [instance["model_answer_position"] for instance in instances]
            stage_summary_ids = [instance["stage_summary_ids"] for instance in instances]
            stage_boundary_positions = [instance["stage_boundary_positions"] for instance in instances]

            reversed_encoder = [seq.flip(0) for seq in encoder_input_ids]
            encoder_input_ids = torch.nn.utils.rnn.pad_sequence(
                reversed_encoder,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ).flip(1)

            decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
                decoder_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=IGNORE_INDEX,
            )
            ref_input_ids = torch.nn.utils.rnn.pad_sequence(
                ref_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            ref_labels = torch.nn.utils.rnn.pad_sequence(
                ref_labels,
                batch_first=True,
                padding_value=IGNORE_INDEX,
            )

            max_summary_len = max(
                len(stage_tensor)
                for sample in stage_summary_ids
                for stage_tensor in sample
            )
            stage_summary_labels = torch.full(
                (len(instances), self.num_stages, max_summary_len),
                IGNORE_INDEX,
                dtype=torch.long,
            )
            for batch_idx, sample in enumerate(stage_summary_ids):
                for stage_idx, stage_tensor in enumerate(sample):
                    stage_summary_labels[batch_idx, stage_idx, : len(stage_tensor)] = stage_tensor

            return {
                "encoder_input_ids": encoder_input_ids,
                "decoder_input_ids": decoder_input_ids,
                "ref_input_ids": ref_input_ids,
                "labels": labels,
                "encoder_attention_mask": encoder_input_ids.ne(self.tokenizer.pad_token_id),
                "ref_answer_position": torch.tensor(ref_answer_position, dtype=torch.long),
                "model_answer_position": torch.tensor(model_answer_position, dtype=torch.long),
                "ref_attention_mask": ref_input_ids.ne(self.tokenizer.pad_token_id),
                "ref_labels": ref_labels,
                "stage_summary_labels": stage_summary_labels,
                "stage_boundary_positions": torch.stack(stage_boundary_positions),
            }

    def make_supervised_data_module(tokenizer, data_args) -> Dict:
        train_dataset = SafetyDataset(
            data_args=data_args,
            tokenizer=tokenizer,
            bot=model.bot_id,
            eot=model.eot_id,
        )
        data_collator = SafetyDataCollator(
            tokenizer=tokenizer,
            num_stages=len(stage_names),
        )
        return {
            "train_dataset": train_dataset,
            "eval_dataset": None,
            "data_collator": data_collator,
        }

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split("/")[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = CustomTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )
    trainer.add_callback(PerStepJSONLCallback())
    resume_ckpt = training_args.resume_from_checkpoint
    
    # Check if we need to load only model weights (not optimizer states)
    # This is needed when the checkpoint was saved with a different number of GPUs
    load_only_weights = getattr(training_args, "load_only_model_weights", False)
    if resume_ckpt and load_only_weights:
        print(f"Loading only model weights from checkpoint: {resume_ckpt}")
        print("WARNING: Optimizer states will be reinitialized. Training will start from step 0.")
        # Load model weights manually
        from safetensors.torch import load_file
        import glob
        
        # Try to load from model.safetensors or pytorch_model.bin
        model_files = glob.glob(os.path.join(resume_ckpt, "model.safetensors")) + \
                      glob.glob(os.path.join(resume_ckpt, "pytorch_model.bin"))
        if model_files:
            state_dict = load_file(model_files[0]) if model_files[0].endswith(".safetensors") else torch.load(model_files[0], map_location="cpu")
            # Filter out non-model keys (like "_module" prefix from DeepSpeed)
            filtered_state_dict = {}
            for k, v in state_dict.items():
                # Remove DeepSpeed module prefix if present
                if k.startswith("_module."):
                    k = k[8:]  # Remove "_module." prefix
                filtered_state_dict[k] = v
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Loaded model weights from {model_files[0]}")
        else:
            # Try loading adapter weights if using PEFT
            adapter_path = os.path.join(resume_ckpt, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                state_dict = load_file(adapter_path)
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded adapter weights from {adapter_path}")
            else:
                print(f"WARNING: No model weights found in {resume_ckpt}")
        
        # Train without resuming from checkpoint (since we loaded weights manually)
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
