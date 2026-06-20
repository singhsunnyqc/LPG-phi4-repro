"""
Latent Policy Guard model implementation for the evaluation framework.

This module wraps the SafetyLatentGuard model (defined in `training/src/model.py`)
as a BaseModel plugin so it can be evaluated through the standard evaluation
pipeline on any supported dataset.

The model loads a trained checkpoint and runs the custom latent inference loop:
  1. Encode the input (content + policies formatted as annotation_input)
  2. Run latent token generation for each stage (intent, risk)
  3. Auto-regressively decode the explicit output verdict

Two checkpoint formats are supported:
  - Full state dict (--ckpt_dir pointing to model.safetensors / pytorch_model.bin)
  - PEFT adapter directory (--adapter_path pointing to adapter_config.json +
    adapter_model.safetensors), optionally combined with --ckpt_dir for extra
    weights like the projection module.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from peft import LoraConfig, PeftModel, TaskType
from safetensors.torch import load_file

from .base import BaseModel, ModelInfo, ModelOutput, register_model

# ---------------------------------------------------------------------------
# Import SafetyLatentGuard from the sibling `training/` package.
# Repository layout (relative to this file):
#   <repo-root>/training/src/model.py
#   <repo-root>/evaluation/models/latent_policy_guard.py   <-- this file
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_DIR = _REPO_ROOT / "training"
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from src.model import (  # noqa: E402  (sys.path tweak above)
    SafetyLatentGuard,
    ModelArguments,
    TrainingArguments,
    extract_simple_verdict,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_lora_config(model_name: str, model_args: ModelArguments) -> LoraConfig:
    """Build LoRA config matching the training setup."""
    model_name_lower = model_name.lower()
    if any(n in model_name_lower for n in ["llama", "mistral", "falcon", "qwen"]):
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ]
    elif "phi" in model_name_lower:
        target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    elif "gpt2" in model_name_lower:
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        init_lora_weights=True,
    )


def _parse_bool(value) -> bool:
    """Robustly convert a value to bool, handling strings like 'False'."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def _format_annotation_input(
    content: str,
    policy: Optional[str] = None,
    policies: Optional[list] = None,
    response: Optional[str] = None,
    dataset_type: str = "multi_policy",
) -> str:
    """
    Format dataset sample fields into the annotation_input format that
    SafetyLatentGuard was trained on.

    The training data uses policies-first ordering:
        **Safety policies (indexed from 0):**
        0: <policy_0>
        1: <policy_1>
        ...**Content to evaluate:**
        <transcript>
    """
    content_to_eval = content
    if response:
        content_to_eval = f"{content}\n\nModel response: {response}"

    if dataset_type == "single_policy":
        if policy:
            return (
                f"**Safety policies (indexed from 0):**\n0: {policy}"
                f"**Content to evaluate:**\n{content_to_eval}"
            )
        return f"**Content to evaluate:**\n{content_to_eval}"
    else:
        if isinstance(policies, str):
            # DynaBench sends the policy block as a raw string —
            # rewrite ``N. text`` headers to ``N: text`` to match
            # the training data format.
            policies_block = _normalize_policy_headers(policies)
        elif policies:
            policies_block = "\n".join(
                f"{idx}: {p}" for idx, p in enumerate(policies)
            )
        else:
            policies_block = ""

        if policies_block:
            return (
                f"**Safety policies (indexed from 0):**\n{policies_block}"
                f"**Content to evaluate:**\n{content_to_eval}"
            )
        return f"**Content to evaluate:**\n{content_to_eval}"


def _normalize_policy_headers(policies: str) -> str:
    """Rewrite ``N. text`` to ``N: text`` for top-level policy headers.

    This matches the training data format where policies are indexed
    with ``N:`` prefix.
    """
    lines = policies.splitlines()
    expected = 0
    for idx, line in enumerate(lines):
        m = re.match(r"^(\d+)\.\s+(.*)$", line)
        if m and int(m.group(1)) == expected:
            lines[idx] = f"{expected}: {m.group(2)}"
            expected += 1
    return "\n".join(lines)


@register_model
class LatentPolicyGuardModel(BaseModel):
    """
    Latent Policy Guard model using SafetyLatentGuard with custom latent
    inference loop.

    This model loads a trained checkpoint (base model + LoRA + projection)
    and runs multi-stage latent reasoning followed by explicit verdict
    generation.

    Supports two loading modes:

    1. **Full state dict** (default): pass ``ckpt_dir`` pointing to a
       directory that contains ``model.safetensors`` or ``pytorch_model.bin``.
       The state dict is loaded over a freshly-initialized
       SafetyLatentGuard (which includes random LoRA weights that get
       overwritten).

    2. **PEFT adapter**: pass ``adapter_path`` pointing to a standard PEFT
       adapter directory (``adapter_config.json`` +
       ``adapter_model.safetensors``).  The base model is created *without*
       LoRA, then the adapter is loaded via ``PeftModel.from_pretrained``.
       If extra weights exist (projection module, special token embeddings),
       pass ``ckpt_dir`` as well — non-LoRA weights will be loaded from it.

    Required kwargs at construction time:
        ckpt_dir:               Path to checkpoint directory (full state dict
                                or extra weights when using adapter_path)
        adapter_path:           Path to PEFT adapter directory (optional)
        lora_r:                 LoRA rank (default 128)
        lora_alpha:             LoRA alpha (default 32)
        num_latent_per_stage:   e.g. "4,6" (default "4,6")
        stage_names:            e.g. "intent,risk" (default "intent,risk")
        use_prj:                Use projection (default True)
        prj_dim:                Projection dim (default 2560)
        remove_eos:             Remove EOS in prefix (default True)
        greedy:                 Greedy decoding (default True)
        model_max_length:       Max input length (default 1024)
    """

    @classmethod
    def get_info(cls) -> ModelInfo:
        return ModelInfo(
            name="latent_policy_guard",
            description="Latent Policy Guard (SafetyLatentGuard) with latent reasoning",
            model_type="huggingface",
            supports_system_prompt=False,
            max_context_length=1024,
        )

    def load(self) -> None:
        ckpt_dir = self.config.get("ckpt_dir")
        adapter_path = self.config.get("adapter_path")

        if not ckpt_dir and not adapter_path:
            raise ValueError(
                "Either --ckpt_dir or --adapter_path (or both) is required "
                "for latent_policy_guard."
            )

        use_adapter = adapter_path is not None

        # Build model args
        model_args = ModelArguments(
            model_name_or_path=self.model_path,
            lora_r=int(self.config.get("lora_r", 128)),
            lora_alpha=int(self.config.get("lora_alpha", 32)),
            lora_dropout=float(self.config.get("lora_dropout", 0.05)),
            full_precision=True,
            use_decoder=False,
            train=False,
            lora_init=True,
            ckpt_dir=ckpt_dir,
        )

        # Build training args
        # Use bf16 only if CUDA is actually available (avoids validation
        # errors on nodes with broken drivers). Model dtype is set later.
        training_args = TrainingArguments(
            output_dir="./outputs",
            bf16=torch.cuda.is_available(),
            model_max_length=int(self.config.get("model_max_length", 1024)),
            num_latent_per_stage=self.config.get("num_latent_per_stage", "4,6"),
            stage_names=self.config.get("stage_names", "intent,risk"),
            # When loading via adapter_path, skip LoRA in __init__ —
            # we'll attach the adapter ourselves afterwards.
            use_lora=not use_adapter,
            use_prj=_parse_bool(self.config.get("use_prj", True)),
            prj_dim=int(self.config.get("prj_dim", 3072)),
            prj_dropout=float(self.config.get("prj_dropout", 0.0)),
            prj_no_ln=False,
            greedy=_parse_bool(self.config.get("greedy", True)),
            remove_eos=_parse_bool(self.config.get("remove_eos", True)),
            print_loss=False,
        )

        lora_config = _build_lora_config(model_args.model_name_or_path, model_args)
        model = SafetyLatentGuard(model_args, training_args, lora_config)

        if use_adapter:
            # ----- Mode 2: PEFT adapter loading -----
            # model.codi is the bare base model (no LoRA applied in __init__
            # because we set use_lora=False above).  Attach the saved adapter.
            model.codi = PeftModel.from_pretrained(
                model.codi,
                adapter_path,
                is_trainable=False,
            )
            # If ckpt_dir is also provided, load non-LoRA extra weights
            # (projection module, resized embeddings, etc.)
            if ckpt_dir:
                extra_path = os.path.join(ckpt_dir, "extra_weights.safetensors")
                if os.path.exists(extra_path):
                    extra_state = load_file(extra_path)
                    model.load_state_dict(extra_state, strict=False)
                else:
                    # Fall back: try loading full state dict but only the
                    # non-LoRA keys will actually take effect (LoRA weights
                    # are already loaded from the adapter).
                    full_path = os.path.join(ckpt_dir, "model.safetensors")
                    if not os.path.exists(full_path):
                        full_path = os.path.join(ckpt_dir, "pytorch_model.bin")
                    if os.path.exists(full_path):
                        if full_path.endswith(".safetensors"):
                            state_dict = load_file(full_path)
                        else:
                            state_dict = torch.load(full_path, map_location="cpu")
                        # Filter out LoRA keys — those are already loaded
                        # from the adapter.
                        non_lora = {
                            k: v for k, v in state_dict.items()
                            if "lora_" not in k
                        }
                        model.load_state_dict(non_lora, strict=False)
        else:
            # ----- Mode 1: Full state dict loading -----
            state_dict_path = os.path.join(ckpt_dir, "model.safetensors")
            if os.path.exists(state_dict_path):
                state_dict = load_file(state_dict_path)
            else:
                state_dict = torch.load(
                    os.path.join(ckpt_dir, "pytorch_model.bin"),
                    map_location="cpu",
                )
            result = model.load_state_dict(state_dict, strict=False)
            if result.unexpected_keys:
                # Filter out training-only keys (decoders, explain projectors)
                real_unexpected = [
                    k for k in result.unexpected_keys
                    if not any(s in k for s in ("summary_decoder", "explain_projector"))
                ]
                if real_unexpected:
                    print(f"WARNING: {len(real_unexpected)} unexpected keys in checkpoint: "
                          f"{real_unexpected[:5]}...")
            if result.missing_keys:
                print(f"WARNING: {len(result.missing_keys)} missing keys not in checkpoint: "
                      f"{result.missing_keys[:5]}...")

        model = model.to(DEVICE)
        # Always convert to bf16 since checkpoints are trained in bf16;
        # avoids dtype mismatches when loading on CPU or mixed-GPU nodes.
        model = model.to(torch.bfloat16)
        model.eval()

        self._model = model
        self._model_args = model_args
        self._training_args = training_args
        self._num_latent_per_stage = training_args.get_num_latent_list()

        # Build tokenizer
        import transformers
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            use_fast=False,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self._tokenizer.pad_token_id = model.pad_token_id
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.convert_tokens_to_ids("[PAD]")

    def generate(
        self,
        system_prompt: str,
        user_input: Optional[str] = None,
        content: Optional[str] = None,
        policy: Optional[str] = None,
        policies: Optional[list] = None,
        response: Optional[str] = None,
        dataset_type: Optional[str] = None,
        max_new_tokens: int = 160,
        temperature: float = 0.1,
        think: Optional[str] = None,
        **kwargs,
    ) -> ModelOutput:
        model = self._model
        tokenizer = self._tokenizer
        training_args = self._training_args

        # Format input as annotation_input (LPG ignores system_prompt —
        # it was trained without one, the policy is embedded in the input)
        if user_input is None:
            user_input = _format_annotation_input(
                content=content,
                policy=policy,
                policies=policies,
                response=response,
                dataset_type=dataset_type or "multi_policy",
            )

        # Tokenize
        batch = tokenizer(
            [user_input],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=training_args.model_max_length,
        )

        # Append <bot> token
        bot_prefix = [model.bot_id]
        if not training_args.remove_eos and tokenizer.eos_token_id is not None:
            bot_prefix = [tokenizer.eos_token_id, model.bot_id]
        bot_tensor = torch.tensor(bot_prefix, dtype=torch.long).unsqueeze(0)

        input_ids = torch.cat((batch["input_ids"], bot_tensor), dim=1).to(DEVICE)
        attention_mask = torch.cat(
            (batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1
        ).to(DEVICE)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            # Encode input
            outputs = model.codi(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=True,
                attention_mask=attention_mask,
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1:, :]

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # Latent reasoning stages
            for num_latent in self._num_latent_per_stage:
                for _ in range(num_latent):
                    outputs = model.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values,
                    )
                    past_key_values = outputs.past_key_values
                    latent_embd = outputs.hidden_states[-1][:, -1:, :]
                    if training_args.use_prj:
                        latent_embd = model.prj(latent_embd)

            # Begin explicit output decoding
            eot_prefix = [model.eot_id]
            if not training_args.remove_eos and tokenizer.eos_token_id is not None:
                eot_prefix.append(tokenizer.eos_token_id)
            eot_ids = torch.tensor(eot_prefix, dtype=torch.long, device=DEVICE)

            output_embeds = (
                model.get_embd(model.codi, model.model_name)(eot_ids)
                .unsqueeze(0)
            )

            pred_tokens = []

            for _ in range(max_new_tokens):
                out = model.codi(
                    inputs_embeds=output_embeds,
                    output_hidden_states=False,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :].clone()

                # Suppress special tokens
                for token_id in [model.pad_token_id, model.bot_id, model.eot_id]:
                    if 0 <= token_id < logits.size(-1):
                        logits[:, token_id] = float("-inf")

                if training_args.greedy:
                    next_token_id = torch.argmax(logits, dim=-1)
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

                token_val = next_token_id.item()
                if token_val == tokenizer.eos_token_id:
                    break
                pred_tokens.append(token_val)

                output_embeds = (
                    model.get_embd(model.codi, model.model_name)(next_token_id)
                    .unsqueeze(1)
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_time

        # Decode and parse — supports both compact ("safe" / "unsafe, policy N")
        # and legacy JSON ({"safe": true/false, ...}) formats.
        decoded = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        verdict = extract_simple_verdict(decoded)

        if verdict is None:
            verdict = {"safe": True}

        return ModelOutput(
            prediction=verdict,
            raw_output=decoded,
            inference_time=inference_time,
            tokens_generated=len(pred_tokens),
        )
