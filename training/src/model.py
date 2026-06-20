"""
SafetyLatentGuard — latent reasoning for safety policy compliance.

This version follows the planned Sim-CoT style workflow:
  1. Stage 1 and Stage 2 are compressed into latent tokens.
  2. Training-only lightweight decoders reconstruct cached summary targets.
  3. The final output verdict stays explicit.
"""

from dataclasses import dataclass, field
import json
import math
import re
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import transformers
from peft import get_peft_model
from safetensors.torch import load_file
from torch.cuda.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer


IGNORE_INDEX = -100

# Workaround: transformers v5 leaks `resume_download` to model __init__
try:
    from transformers import Qwen3ForCausalLM as _Qwen3
    _orig_qwen3_init = _Qwen3.__init__
    def _patched_qwen3_init(self, config, *args, resume_download=None, **kwargs):
        _orig_qwen3_init(self, config, *args, **kwargs)
    _Qwen3.__init__ = _patched_qwen3_init
except (ImportError, AttributeError):
    pass


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-4B")
    lora_r: int = field(default=128, metadata={"help": "LoRA rank."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    full_precision: bool = field(
        default=True,
        metadata={"help": "Whether to use full precision for the base model."},
    )
    use_decoder: bool = field(
        default=True,
        metadata={"help": "Use training-only summary decoders."},
    )
    save_ablation: bool = field(default=False)
    train: bool = field(
        default=True,
        metadata={"help": "If true, initialize for training; else for inference."},
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: initialize new LoRA adapters."},
    )
    token: Optional[str] = field(default=None, metadata={"help": "HF token."})
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a LoRA adapter."},
    )
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha scaling."})
    ckpt_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint dir for inference."},
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the JSONL data."})
    summary_cache_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional augmented JSONL with teacher summaries."},
    )
    debug_data: bool = field(default=False, metadata={"help": "Enable debug dataset."})
    batch_size: int = field(default=1, metadata={"help": "Batch size during inference."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={"help": "Maximum sequence length."},
    )
    restore_from: str = field(default="", metadata={"help": "Checkpoint to restore from."})
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    expt_name: str = field(default="default", metadata={"help": "Experiment name."})

    num_latent_per_stage: str = field(
        default="2,4",
        metadata={
            "help": (
                "Comma-separated list of latent counts per reasoning stage. "
                "For this project Stage 1=Intent and Stage 2=Risk."
            )
        },
    )
    stage_names: str = field(
        default="intent,risk",
        metadata={"help": "Comma-separated names for latent stages."},
    )

    use_lora: bool = field(default=True, metadata={"help": "Use LoRA."})
    greedy: bool = field(default=False, metadata={"help": "Greedy decoding at inference."})
    exp_mode: bool = field(default=False, metadata={"help": "Use partial data for debugging."})
    exp_data_num: int = field(default=10000)
    use_prj: bool = field(default=True, metadata={"help": "Use a projection module."})
    prj_dim: int = field(default=3072, metadata={"help": "Projection hidden dim."})
    prj_dropout: float = field(default=0.0)
    prj_no_ln: bool = field(default=False, metadata={"help": "Remove LayerNorm from projection."})
    distill_loss_div_std: bool = field(default=False)
    distill_loss_type: str = field(default="smooth_l1")
    distill_loss_factor: float = field(default=1.0)
    stage_align_loss_factor: float = field(
        default=1.0,
        metadata={"help": "Weight for stage-boundary alignment against teacher states."},
    )
    explain_loss_factor: float = field(
        default=0.5,
        metadata={"help": "Weight for explain (summary reconstruction) loss."},
    )
    ce_loss_factor: float = field(
        default=2.0,
        metadata={"help": "Weight for output verdict CE loss."},
    )
    ref_loss_factor: float = field(default=1.0)
    use_baselm_explain: bool = field(
        default=True,
        metadata={"help": "Use base LM for explain loss instead of lightweight decoder."},
    )
    explain_proj_layers: int = field(
        default=2,
        metadata={"help": "Number of projection layers before feeding latent to base LM for explain."},
    )
    inf_latent_iterations: int = field(default=1)
    inf_num_iterations: int = field(default=5, metadata={"help": "Run multiple inference passes."})
    remove_eos: bool = field(default=False)
    print_ref_model_stats: bool = field(default=False)
    fix_attn_mask: bool = field(default=False)
    log_full: bool = field(default=False)
    print_loss: bool = field(default=True)
    max_token_num: int = field(default=1000, metadata={"help": "Skip overly long examples."})
    require_teacher_summaries: bool = field(
        default=False,
        metadata={"help": "Fail if teacher summaries are missing from the training data."},
    )
    summary_max_target_length: int = field(
        default=192,
        metadata={"help": "Maximum length for cached summary targets."},
    )
    load_only_model_weights: bool = field(
        default=False,
        metadata={"help": "Load only model weights from checkpoint, skipping optimizer states. Useful when resuming with different GPU count."},
    )
    intent_decoder_hidden_size: int = field(
        default=512,
        metadata={"help": "Hidden size for the Stage 1 decoder."},
    )
    intent_decoder_layers: int = field(
        default=2,
        metadata={"help": "Number of layers for the Stage 1 decoder."},
    )
    intent_decoder_num_heads: int = field(
        default=8,
        metadata={"help": "Attention heads for the Stage 1 decoder."},
    )
    risk_decoder_hidden_size: int = field(
        default=512,
        metadata={"help": "Hidden size for the Stage 2 decoder."},
    )
    risk_decoder_layers: int = field(
        default=2,
        metadata={"help": "Number of layers for the Stage 2 decoder."},
    )
    risk_decoder_num_heads: int = field(
        default=8,
        metadata={"help": "Attention heads for the Stage 2 decoder."},
    )
    decoder_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout for the lightweight summary decoders."},
    )

    def get_num_latent_list(self) -> List[int]:
        return [int(x.strip()) for x in self.num_latent_per_stage.split(",")]

    def get_stage_names(self) -> List[str]:
        return [s.strip().lower() for s in self.stage_names.split(",")]

    @property
    def total_num_latent(self) -> int:
        return sum(self.get_num_latent_list())


def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(
        f"trainable params: {trainable} || all params: {total} "
        f"|| trainable%: {100 * trainable / total:.2f}"
    )


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


def _tag_variants(tag: str) -> List[str]:
    return [tag, tag.lower(), tag.upper(), tag.capitalize()]


def extract_tagged_block(text: str, tag: str) -> Optional[str]:
    for variant in _tag_variants(tag):
        pattern = rf"<{re.escape(variant)}>(.*?)</{re.escape(variant)}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def extract_json_verdict(text: str) -> Optional[dict]:
    match = re.search(r'\{[^}]*"safe"\s*:\s*(true|false)[^}]*\}', text, re.IGNORECASE)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def extract_simple_verdict(text: str) -> Optional[dict]:
    """Parse the compact verdict format produced by the training pipeline.

    Accepted formats (case-insensitive, leading/trailing whitespace ignored):
        ``safe``
        ``unsafe, policy N``                   (single violation)
        ``unsafe, policy N1, N2, N3``          (multi-violation, new)
        ``unsafe, policies N1, N2, N3``        (multi-violation, plural form)
        ``unsafe``

    Returns a dict with:
      - ``safe``: bool
      - ``policy_index``: int or None  (first violated index; kept for
        backward compatibility with single-index downstream code)
      - ``policy_indices``: List[int]  (all violated indices; single-
        index verdicts yield a list of length 1)

    Falls back to :func:`extract_json_verdict` for JSON-style verdicts.
    """
    cleaned = text.strip()
    if re.match(r"^safe$", cleaned, re.IGNORECASE):
        return {"safe": True}
    # Multi-index form: `unsafe, policy 2, 5, 7` or `unsafe policies 2, 5, 7`
    m = re.match(
        r"^unsafe,?\s*polic(?:y|ies)\s+(\d+(?:\s*,\s*\d+)*)$",
        cleaned,
        re.IGNORECASE,
    )
    if m:
        indices = [int(s) for s in re.findall(r"\d+", m.group(1))]
        return {
            "safe": False,
            "policy_index": indices[0] if indices else None,
            "policy_indices": indices,
        }
    if re.match(r"^unsafe$", cleaned, re.IGNORECASE):
        return {"safe": False, "policy_indices": []}
    # Fallback: try the old JSON format for backward compatibility
    verdict = extract_json_verdict(text)
    if verdict is not None:
        # Normalize: if json has policy_index but not policy_indices, fill it in.
        if "policy_indices" not in verdict and verdict.get("policy_index") is not None:
            verdict["policy_indices"] = [verdict["policy_index"]]
        elif "policy_indices" in verdict and verdict.get("policy_index") is None:
            idxs = verdict["policy_indices"]
            verdict["policy_index"] = idxs[0] if idxs else None
    return verdict


def summary_tag_for_stage(stage_name: str) -> str:
    return "".join(part.capitalize() for part in stage_name.split("_")) + "Summary"


def normalize_tagged_summary(text: str, stage_name: str) -> str:
    tag = summary_tag_for_stage(stage_name)
    body = extract_tagged_block(text, tag)
    if body is None:
        body = text.strip()
    return f"<{tag}>\n{body.strip()}\n</{tag}>"


def parse_safety_stages(generated_reasoning: str, stage_names: List[str]) -> Dict[str, str]:
    stages = {}
    for name in stage_names:
        tag = "".join(part.capitalize() for part in name.split("_"))
        stages[name] = extract_tagged_block(generated_reasoning, tag) or ""
    return stages


def extract_output_block(generated_reasoning: str) -> str:
    output_body = extract_tagged_block(generated_reasoning, "Output")
    if output_body is not None:
        return f"<Output>\n{output_body.strip()}\n</Output>"
    verdict = extract_json_verdict(generated_reasoning)
    if verdict is None:
        return ""
    return f"<Output>\n{json.dumps(verdict, ensure_ascii=False)}\n</Output>"


def build_teacher_reasoning_text(generated_reasoning: str, stage_names: List[str]) -> str:
    stages = parse_safety_stages(generated_reasoning, stage_names)
    blocks = []
    for stage_name in stage_names:
        tag = "".join(part.capitalize() for part in stage_name.split("_"))
        blocks.append(f"<{tag}>\n{stages.get(stage_name, '').strip()}\n</{tag}>")
    return "\n\n".join(blocks).strip()


class LowRankProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int = 64):
        super().__init__()
        self.U = nn.Parameter(torch.randn(input_dim, rank))
        self.V = nn.Parameter(torch.randn(rank, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(x, self.U), self.V)


class StageSummaryDecoder(nn.Module):
    """A lightweight cross-attentive decoder used only during training (legacy)."""

    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        pad_token_id: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_positions: int = 256,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Decoder hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )
        self.pad_token_id = pad_token_id
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.position_embed = nn.Embedding(max_positions, hidden_size)
        self.memory_proj = nn.Linear(latent_dim, hidden_size)
        self.input_ln = nn.LayerNorm(hidden_size)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_ln = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(
        self,
        latent_states: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        if target_input_ids is None or target_labels is None:
            return latent_states.sum() * 0.0
        if (target_labels != IGNORE_INDEX).sum() == 0:
            return latent_states.sum() * 0.0

        seq_len = target_input_ids.size(1)
        if seq_len > self.position_embed.num_embeddings:
            raise ValueError(
                f"Summary target length {seq_len} exceeds decoder max positions "
                f"{self.position_embed.num_embeddings}."
            )

        compute_dtype = self.memory_proj.weight.dtype
        positions = torch.arange(seq_len, device=target_input_ids.device).unsqueeze(0)
        decoder_inputs = self.token_embed(target_input_ids) + self.position_embed(positions)
        decoder_inputs = self.input_ln(decoder_inputs).to(compute_dtype)
        memory = self.memory_proj(latent_states.to(compute_dtype))

        causal_mask = torch.full(
            (seq_len, seq_len),
            float("-inf"),
            device=target_input_ids.device,
            dtype=compute_dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)

        decoded = self.decoder(
            tgt=decoder_inputs,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_input_ids.eq(self.pad_token_id),
        )
        logits = self.lm_head(self.output_ln(decoded))
        loss = self.loss_fct(logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1))
        return loss


class ExplainProjector(nn.Module):
    """Projection layers that map latent tokens into the base LM embedding space.

    The projected latent embeddings are prepended to the summary target token
    embeddings and then fed through the base LM for next-token prediction.
    This is training-only — discarded at inference.
    """

    def __init__(self, dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.LayerNorm(dim),
                nn.Dropout(dropout),
            ])
        self.proj = nn.Sequential(*layers)

    def forward(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.proj(latent_states)


class SafetyLatentGuard(torch.nn.Module):
    """
    Latent reasoning model for safety policy compliance.

    The base causal LM performs:
      - question encoding
      - latent reasoning for Stage 1 and Stage 2
      - explicit verdict generation

    Two lightweight training-only decoders reconstruct the cached summaries:
      - Stage 1: intent summary
      - Stage 2: risk summary
    """

    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path

        self.num_latent_per_stage = training_args.get_num_latent_list()
        self.stage_names = training_args.get_stage_names()
        if len(self.stage_names) != 2:
            raise ValueError(
                f"This implementation expects exactly 2 latent stages, got {self.stage_names}."
            )
        if len(self.num_latent_per_stage) != len(self.stage_names):
            raise ValueError(
                "num_latent_per_stage and stage_names must have the same number of entries."
            )
        self.total_num_latent = training_args.total_num_latent

        load_kwargs = dict(
            torch_dtype=torch.float16 if not training_args.bf16 else torch.bfloat16,
            attn_implementation="eager",
        )
        if model_args.full_precision:
            self.codi = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        else:
            load_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
            )
            self.codi = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

        self.training = self.model_args.train
        original_vocab_size = self.codi.config.vocab_size
        self.pad_token_id = original_vocab_size
        self.bot_id = original_vocab_size + 1
        self.eot_id = original_vocab_size + 2
        self.codi.resize_token_embeddings(original_vocab_size + 3)

        self.dim = self.codi.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.pad_token_id = self.pad_token_id
        self.decoder_start_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.eos_token_id
        )
        if self.decoder_start_token_id is None:
            self.decoder_start_token_id = self.tokenizer.pad_token_id

        if training_args.use_lora:
            self.codi = get_peft_model(self.codi, lora_config)

        self.use_prj = training_args.use_prj
        if training_args.use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(training_args.prj_dropout),
                nn.Linear(self.dim, training_args.prj_dim),
                nn.GELU(),
                nn.Linear(training_args.prj_dim, self.dim),
            )
            if not training_args.prj_no_ln:
                self.prj.add_module("ln", nn.LayerNorm(self.dim))

        vocab_size = self.get_embd(self.codi, self.model_name).num_embeddings
        self.use_baselm_explain = training_args.use_baselm_explain
        if model_args.use_decoder:
            if self.use_baselm_explain:
                # New: use base LM for explain loss with projection layers
                self.explain_projectors = nn.ModuleDict(
                    {
                        name: ExplainProjector(
                            dim=self.dim,
                            num_layers=training_args.explain_proj_layers,
                            dropout=training_args.decoder_dropout,
                        )
                        for name in self.stage_names
                    }
                )
                self.explain_loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            else:
                # Legacy: lightweight standalone decoders
                self.summary_decoders = nn.ModuleDict(
                    {
                        self.stage_names[0]: StageSummaryDecoder(
                            latent_dim=self.dim,
                            vocab_size=vocab_size,
                            pad_token_id=self.tokenizer.pad_token_id,
                            hidden_size=training_args.intent_decoder_hidden_size,
                            num_layers=training_args.intent_decoder_layers,
                            num_heads=training_args.intent_decoder_num_heads,
                            dropout=training_args.decoder_dropout,
                            max_positions=training_args.summary_max_target_length,
                        ),
                        self.stage_names[1]: StageSummaryDecoder(
                            latent_dim=self.dim,
                            vocab_size=vocab_size,
                            pad_token_id=self.tokenizer.pad_token_id,
                            hidden_size=training_args.risk_decoder_hidden_size,
                            num_layers=training_args.risk_decoder_layers,
                            num_heads=training_args.risk_decoder_num_heads,
                            dropout=training_args.decoder_dropout,
                            max_positions=training_args.summary_max_target_length,
                        ),
                    }
                )

        self.print_loss = training_args.print_loss
        self.ref_loss_factor = training_args.ref_loss_factor
        self.ce_loss_factor = training_args.ce_loss_factor
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.distill_loss_div_std = training_args.distill_loss_div_std
        self.distill_loss_type = training_args.distill_loss_type
        self.distill_loss_factor = training_args.distill_loss_factor
        if self.distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif self.distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        else:
            raise NotImplementedError(f"Unknown distill_loss_type: {self.distill_loss_type}")
        self.explain_loss_factor = training_args.explain_loss_factor
        self.stage_align_loss_factor = training_args.stage_align_loss_factor
        self.fix_attn_mask = training_args.fix_attn_mask

        if self.training:
            self.init()

    def get_embd(self, model, model_name):
        try:
            try:
                return model.get_base_model().model.embed_tokens
            except Exception:
                return model.model.embed_tokens
        except AttributeError as exc:
            raise NotImplementedError(f"Cannot find embedding layer for {model_name}") from exc

    def init(self):
        print_trainable_parameters(self)
        if self.training_args.restore_from:
            print(f"Loading from checkpoint: {self.training_args.restore_from}...")
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict, strict=False)
            print(f"Finished loading from {self.training_args.restore_from}")

    def build_stage_decoder_inputs(self, stage_summary_labels: torch.Tensor) -> torch.Tensor:
        stage_inputs = stage_summary_labels.clone()
        stage_inputs = stage_inputs.masked_fill(stage_inputs.eq(IGNORE_INDEX), self.tokenizer.pad_token_id)
        start_column = torch.full(
            (stage_inputs.size(0), 1),
            self.decoder_start_token_id,
            dtype=stage_inputs.dtype,
            device=stage_inputs.device,
        )
        if stage_inputs.size(1) == 1:
            return start_column
        return torch.cat([start_column, stage_inputs[:, :-1]], dim=1)

    def forward(
        self,
        encoder_input_ids: torch.LongTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        ref_input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        ref_answer_position: Optional[torch.LongTensor] = None,
        model_answer_position: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.LongTensor] = None,
        ref_labels: torch.LongTensor = None,
        stage_summary_labels: Optional[torch.LongTensor] = None,
        stage_boundary_positions: Optional[torch.LongTensor] = None,
        step: int = None,
        step_ratio: float = None,
    ):
        del step
        del step_ratio
        if not self.fix_attn_mask:
            ref_attention_mask = None

        batch_size = encoder_input_ids.size(0)
        outputs = self.codi(
            input_ids=encoder_input_ids,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=encoder_attention_mask,
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1:, :]

        if self.use_prj:
            with autocast(dtype=torch.bfloat16, enabled=True):
                latent_embd = self.prj(latent_embd)

        explain_loss_total = encoder_input_ids.new_tensor(0.0, dtype=torch.float32)
        stage_align_loss_total = encoder_input_ids.new_tensor(0.0, dtype=torch.float32)
        effective_stage_cnt = 0
        effective_stage_align_cnt = 0
        stage_representatives = []

        for stage_idx, stage_name in enumerate(self.stage_names):
            stage_latents = []
            stage_hidden = None
            for _ in range(self.num_latent_per_stage[stage_idx]):
                with autocast(dtype=torch.bfloat16):
                    outputs = self.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values,
                    )
                past_key_values = outputs.past_key_values
                stage_hidden = outputs.hidden_states[-1][:, -1:, :]
                latent_embd = stage_hidden
                if self.use_prj:
                    with autocast(dtype=torch.bfloat16, enabled=True):
                        latent_embd = self.prj(latent_embd)
                stage_latents.append(latent_embd)

            if stage_hidden is not None:
                stage_representatives.append(stage_hidden)

            if self.model_args.use_decoder and stage_summary_labels is not None:
                stage_targets = stage_summary_labels[:, stage_idx, :]
                if (stage_targets != IGNORE_INDEX).sum() == 0:
                    continue
                latent_block = torch.cat(stage_latents, dim=1)

                if self.use_baselm_explain:
                    # Project latent tokens and feed through base LM
                    # Cast to projector dtype (float32) to avoid dtype mismatch with DeepSpeed
                    proj_module = self.explain_projectors[stage_name]
                    proj_dtype = next(proj_module.parameters()).dtype
                    projected = proj_module(latent_block.to(proj_dtype))
                    # Get token embeddings for summary targets (teacher-forced)
                    summary_inputs = self.build_stage_decoder_inputs(stage_targets)
                    summary_embds = self.get_embd(self.codi, self.model_name)(summary_inputs)
                    # Concat: [projected_latent_tokens, summary_token_embeds]
                    combined_embds = torch.cat([projected, summary_embds], dim=1)
                    with autocast(dtype=torch.bfloat16):
                        explain_out = self.codi(
                            inputs_embeds=combined_embds,
                            output_hidden_states=False,
                        )
                    # Only compute loss on summary positions (after latent prefix)
                    num_latent = projected.size(1)
                    summary_logits = explain_out.logits[:, num_latent - 1:-1, :]
                    stage_loss = self.explain_loss_fct(
                        summary_logits.reshape(-1, summary_logits.size(-1)),
                        stage_targets.reshape(-1),
                    )
                else:
                    # Legacy lightweight decoder
                    stage_inputs = self.build_stage_decoder_inputs(stage_targets)
                    stage_loss = self.summary_decoders[stage_name](
                        latent_states=latent_block,
                        target_input_ids=stage_inputs,
                        target_labels=stage_targets,
                    )
                explain_loss_total = explain_loss_total + stage_loss
                effective_stage_cnt += 1

        with torch.no_grad():
            ref_outputs = self.codi(
                input_ids=ref_input_ids,
                output_hidden_states=True,
                attention_mask=ref_attention_mask,
            )
        ref_outputs_with_grad = self.codi(
            input_ids=ref_input_ids,
            output_hidden_states=True,
            attention_mask=ref_attention_mask,
        )

        if stage_boundary_positions is not None and stage_representatives:
            teacher_top_hidden = ref_outputs.hidden_states[-1]
            for stage_idx, student_state in enumerate(stage_representatives):
                target_positions = stage_boundary_positions[:, stage_idx]
                valid_mask = target_positions.ge(0) & target_positions.lt(teacher_top_hidden.size(1))
                if not valid_mask.any():
                    continue
                teacher_state = teacher_top_hidden[valid_mask].gather(
                    1,
                    target_positions[valid_mask]
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(-1, 1, teacher_top_hidden.size(-1)),
                )
                student_selected = student_state[valid_mask]
                stage_loss = self.distill_loss_fct(student_selected, teacher_state.detach())
                if self.distill_loss_div_std:
                    stage_loss = stage_loss / teacher_state.std().clamp_min(1e-6)
                stage_align_loss_total = stage_align_loss_total + stage_loss
                effective_stage_align_cnt += 1

        if "qwen" in self.model_name.lower() or "llama" in self.model_name.lower():
            model_answer_position = model_answer_position + 1
            ref_answer_position = ref_answer_position + 1
        model_answer_position = model_answer_position - 1
        ref_answer_position = ref_answer_position - 1

        embds = self.get_embd(self.codi, self.model_name)(decoder_input_ids)

        dynamic_mask = None
        if self.fix_attn_mask:
            dynamic_mask = torch.ones(
                (encoder_attention_mask.size(0), self.total_num_latent),
                device=ref_labels.device,
            )
            decoder_mask = torch.ones((embds.size(0), embds.size(1)), dtype=torch.bool).to(dynamic_mask)
            dynamic_mask = torch.cat((encoder_attention_mask, dynamic_mask, decoder_mask), dim=1).bool()

        with autocast(dtype=torch.bfloat16):
            outputs = self.codi(
                inputs_embeds=embds,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask=dynamic_mask,
            )

        distill_loss_total = embds.new_tensor(0.0)
        for out, ref_out in zip(outputs.hidden_states, ref_outputs.hidden_states):
            ref_selected = ref_out.gather(
                1,
                ref_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_out.size(-1)),
            )
            out_selected = out.gather(
                1,
                model_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out.size(-1)),
            )
            distill_loss_tmp = self.distill_loss_fct(out_selected, ref_selected.detach())
            if self.distill_loss_div_std:
                distill_loss_tmp = distill_loss_tmp / ref_selected.std()
            distill_loss_total = distill_loss_total + distill_loss_tmp
        distill_loss_total = distill_loss_total / len(outputs.hidden_states)

        logits = outputs.logits
        ce_loss_total = self.loss_fct(logits[:, :-1, :].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        ce_loss_total = ce_loss_total * self.ce_loss_factor

        ref_logits = ref_outputs_with_grad.logits
        ref_ce_loss = self.loss_fct(
            ref_logits[:, :-1, :].reshape(-1, ref_logits.size(-1)),
            ref_labels[:, 1:].reshape(-1),
        )
        ref_ce_loss = ref_ce_loss * self.ref_loss_factor

        distill_loss_total = distill_loss_total * self.distill_loss_factor
        stage_align_loss_total = stage_align_loss_total * self.stage_align_loss_factor
        stage_align_loss_total = stage_align_loss_total / max(1, effective_stage_align_cnt)
        if self.model_args.use_decoder:
            explain_loss_total = explain_loss_total * self.explain_loss_factor
            explain_loss_total = explain_loss_total / max(1, effective_stage_cnt)

        loss = ce_loss_total + distill_loss_total + ref_ce_loss
        stage_align_loss_total = stage_align_loss_total.to(device=loss.device, dtype=loss.dtype)
        loss = loss + stage_align_loss_total
        if self.model_args.use_decoder:
            explain_loss_total = explain_loss_total.to(device=loss.device, dtype=loss.dtype)
            loss = loss + explain_loss_total

        if self.print_loss:
            if self.model_args.use_decoder:
                print(
                    f"loss={loss}, ce_loss={ce_loss_total}, distill_loss={distill_loss_total}, "
                    f"ref_ce_loss={ref_ce_loss}, explain_loss={explain_loss_total}, "
                    f"stage_align_loss={stage_align_loss_total}"
                )
            else:
                print(
                    f"loss={loss}, ce_loss={ce_loss_total}, distill_loss={distill_loss_total}, "
                    f"ref_ce_loss={ref_ce_loss}, stage_align_loss={stage_align_loss_total}"
                )

        result = {
            "loss": loss,
            "logits": logits,
            "ce_loss": ce_loss_total.detach(),
            "distill_loss": distill_loss_total.detach(),
            "ref_ce_loss": ref_ce_loss.detach(),
            "stage_align_loss": stage_align_loss_total.detach(),
        }
        if self.model_args.use_decoder:
            result["explain_loss"] = explain_loss_total.detach()
        return result
