"""
Policy Evaluate Framework

A modular framework for evaluating policy guardrail models on various datasets.

This framework provides:
- Plugin architecture for models and datasets
- Support for Qwen models (HuggingFace and VLLM)
- Support for DynaBench, Guardset-X, PolicyGuardBench, HarmBench, and WildGuard datasets
- Comprehensive evaluation metrics
- Batch evaluation support

Quick Start:
    # Single evaluation
    python evaluate.py --model qwen_hf --model_path /path/to/model \
        --dataset dynabench --dataset_path /path/to/data.json \
        --output results.json

    # Batch evaluation
    python batch_evaluate.py --config configs/example_config.json
"""

__version__ = "1.0.0"
__author__ = "Policy Evaluate Team"

from models import (
    BaseModel,
    ModelInfo,
    ModelOutput,
    ModelRegistry,
    register_model,
    QwenHuggingFaceModel,
    QwenVLLMModel,
    TrainedQwenModel,
    LatentPolicyGuardModel,
)

from datasets import (
    BaseDataset,
    DatasetInfo,
    DatasetSample,
    DatasetRegistry,
    register_dataset,
    DynaBenchDataset,
    GuardsetXDataset,
    PolicyGuardBenchDataset,
)

from metrics import (
    SafetyMetrics,
    RuleMetrics,
    EvaluationResults,
    compute_safety_metrics,
    compute_rule_metrics,
    aggregate_results,
    compute_asr,
    print_asr_summary,
)

__all__ = [
    "BaseModel",
    "ModelInfo",
    "ModelOutput",
    "ModelRegistry",
    "register_model",
    "QwenHuggingFaceModel",
    "QwenVLLMModel",
    "TrainedQwenModel",
    "LatentPolicyGuardModel",
    "BaseDataset",
    "DatasetInfo",
    "DatasetSample",
    "DatasetRegistry",
    "register_dataset",
    "DynaBenchDataset",
    "GuardsetXDataset",
    "PolicyGuardBenchDataset",
    "SafetyMetrics",
    "RuleMetrics",
    "EvaluationResults",
    "compute_safety_metrics",
    "compute_rule_metrics",
    "aggregate_results",
    "compute_asr",
    "print_asr_summary",
]
