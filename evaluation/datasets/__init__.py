"""
Datasets module for Policy Evaluate framework.

This module provides the dataset plugin system including:
- BaseDataset: Abstract base class for all datasets
- DatasetRegistry: Registry for discovering and loading datasets
- Dataset implementations: DynaBench, Guardset-X, PolicyGuardBench,
  HarmBench, WildGuard
"""

from .base import (
    BaseDataset,
    DatasetInfo,
    DatasetSample,
    DatasetRegistry,
    register_dataset,
)

from .policy_datasets import (
    DynaBenchDataset,
    GuardsetXDataset,
    PolicyGuardBenchDataset,
    HarmBenchDataset,
    WildGuardDataset,
    parse_policy_to_rules,
)

__all__ = [
    "BaseDataset",
    "DatasetInfo",
    "DatasetSample",
    "DatasetRegistry",
    "register_dataset",
    "DynaBenchDataset",
    "GuardsetXDataset",
    "PolicyGuardBenchDataset",
    "HarmBenchDataset",
    "WildGuardDataset",
    "parse_policy_to_rules",
]
