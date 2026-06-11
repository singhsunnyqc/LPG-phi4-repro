"""
Dataset implementations for policy evaluation.

This module provides implementations for the policy guardrail evaluation
datasets: DynaBench, GuardSet-X, PolicyGuardBench, HarmBench, and WildGuard.
"""

import csv
import json
import os
import random
import re
from typing import Dict, Any, Optional, List, Iterator, Tuple

from .base import BaseDataset, DatasetSample, DatasetInfo, register_dataset
from .policy_books import (
    HARMBENCH_POLICY_BOOK,
    WILDGUARD_POLICY_BOOK,
)
from .policy_books.wildguard import SUBCATEGORY_ALIASES as WILDGUARD_ALIASES


def _sample_policy_subset(
    policy_book: Dict[str, str],
    violated_tax_ids: List[str],
    rng: random.Random,
    k_min: int = 4,
    k_max: int = 10,
) -> List[str]:
    """Build a per-example policy list, always including any violated tax_ids.

    Mirrors the per-example subset sampler used to build the training set,
    but without cross-taxonomy mixing (eval keeps the per-dataset book pure
    so F1 is interpretable).
    """
    violated = [t for t in (violated_tax_ids or []) if t in policy_book]
    seen = set()
    deduped: List[str] = []
    for t in violated:
        if t not in seen:
            deduped.append(t)
            seen.add(t)
    violated = deduped

    own_pool = [v for k, v in policy_book.items() if k not in seen]
    violated_texts = [policy_book[t] for t in violated]

    lo = max(k_min, len(violated_texts) + 1) if violated_texts else k_min
    hi = max(k_max, lo)
    k = rng.randint(lo, hi)

    remaining = max(k - len(violated_texts), 0)
    fillers = rng.sample(own_pool, k=min(remaining, len(own_pool)))

    combined = violated_texts + fillers
    rng.shuffle(combined)
    return combined


def _load_records(
    dataset_path: str,
    *,
    hf_config: Optional[str] = None,
    hf_split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load records from a local file (csv/jsonl/json/parquet) or an HF id.

    A path is treated as an HF dataset id when it has no file extension and
    contains '/'. In that case `hf_config` and `hf_split` select the slice.
    """
    has_ext = os.path.splitext(dataset_path)[1] != ""
    looks_like_hf_id = ("/" in dataset_path) and (not has_ext) and (not os.path.exists(dataset_path))

    if looks_like_hf_id:
        # The local `datasets` package shadows the HF `datasets` library.
        # Temporarily remove local paths from sys.modules and sys.path.
        import sys as _sys
        _saved_datasets = _sys.modules.get('datasets')
        _saved_subs = {k: v for k, v in list(_sys.modules.items()) if k.startswith('datasets.')}
        for k in _saved_subs:
            del _sys.modules[k]
        if 'datasets' in _sys.modules:
            del _sys.modules['datasets']
        # Remove every sys.path entry that exposes our *project-local*
        # ``datasets/`` subpackage, so that ``import datasets`` below
        # finds the HF library instead. We fingerprint the local package
        # by the presence of ``policy_datasets.py`` (our own file), so
        # that the venv's site-packages copy of HF ``datasets`` — which
        # of course also has ``__init__.py`` — is never removed.
        _rm_paths = []
        for p in _sys.path:
            probe_dir = p if p else os.getcwd()
            if p == ".":
                probe_dir = os.getcwd()
            try:
                if os.path.isfile(
                    os.path.join(probe_dir, "datasets", "policy_datasets.py")
                ):
                    _rm_paths.append(p)
            except OSError:
                pass
        for p in _rm_paths:
            try:
                _sys.path.remove(p)
            except ValueError:
                pass
        try:
            import datasets as _hf_datasets
            if hf_config:
                ds = _hf_datasets.load_dataset(dataset_path, hf_config, split=hf_split or "test")
            else:
                ds = _hf_datasets.load_dataset(dataset_path, split=hf_split or "test")
            result = [dict(r) for r in ds]
        except ImportError as e:
            raise RuntimeError(
                "huggingface `datasets` is required to load by repo id. "
                "Install with `pip install datasets`."
            ) from e
        finally:
            if _saved_datasets is not None:
                _sys.modules['datasets'] = _saved_datasets
            for k, v in _saved_subs.items():
                _sys.modules[k] = v
            for p in _rm_paths:
                _sys.path.append(p)
        return result

    ext = os.path.splitext(dataset_path)[1].lower()
    if ext == ".jsonl":
        records: List[Dict[str, Any]] = []
        with open(dataset_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    if ext == ".json":
        with open(dataset_path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    if ext == ".csv":
        with open(dataset_path, "r") as f:
            return list(csv.DictReader(f))
    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError(
                "pyarrow is required to read parquet files. "
                "Install with `pip install pyarrow`."
            ) from e
        return pq.read_table(dataset_path).to_pylist()

    raise ValueError(f"Unsupported dataset file extension: {ext}")


@register_dataset
class DynaBenchDataset(BaseDataset):
    """
    DynaBench dataset for policy guardrail evaluation.
    
    Each record contains:
    - policy: The policy text with numbered rules
    - transcript: The conversation text
    - label: "PASS" or "FAIL"
    - metadata: JSON string with rules_violated array
    - base_id: Unique identifier
    
    Returns:
        - content: The transcript text (content to evaluate)
        - policy: The policy text (separate from content)
    """
    
    @classmethod
    def get_info(cls) -> DatasetInfo:
        return DatasetInfo(
            name="dynabench",
            description="DynaBench dataset for policy guardrail evaluation",
            dataset_type="multi_policy",
            file_format="json",
        )
    
    def load(self) -> None:
        with open(self.dataset_path, "r") as f:
            self._data = json.load(f)
    
    def __iter__(self) -> Iterator[DatasetSample]:
        for item in self._data:
            policy = item.get("policy", "")
            transcript = item.get("transcript", "")
            label = item.get("label", "PASS")
            metadata_str = item.get("metadata", "{}")
            base_id = item.get("base_id", "")

            try:
                metadata = json.loads(metadata_str)
            except json.JSONDecodeError:
                metadata = {}
            
            rules_violated = metadata.get("rules_violated", [])
            safe = (label == "PASS")

            yield DatasetSample(
                sample_id=base_id,
                content=transcript,
                safe=safe,
                policies=policy,
                violated_policy_index=rules_violated[0] if rules_violated else None,
                metadata={
                    "label": label,
                    **metadata
                },
            )
    
    def __len__(self) -> int:
        return len(self._data)


@register_dataset
class GuardsetXDataset(BaseDataset):
    """
    Guardset-X dataset for policy guardrail evaluation.
    
    Supports both single-policy and multi-policy formats.
    
    Returns:
        - content: The prompt text (content to evaluate)
        - response: Optional model response (separate)
        - policies: List of policies (shuffled)
    """
    
    @classmethod
    def get_info(cls) -> DatasetInfo:
        return DatasetInfo(
            name="guardset_x",
            description="Guardset-X dataset for policy guardrail evaluation",
            dataset_type="multi_policy",
            file_format="jsonl",
        )
    
    def load(self) -> None:
        self._data = []
        with open(self.dataset_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._data.append(json.loads(line))
    
    def __iter__(self) -> Iterator[DatasetSample]:
        import random
        
        seed = self.config.get("seed", 42)
        random.seed(seed)
        
        for idx, item in enumerate(self._data):
            prompt = item.get("prompt") or item.get("instance") or ""
            response = item.get("response")
            
            policies = item.get("policy") or []
            if isinstance(policies, str):
                policies = [policies]
            
            safe = bool(item.get("safe", True))
            rule = item.get("rule")
            
            shuffled_policies = list(policies)
            random.shuffle(shuffled_policies)
            
            gt_policy_index = None
            if not safe and rule is not None and rule in shuffled_policies:
                gt_policy_index = shuffled_policies.index(rule)
            
            yield DatasetSample(
                sample_id=str(idx),
                content=prompt,
                safe=safe,
                policies=shuffled_policies,
                response=response,
                violated_policy_index=gt_policy_index,
                metadata={
                    "original_policies": policies,
                    "rule": rule,
                },
            )
    
    def __len__(self) -> int:
        return len(self._data)


@register_dataset
class PolicyGuardBenchDataset(BaseDataset):
    """
    PolicyGuardBench dataset for evaluating guardrail models on web-agent
    policy violations, including jailbreak-attack variants.

    Each record contains:
    - instruction: Task instruction (same for all items)
    - input: Contains Policy + Trajectory Actions + Domain/Subdomain
    - output: "violation" or "no_violation"
    - attack_type (optional): "reasoning_distraction", "category_shift", "instruction_force"

    Returns:
        - content: The Trajectory Actions text (content to evaluate)
        - policy: The Policy text (separate from content)
        - metadata: Contains attack_type, domain, subdomain
    """

    @classmethod
    def get_info(cls) -> DatasetInfo:
        return DatasetInfo(
            name="policyguardbench",
            description="PolicyGuardBench dataset for web-agent policy violation evaluation (with jailbreak attacks)",
            dataset_type="single_policy",
            file_format="json",
        )

    def load(self) -> None:
        # Accept either a local file (json/jsonl/parquet/csv) or an HF
        # dataset id like ``Rakancorle1/PolicyGuardBench``. The HF release
        # ships a single ``default`` config with train/test splits — pick
        # the test split unless the caller overrides via config.
        self._data = _load_records(
            self.dataset_path,
            hf_config=self.config.get("hf_config"),
            hf_split=self.config.get("hf_split", "test"),
        )

    @staticmethod
    def _parse_input(input_text: str) -> Dict[str, str]:
        """Parse the input field into policy, trajectory, domain, subdomain."""
        result = {"policy": "", "trajectory": "", "domain": "", "subdomain": ""}

        # Split at "Trajectory Actions:" to get policy and the rest
        if "Trajectory Actions:" in input_text:
            parts = input_text.split("Trajectory Actions:", 1)
            policy_part = parts[0].strip()
            rest = parts[1].strip()
        else:
            policy_part = input_text
            rest = ""

        # Extract policy (remove "Policy:" prefix if present)
        if policy_part.startswith("Policy:"):
            policy_part = policy_part[len("Policy:"):].strip()
        result["policy"] = policy_part

        # Extract domain and subdomain from the rest
        trajectory = rest
        if "\nDomain:" in rest:
            traj_parts = rest.split("\nDomain:", 1)
            trajectory = traj_parts[0].strip()
            domain_block = traj_parts[1].strip()
            # Parse domain
            domain_lines = domain_block.split("\n")
            result["domain"] = domain_lines[0].strip()
            # Parse subdomain
            for line in domain_lines[1:]:
                if line.strip().startswith("Subdomain:"):
                    result["subdomain"] = line.strip()[len("Subdomain:"):].strip()
                    break

        result["trajectory"] = trajectory
        return result

    def __iter__(self) -> Iterator[DatasetSample]:
        for idx, item in enumerate(self._data):
            input_text = item.get("input", "")
            output = item.get("output", "no_violation")
            attack_type = item.get("attack_type", None)

            parsed = self._parse_input(input_text)
            safe = (output == "no_violation")

            yield DatasetSample(
                sample_id=str(idx),
                content=parsed["trajectory"],
                safe=safe,
                policy=parsed["policy"],
                metadata={
                    "attack_type": attack_type,
                    "domain": parsed["domain"],
                    "subdomain": parsed["subdomain"],
                    "original_output": output,
                },
            )

    def __len__(self) -> int:
        return len(self._data)


@register_dataset
class HarmBenchDataset(BaseDataset):
    """HarmBench evaluation dataset (centerforaisafety/HarmBench).

    Source CSV: `data/behavior_datasets/harmbench_behaviors_text_test.csv`
    Columns: Behavior, FunctionalCategory, SemanticCategory, Tags,
             ContextString, BehaviorID.

    All HarmBench behaviors are unsafe by construction (red-teaming targets).
    Each sample is shown to LPG with the full HarmBench policy book, sampled
    down to ~K policies (always including the violated SemanticCategory)
    so the policy-list distribution matches training.

    `dataset_path` may be either a local CSV file or an HF dataset id
    (e.g. `walledai/HarmBench`).
    """

    @classmethod
    def get_info(cls) -> DatasetInfo:
        return DatasetInfo(
            name="harmbench",
            description="HarmBench red-teaming behaviors (test split, all unsafe)",
            dataset_type="multi_policy",
            file_format="csv",
        )

    def load(self) -> None:
        self._data = _load_records(
            self.dataset_path,
            hf_split=self.config.get("hf_split", "train"),
        )

    def __iter__(self) -> Iterator[DatasetSample]:
        seed = self.config.get("seed", 42)
        for idx, item in enumerate(self._data):
            behavior = (item.get("Behavior") or item.get("behavior") or "").strip()
            context = (item.get("ContextString") or item.get("context_string") or "").strip()
            sem_cat = (item.get("SemanticCategory") or item.get("semantic_category") or "").strip()
            func_cat = (item.get("FunctionalCategory") or item.get("functional_category") or "").strip()
            behavior_id = item.get("BehaviorID") or item.get("behavior_id") or str(idx)

            if not behavior:
                continue

            content = behavior if not context else f"{behavior}\n\nContext:\n{context}"

            violated_ids = [sem_cat] if sem_cat in HARMBENCH_POLICY_BOOK else []
            rng = random.Random(seed + idx)
            policies = _sample_policy_subset(
                HARMBENCH_POLICY_BOOK, violated_ids, rng
            )

            yield DatasetSample(
                sample_id=str(behavior_id),
                content=content,
                safe=False,
                policies=policies,
                metadata={
                    "semantic_category": sem_cat,
                    "functional_category": func_cat,
                },
            )

    def __len__(self) -> int:
        return sum(
            1 for r in self._data
            if (r.get("Behavior") or r.get("behavior") or "").strip()
        )


@register_dataset
class WildGuardDataset(BaseDataset):
    """WildGuardTest evaluation dataset (allenai/wildguardmix, wildguardtest).

    Columns: prompt, adversarial, response, prompt_harm_label,
             response_harm_label, response_refusal_label, subcategory.

    We classify the (prompt, response) interaction as a single sample and
    label it unsafe if either prompt_harm_label or response_harm_label is
    "harmful". The fine-grained `subcategory` is mapped to a 13-policy
    canonical taxonomy (see policy_books/wildguard.py) and included in the
    policy list when present.

    `dataset_path` may be a local parquet/jsonl/json file or the HF id
    `allenai/wildguardmix` (config `wildguardtest`, split `test`).
    """

    @classmethod
    def get_info(cls) -> DatasetInfo:
        return DatasetInfo(
            name="wildguard",
            description="WildGuardTest split for prompt+response harm classification",
            dataset_type="multi_policy",
            file_format="parquet",
        )

    def load(self) -> None:
        # When loading by HF id, the dataset has named configs; default
        # to `wildguardtest` unless caller overrides.
        path = self.dataset_path
        has_ext = os.path.splitext(path)[1] != ""
        looks_like_hf_id = ("/" in path) and (not has_ext) and (not os.path.exists(path))
        hf_config = self.config.get("hf_config")
        if looks_like_hf_id and hf_config is None:
            hf_config = "wildguardtest"
        self._data = _load_records(
            path,
            hf_config=hf_config,
            hf_split=self.config.get("hf_split", "test"),
        )

    @staticmethod
    def _map_subcategory(sub: Optional[str]) -> Optional[str]:
        """Resolve a dataset subcategory string to a canonical policy key.

        The canonical keys in WILDGUARD_POLICY_BOOK match the WildGuardTest
        `subcategory` column verbatim (e.g. ``mental_health_over-reliance_crisis``),
        so the common path is a direct lookup. We then try a normalized
        lookup in the alias table for legacy spellings.
        """
        if not sub:
            return None
        s = sub.strip().lower()
        if s in WILDGUARD_POLICY_BOOK:
            return s
        # Normalize spaces / dashes for alias lookup, but try the
        # space-only variant first so that '-' inside canonical keys
        # (e.g. 'over-reliance') still resolves correctly.
        n_space = s.replace(" ", "_")
        if n_space in WILDGUARD_ALIASES:
            return WILDGUARD_ALIASES[n_space]
        n_full = n_space.replace("-", "_")
        if n_full in WILDGUARD_POLICY_BOOK:
            return n_full
        if n_full in WILDGUARD_ALIASES:
            return WILDGUARD_ALIASES[n_full]
        return None

    def __iter__(self) -> Iterator[DatasetSample]:
        """Yield prompt-only samples for prompt-harm classification.

        We mirror the WildGuard / DynaGuard official protocol: feed the
        model only the user prompt and judge it against
        ``prompt_harm_label``. The response and ``response_harm_label`` are
        ignored here so that this F1 is directly comparable to the
        prompt-classification numbers reported in those papers.
        """
        seed = self.config.get("seed", 42)
        for idx, item in enumerate(self._data):
            prompt = (item.get("prompt") or "").strip()
            p_harm = (item.get("prompt_harm_label") or "").strip().lower()
            sub = item.get("subcategory")

            if not prompt:
                continue

            # Skip rows where the prompt label is missing (no annotator agreement)
            if p_harm not in ("harmful", "unharmful"):
                continue

            is_safe = p_harm != "harmful"

            mapped = self._map_subcategory(sub)
            violated = [mapped] if (mapped and not is_safe) else []

            rng = random.Random(seed + idx)
            policies = _sample_policy_subset(
                WILDGUARD_POLICY_BOOK, violated, rng
            )

            yield DatasetSample(
                sample_id=str(idx),
                content=prompt,
                safe=is_safe,
                policies=policies,
                metadata={
                    "subcategory": sub,
                    "mapped_category": mapped,
                    "adversarial": bool(item.get("adversarial", False)),
                    "prompt_harm_label": p_harm or None,
                    "eval_mode": "prompt_only",
                },
            )

    def __len__(self) -> int:
        return sum(
            1 for r in self._data
            if (r.get("prompt") or "").strip()
            and (r.get("prompt_harm_label") or "").strip().lower()
            in ("harmful", "unharmful")
        )


def parse_policy_to_rules(policy_text: str) -> Dict[int, str]:
    """
    Parse policy text into a dictionary of rule numbers to rule text.
    
    Args:
        policy_text: Full policy text with numbered rules
    
    Returns:
        Dictionary mapping rule number to rule text
    """
    rules = {}
    lines = policy_text.strip().split('\n')
    current_rule_num = None
    current_rule_text = []
    
    for line in lines:
        rule_match = re.match(r'^(\d+)\.\s*(.+)', line.strip())
        if rule_match:
            if current_rule_num is not None:
                rules[current_rule_num] = ' '.join(current_rule_text).strip()
            current_rule_num = int(rule_match.group(1))
            current_rule_text = [rule_match.group(2)]
        elif current_rule_num is not None:
            current_rule_text.append(line.strip())
    
    if current_rule_num is not None:
        rules[current_rule_num] = ' '.join(current_rule_text).strip()
    
    return rules
