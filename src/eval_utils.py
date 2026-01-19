from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, MutableMapping, Optional, Set, Tuple
import torch


logger = logging.getLogger(__name__)

# Type alias for calibrator selection
CalibratorType = Literal["temperature_beta", "temperature_gaussian", "temperature_laplace", "kccd"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvalDataConfig:
	"""Declarative configuration for locating evaluation artefacts.

	Parameters
	----------
	dataset_id:
		Identifier that appears in filenames (e.g. ``pmri``).
	split_id:
		Required corpus split token (e.g. ``threet-to-onepointfivet``).  This
		keeps loaders from accidentally globbing across mismatched domains.
	metric_id:
		Metric token embedded in result filenames (e.g. ``dice`` or
		``score``).
	calibration_source:
		Which split should be treated as "calibration" when a results file
		contains multiple partitions.  Common values are ``"val"`` (default)
		or ``"train"``.
	zero_inflated:
		Whether downstream consumers should expect a point-mass at zero (used
		for certain score-agreement variants).
	include_variants:
		Filters applied after directory discovery.  Typical entries include
		``"adversarial"``, ``"no-adversarial"``, ``"score-agreement"``, or the
		distribution family such as ``"beta-regression"`` and ``"mse"``.
	neural_root:
		Base directory containing method subdirectories (beta/, location_mse/, etc.).
		Each method subdirectory contains run directories with eval_data.pkl files.
	score_agreement_root:
		Directory containing score-agreement results (can be subdirectory of neural_root).
	run_aggregation:
		Strategy for handling multiple runs of the same method:
		- "first": Use first run (by iterator number) (default)
		- "all": Load all runs as separate methods with _run{i} suffix
	selected_runs:
		Optional set of run indices to load. If None, uses run_aggregation strategy.
		Example: {0, 2, 4} loads only runs 0, 2, and 4.
	temperature_mode:
		Controls how temperature scaling should behave downstream.  The loader
		itself does not act on this flag; it merely conveys intent (``"auto"``
		for CRPS-driven tuning, ``"fixed"`` to bypass optimisation, etc.).
	"""

	dataset_id: str
	split_id: str
	metric_id: str
	calibration_source: str = "val"
	zero_inflated: bool = False
	include_variants: Set[str] = field(default_factory=set)
	neural_root: Path = field(
		default_factory=lambda: Path("../../results/isbi_result").resolve()
	)
	score_agreement_root: Path = field(
		default_factory=lambda: Path("../../results/isbi_result/score-agreement").resolve()
	)
	mahalanobis_root: Path = field(
		default_factory=lambda: Path("../../results/isbi_result/mahalanobis_single").resolve()
	)
	entropy_root: Path = field(
		default_factory=lambda: Path("../../results/isbi_result/comp_entropy_ensemble").resolve()
	)
	run_aggregation: Literal["first", "all"] = "first"
	selected_runs: Optional[Set[int]] = None
	temperature_mode: str = "auto"

	def variant_allowed(self, label: str) -> bool:
		"""Return ``True`` if *label* passes any configured include filters.

		When :attr:`include_variants` is empty we accept all labels.  Otherwise
		a label must contain at least one of the requested tokens (case
		insensitive).
		"""

		if not self.include_variants:
			return True
		norm = label.lower()
		return any(tok.lower() in norm for tok in self.include_variants)


# ---------------------------------------------------------------------------
# Filesystem inventory
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MethodDescriptor:
	"""Minimal description of a discovered method variant."""

	label: str
	path: Path
	variant: str
	adversarial: str  # "none", "location", "concentration", or "location+concentration"
	source: str
	run_index: Optional[int] = None  # Run iterator (0, 1, 2, ...) for multi-run methods


class ResultInventory:
	"""Filesystem discovery helper for calibration experiments.

	The class is purposely stateless apart from the configuration pointer; all
	expensive operations cache their results so repeated calls are cheap.
	"""

	def __init__(self, config: EvalDataConfig) -> None:
		self._cfg = config
		self._score_agreement_cache: Optional[List[Path]] = None
		self._mahalanobis_cache: Optional[List[Path]] = None
		self._entropy_cache: Optional[List[Path]] = None
		self._nn_cache: Optional[List[Path]] = None

	# -- public discovery -------------------------------------------------

	def available_datasets_metrics(self) -> Tuple[List[str], List[str]]:
		"""Return sorted datasets and metrics inferred from the result roots."""

		datasets: Set[str] = set()
		metrics: Set[str] = set()

		for file in self._score_agreement_files():
			parts = file.stem.split("_")
			if len(parts) >= 3:
				datasets.add(parts[0])
				metrics.add(parts[2])

		for path in self._nn_eval_files():
			# path.parent.name looks like ``dataset_split_metric-etc``
			tokens = path.parent.name.split("_")
			if len(tokens) >= 3:
				datasets.add(tokens[0])
				metrics.add(tokens[2])

		if not datasets:
			datasets.add(self._cfg.dataset_id)
		if not metrics:
			metrics.add(self._cfg.metric_id)

		return sorted(datasets), sorted(metrics)

	def available_method_labels(self) -> Dict[str, MethodDescriptor]:
		"""Enumerate method labels that satisfy configuration filters.
		
		Handles multi-run aggregation according to cfg.run_aggregation strategy:
		- "first": Use first run by iterator number
		- "all": Return all runs with _run{i} suffix
		"""
		descriptors: Dict[str, MethodDescriptor] = {}

		# Correlate-based methods with multi-run support
		# Group runs by method type: score-agreement, mahalanobis_single, comp_entropy_ensemble
		sa_runs: List[Tuple[Path, int]] = []
		maha_runs: List[Tuple[Path, int]] = []
		entropy_runs: List[Tuple[Path, int]] = []
		
		# Parse score-agreement files
		for file in self._score_agreement_files():
			if not self._matches_primary_tokens(file.stem.split("_")):
				continue
			fname = file.stem
			# Pattern: {dataset}_{split}_{metric}_score-agreement-15-{iterator}.pt
			parts = fname.split('-')
			if len(parts) >= 3:
				try:
					iterator = int(parts[-1])
					if self._cfg.selected_runs is None or iterator in self._cfg.selected_runs:
						sa_runs.append((file, iterator))
				except ValueError:
					logger.warning(f"Could not parse iterator from score-agreement file: {file.name}")
		
		# Parse mahalanobis files
		for file in self._mahalanobis_files():
			if not self._matches_primary_tokens(file.stem.split("_")):
				continue
			fname = file.stem
			# Pattern: {dataset}_{split}_{metric}_mahalanobis-{algorithm}-swivel{indices}-{iterator}.pt
			parts = fname.split('-')
			if len(parts) >= 2:
				try:
					iterator = int(parts[-1])
					if self._cfg.selected_runs is None or iterator in self._cfg.selected_runs:
						maha_runs.append((file, iterator))
				except ValueError:
					logger.warning(f"Could not parse iterator from mahalanobis file: {file.name}")
		
		# Parse comp-entropy files
		for file in self._entropy_files():
			if not self._matches_primary_tokens(file.stem.split("_")):
				continue
			fname = file.stem
			# Pattern: {dataset}_{split}_{metric}_comp-entropy-15-{iterator}.pt
			parts = fname.split('-')
			if len(parts) >= 3:
				try:
					iterator = int(parts[-1])
					if self._cfg.selected_runs is None or iterator in self._cfg.selected_runs:
						entropy_runs.append((file, iterator))
				except ValueError:
					logger.warning(f"Could not parse iterator from comp-entropy file: {file.name}")
		
		# Apply aggregation strategy for score-agreement
		label = "score-agreement"
		if self._cfg.variant_allowed(label) and sa_runs:
			sa_runs.sort(key=lambda x: x[1])  # Sort by iterator
			
			if self._cfg.run_aggregation == "first":
				file, iterator = sa_runs[0]
				descriptors[label] = MethodDescriptor(
					label=label,
					path=file,
					variant="score-agreement",
					adversarial="none",
					source="score_agreement",
					run_index=iterator
				)
			
			elif self._cfg.run_aggregation == "all":
				# Create separate descriptors for each run
				for file, iterator in sa_runs:
					run_label = f"{label}_run{iterator}"
					descriptors[run_label] = MethodDescriptor(
						label=run_label,
						path=file,
						variant="score-agreement",
						adversarial="none",
						source="score_agreement",
						run_index=iterator
					)
		
		# Apply aggregation strategy for mahalanobis_single
		label = "mahalanobis_single"
		if self._cfg.variant_allowed(label) and maha_runs:
			maha_runs.sort(key=lambda x: x[1])
			
			if self._cfg.run_aggregation == "first":
				file, iterator = maha_runs[0]
				descriptors[label] = MethodDescriptor(
					label=label,
					path=file,
					variant="mahalanobis_single",
					adversarial="none",
					source="mahalanobis",
					run_index=iterator
				)
			
			elif self._cfg.run_aggregation == "all":
				for file, iterator in maha_runs:
					run_label = f"{label}_run{iterator}"
					descriptors[run_label] = MethodDescriptor(
						label=run_label,
						path=file,
						variant="mahalanobis_single",
						adversarial="none",
						source="mahalanobis",
						run_index=iterator
					)
		
		# Apply aggregation strategy for comp_entropy_ensemble
		label = "comp_entropy_ensemble"
		if self._cfg.variant_allowed(label) and entropy_runs:
			entropy_runs.sort(key=lambda x: x[1])
			
			if self._cfg.run_aggregation == "first":
				file, iterator = entropy_runs[0]
				descriptors[label] = MethodDescriptor(
					label=label,
					path=file,
					variant="comp_entropy_ensemble",
					adversarial="none",
					source="entropy",
					run_index=iterator
				)
			
			elif self._cfg.run_aggregation == "all":
				for file, iterator in entropy_runs:
					run_label = f"{label}_run{iterator}"
					descriptors[run_label] = MethodDescriptor(
						label=run_label,
						path=file,
						variant="comp_entropy_ensemble",
						adversarial="none",
						source="entropy",
						run_index=iterator
					)

		# Neural network variants with multi-run support
		# Group runs by method label
		runs_by_method: Dict[str, List[Tuple[Path, int]]] = {}
		
		for path in self._nn_eval_files():
			parent_name = path.parent.name
			parsed = self._parse_label_from_dirname(parent_name)
			if parsed is None:
				logger.debug(f"Could not parse method from: {parent_name}")
				continue
			
			label, iterator = parsed
			
			# Apply run filtering if specified
			if self._cfg.selected_runs is not None:
				if iterator not in self._cfg.selected_runs:
					continue
			
			if not self._cfg.variant_allowed(label):
				continue
			
			if label not in runs_by_method:
				runs_by_method[label] = []
			runs_by_method[label].append((path, iterator))
		
		# Apply run aggregation strategy
		for label, runs in runs_by_method.items():
			runs.sort(key=lambda x: x[1])  # Sort by iterator
			
			if self._cfg.run_aggregation == "first":
				path, iterator = runs[0]
				descriptors[label] = self._create_descriptor(label, path, iterator)
			
			elif self._cfg.run_aggregation == "all":
				# Create separate descriptors for each run
				for path, iterator in runs:
					run_label = f"{label}_run{iterator}"
					descriptors[run_label] = self._create_descriptor(label, path, iterator)
		
		return descriptors
	
	def _create_descriptor(self, label: str, path: Path, iterator: int) -> MethodDescriptor:
		"""Helper to create MethodDescriptor from parsed information."""
		return MethodDescriptor(
			label=label,
			path=path,
			variant=self._extract_variant(label),
			adversarial=self._is_adversarial(label),
			source="neural",
			run_index=iterator
		)

	# -- internal helpers -------------------------------------------------

	def _matches_primary_tokens(self, tokens: Iterable[str]) -> bool:
		"""Check dataset/split/metric tokens against the configuration."""

		toks = list(tokens)
		if len(toks) < 3:
			return False
		dataset, split, metric = toks[0], toks[1], toks[2]
		return (
			dataset == self._cfg.dataset_id
			and split == self._cfg.split_id
			and metric == self._cfg.metric_id
		)

	def _score_agreement_files(self) -> List[Path]:
		"""Find all score-agreement files matching dataset/split/metric.
		
		Supports multi-run pattern: {dataset}_{split}_{metric}_score-agreement-15-{iterator}.pt
		"""
		if self._score_agreement_cache is None:
			root = self._cfg.score_agreement_root
			pattern = f"{self._cfg.dataset_id}_{self._cfg.split_id}_{self._cfg.metric_id}_score-agreement-15-*.pt"
			files = sorted(root.glob(pattern)) if root.exists() else []
			if not files:
				logger.debug("No score-agreement files matched pattern %s", pattern)
			self._score_agreement_cache = files
		return self._score_agreement_cache

	def _mahalanobis_files(self) -> List[Path]:
		"""Find all mahalanobis files matching dataset/split/metric.
		
		Supports multi-run pattern: {dataset}_{split}_{metric}_mahalanobis-*-swivel*-{iterator}.pt
		"""
		if self._mahalanobis_cache is None:
			root = self._cfg.mahalanobis_root
			pattern = f"{self._cfg.dataset_id}_{self._cfg.split_id}_{self._cfg.metric_id}_mahalanobis-*-swivel*-*.pt"
			files = sorted(root.glob(pattern)) if root.exists() else []
			if not files:
				logger.debug("No mahalanobis files matched pattern %s", pattern)
			self._mahalanobis_cache = files
		return self._mahalanobis_cache

	def _entropy_files(self) -> List[Path]:
		"""Find all comp-entropy files matching dataset/split/metric.
		
		Supports multi-run pattern: {dataset}_{split}_{metric}_comp-entropy-15-{iterator}.pt
		"""
		if self._entropy_cache is None:
			root = self._cfg.entropy_root
			pattern = f"{self._cfg.dataset_id}_{self._cfg.split_id}_{self._cfg.metric_id}_comp-entropy-15-*.pt"
			files = sorted(root.glob(pattern)) if root.exists() else []
			if not files:
				logger.debug("No comp-entropy files matched pattern %s", pattern)
			self._entropy_cache = files
		return self._entropy_cache

	def _nn_eval_files(self) -> List[Path]:
		"""Scan method subdirectories for eval_data.pkl files with run iterators."""
		if self._nn_cache is None:
			root = self._cfg.neural_root
			if not root.exists():
				logger.warning(f"Neural root does not exist: {root}")
				self._nn_cache = []
				return self._nn_cache
			
			files = []
			# Iterate through method directories (beta, location_beta, location_mse, etc.)
			for method_dir in root.iterdir():
				if not method_dir.is_dir() or method_dir.name.startswith('.'):
					continue
				
				# Skip score-agreement directory (handled separately)
				if method_dir.name == 'score-agreement':
					continue
				
				# Find all run directories matching the dataset/split/metric pattern
				# Pattern: {dataset}_{split}_{metric}_*_*/eval_data.pkl
				pattern = f"{self._cfg.dataset_id}_{self._cfg.split_id}_{self._cfg.metric_id}_*/eval_data.pkl"
				method_files = list(method_dir.glob(pattern))
				files.extend(method_files)
			
			files.sort()
			if not files:
				logger.debug(f"No neural results found in {root}")
			else:
				logger.info(f"Found {len(files)} eval_data.pkl files across method directories")
			
			self._nn_cache = files
		return self._nn_cache

	@staticmethod
	def _parse_label_from_dirname(dirname: str) -> Optional[Tuple[str, int]]:
		"""Parse method label and run iterator from directory name.
		
		Returns (label, iterator) or None if parsing fails.
		
		Examples:
		  "mnmv2_pathology-norm-vs-fall-scanners-all_dice_predictor-adversarial-location_mse-debug_2_2025-10-29..."
		  Returns: ("location_mse", 2)
		  
		  "pmri_promise12_dice_predictor-adversarial-location_beta-debug_0_2025-10-26..."
		  Returns: ("location_beta", 0)
		"""
		parts = dirname.split("_")
		if len(parts) < 6:
			return None
		
		# Skip dataset, split, metric (first 3 parts)
		# Collect method descriptor parts until we hit the iterator (digit-only part)
		method_parts = []
		iterator = None
		
		for i, part in enumerate(parts[3:], start=3):
			# Check if this is the iterator (single digit or digits)
			if part.isdigit():
				iterator = int(part)
				break
			method_parts.append(part)
		
		if not method_parts or iterator is None:
			return None
		
		# Reconstruct method descriptor string
		method_str = "_".join(method_parts).lower()
		
		# Extract adversarial type
		adv_type = "none"
		if "predictor-adversarial-location-concentration" in method_str or "location-concentration" in method_str:
			adv_type = "location+concentration"
		elif "predictor-adversarial-concentration" in method_str:
			adv_type = "concentration"
		elif "predictor-adversarial-location" in method_str:
			adv_type = "location"
		
		# Extract variant
		variant = None
		if "betahomoscedastic" in method_str:
			variant = "betahomoscedastic"
		elif "beta" in method_str:
			variant = "beta"
		elif "mse" in method_str:
			variant = "mse"
		elif "laplace" in method_str:
			variant = "laplace"
		elif "gaussian" in method_str:
			variant = "gaussian"
		
		if variant is None:
			return None
		
		# Construct label
		label = f"{adv_type}_{variant}" if adv_type != "none" else variant
		
		return (label, iterator)

	@staticmethod
	def _extract_variant(label: str) -> str:
		parts = label.split("_", maxsplit=1)
		return parts[1] if len(parts) == 2 else label

	@staticmethod
	def _is_adversarial(label: str) -> str:
		"""Extract adversarial strategy from label.
		
		Returns one of: "none", "location", "concentration", "location+concentration"
		"""
		parts = label.split("_", maxsplit=1)
		if len(parts) == 0:
			return "none"
		adv_part = parts[0].lower()
		# Return the adversarial strategy directly
		return adv_part if adv_part in {"none", "location", "concentration", "location+concentration"} else "none"


# ---------------------------------------------------------------------------
# Payload loaders
# ---------------------------------------------------------------------------


def _load_score_agreement_payload(path: Path, split_id: str) -> Dict[str, Any]:
	"""Load score-agreement result file and partition by train/val/test splits.

	Parameters
	----------
	path:
		Path to the `.pt` file containing score-agreement results.
	split_id:
		Split identifier used to build section keys (e.g. ``threet-to-onepointfivet``).

	Returns
	-------
	Dict mapping section names (``"train"``, ``"val"``, ``"test"``) to their
	tensors, plus ``"__raw__"`` containing the full checkpoint dict.
	"""
	if torch is None:
		raise ImportError("torch is required to load score-agreement payloads")
	
	blob = torch.load(path, map_location="cpu", weights_only=True)
	sections = {}
	for section in ("train", "val", "test"):
		key = f"{split_id}_{section}"
		if key in blob:
			sections[section] = blob[key]
	sections["__raw__"] = blob
	return sections


def _load_mahalanobis_payload(path: Path, split_id: str) -> Dict[str, Any]:
	"""Load mahalanobis result file and partition by train/val/test splits.

	Parameters
	----------
	path:
		Path to the `.pt` file containing mahalanobis results.
	split_id:
		Split identifier used to build section keys.

	Returns
	-------
	Dict mapping section names to their tensors, plus ``"__raw__"``.
	
	Notes
	-----
	Mahalanobis files have structure similar to score-agreement but use
	'mahalanobis' as the correlate field instead of 'dice_agreement'.
	"""
	if torch is None:
		raise ImportError("torch is required to load mahalanobis payloads")
	
	blob = torch.load(path, map_location="cpu", weights_only=True)
	sections = {}
	for section in ("train", "val", "test"):
		key = f"{split_id}_{section}"
		if key in blob:
			sections[section] = blob[key]
	sections["__raw__"] = blob
	return sections


def _load_entropy_payload(path: Path, split_id: str) -> Dict[str, Any]:
	"""Load comp-entropy result file and partition by train/val/test splits.

	Parameters
	----------
	path:
		Path to the `.pt` file containing comp-entropy results.
	split_id:
		Split identifier used to build section keys.

	Returns
	-------
	Dict mapping section names to their tensors, plus ``"__raw__"``.
	
	Notes
	-----
	Entropy files have structure similar to score-agreement but use
	'entropy' as the correlate field instead of 'dice_agreement'.
	"""
	if torch is None:
		raise ImportError("torch is required to load entropy payloads")
	
	blob = torch.load(path, map_location="cpu", weights_only=True)
	sections = {}
	for section in ("train", "val", "test"):
		key = f"{split_id}_{section}"
		if key in blob:
			sections[section] = blob[key]
	sections["__raw__"] = blob
	return sections


def _load_neural_payload(path: Path, split_id: str) -> Dict[str, Any]:
	"""Load neural predictor eval_data.pkl and partition by train/val/test splits.

	Parameters
	----------
	path:
		Path to the ``eval_data.pkl`` file.
	split_id:
		Split identifier used to build section keys.

	Returns
	-------
	Dict mapping section names to their evaluation dicts, plus ``"metadata"``
	containing the checkpoint metadata.
	"""
	with path.open("rb") as f:
		blob = pickle.load(f)
	evaluation = blob.get("evaluation", {})
	sections = {}
	for section in ("train", "val", "test"):
		key = f"{split_id}_{section}"
		if key in evaluation:
			sections[section] = evaluation[key]
	sections["metadata"] = blob.get("metadata", {})
	return sections


def load_method_payloads(
	inventory: ResultInventory, 
	config: EvalDataConfig
) -> Dict[str, Dict[str, Any]]:
	"""Load all evaluation payloads discovered by the inventory.

	Parameters
	----------
	inventory:
		Configured inventory instance that has discovered method labels.
	config:
		Configuration containing split_id and other settings.

	Returns
	-------
	Dict mapping method labels to payload dicts, each containing:
		- ``variant``: variant name (e.g. ``"beta-regression"``)
		- ``adversarial``: bool or None indicating adversarial training
		- ``source``: ``"score_agreement"`` or ``"neural"``
		- ``path``: Path to the source file
		- ``sections``: Dict mapping ``"train"``/``"val"``/``"test"`` to data

	Notes
	-----
	The returned structure provides split-level data ready for calibration
	fitting and evaluation. For neural methods, ``sections[split]`` contains
	tensors like ``mu``, ``kappa``, ``true_score``, etc. For score-agreement,
	``sections[split]`` contains ``scores`` dict with ``dice`` and
	``dice_agreement`` arrays.
	"""
	descriptors = inventory.available_method_labels()
	payloads: Dict[str, Dict[str, Any]] = {}
	for label, descriptor in descriptors.items():
		if descriptor.source == "score_agreement":
			sections = _load_score_agreement_payload(descriptor.path, config.split_id)
		elif descriptor.source == "mahalanobis":
			sections = _load_mahalanobis_payload(descriptor.path, config.split_id)
		elif descriptor.source == "entropy":
			sections = _load_entropy_payload(descriptor.path, config.split_id)
		else:
			sections = _load_neural_payload(descriptor.path, config.split_id)
		payloads[label] = {
			"variant": descriptor.variant,
			"adversarial": descriptor.adversarial,
			"source": descriptor.source,
			"path": descriptor.path,
			"sections": sections,
		}
	return payloads