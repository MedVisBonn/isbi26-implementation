import numpy as np
import torch
from typing import Union, Optional
from scipy.special import erf
from .base import Calibrator


class ThresholdCalibrator(Calibrator):
    """
    Threshold-based QC calibrator controlling recall_bad.

    Given a scalar correlate score s and target quality y (e.g. Dice),
    this learns a threshold t(tau, target_recall_bad) such that on the
    calibration (validation) set:

        recall_bad ≈ target_recall_bad

    with

        bad := {y < tau}
        reject(x)  := (s(x) <  t)  if higher_is_better
                     (s(x) >  t)  otherwise
        accept(x)  := not reject(x)

        recall_bad = P(reject | bad)

    So for target_recall_bad = 0.95 we reject ~95% of bad cases.
    """

    def __init__(self, higher_is_better: bool = True):
        """
        Args:
            higher_is_better:
                If True, larger scores mean better quality
                (accept if s >= t, reject if s < t).
                If False, smaller scores mean better quality
                (accept if s <= t, reject if s > t).
        """
        self.higher_is_better = higher_is_better

        self.s_cal: Optional[np.ndarray] = None  # raw validation scores
        self.y_cal: Optional[np.ndarray] = None  # raw validation targets
        self.n_cal: int = 0

        # Cache: (tau, target_recall_bad) -> threshold
        self._threshold_cache = {}

    def fit(
        self,
        correlates: Union[np.ndarray, 'torch.Tensor'],
        targets:   Union[np.ndarray, 'torch.Tensor'],
    ) -> 'ThresholdCalibrator':
        """
        Fit calibrator on validation correlates and targets.

        Args:
            correlates: QC scores, shape (N,) or (N, D) but will be flattened
                        to 1D scalar scores here.
            targets:    Target quality values (e.g. Dice), shape (N,).

        Returns:
            self
        """
        scores  = self._to_numpy(correlates).reshape(-1)
        targets = self._to_numpy(targets).reshape(-1)

        if scores.shape[0] != targets.shape[0]:
            raise ValueError("correlates and targets must have same number of samples")

        finite_mask = np.isfinite(scores) & np.isfinite(targets)
        scores  = scores[finite_mask]
        targets = targets[finite_mask]

        if scores.size == 0:
            raise ValueError("No finite data points for ThresholdCalibrator fitting")

        self.s_cal = scores.astype(np.float32)
        self.y_cal = targets.astype(np.float32)
        self.n_cal = self.s_cal.shape[0]

        self._threshold_cache.clear()
        print(f"[ThresholdCalibrator] Fitted on {self.n_cal} points.")

        return self

    def _compute_threshold_for_tau(
        self,
        tau: float,
        target_recall_bad: float = 0.95,
    ) -> Optional[float]:
        """
        Internal: compute score threshold for a given (tau, target_recall_bad).

        On calibration data, we want

            recall_bad = P(reject | y < tau) ≈ target_recall_bad.

        For higher_is_better=True, reject if s < thr, and we choose thr as
        the target_recall_bad-quantile of the 'bad' scores.
        """
        if self.s_cal is None or self.y_cal is None:
            raise RuntimeError("ThresholdCalibrator not fitted. Call fit() first.")

        y = self.y_cal
        s = self.s_cal

        bad_mask = (y < float(tau))
        if not np.any(bad_mask):
            # no bad cases in calibration; can't define recall_bad
            return None

        s_bad = s[bad_mask]
        n_bad = s_bad.shape[0]

        # target_recall_bad in [0,1]
        r = float(target_recall_bad)
        r = min(max(r, 0.0), 1.0)

        if r == 0.0:
            # recall_bad = 0 -> never reject -> extreme corner case
            if self.higher_is_better:
                return float(np.min(s) - 1e-6)
            else:
                return float(np.max(s) + 1e-6)

        # Quantile index among bad scores
        k = int(np.ceil(r * n_bad)) - 1
        k = max(min(k, n_bad - 1), 0)

        if self.higher_is_better:
            # bad scores should be low; reject if s < thr
            s_bad_sorted = np.sort(s_bad)          # ascending: worst -> better
            thr = float(s_bad_sorted[k])
        else:
            # bad scores should be high; reject if s > thr
            s_bad_sorted = np.sort(s_bad)[::-1]    # descending: worst -> better
            thr = float(s_bad_sorted[k])

        return thr

    def get_threshold(
        self,
        tau: float,
        target_recall_bad: float = 0.95,
        recompute: bool = False,
    ) -> Optional[float]:
        """
        Public API: get score threshold for (tau, target_recall_bad).

        On calibration data, this chooses thr so that

            recall_bad = P(reject | y < tau) ≈ target_recall_bad.

        Returns:
            threshold (float) or None if impossible (e.g. no bad cases).
        """
        key = (float(tau), float(target_recall_bad))

        if (not recompute) and key in self._threshold_cache:
            return self._threshold_cache[key]

        thr = self._compute_threshold_for_tau(
            tau=tau,
            target_recall_bad=target_recall_bad,
        )
        self._threshold_cache[key] = thr
        return thr

    def accept(
        self,
        correlates: Union[np.ndarray, 'torch.Tensor'],
        tau: float,
        target_recall_bad: float = 0.95,
        recompute: bool = False,
    ):
        """
        Decide accept/reject for new cases at a given (tau, target_recall_bad).

        Semantics:

            bad      := y < tau       (on calibration, for computing thr)
            reject   := s < thr       if higher_is_better
                        s > thr       otherwise
            accept   := not reject

            recall_bad ≈ target_recall_bad on calibration.

        Args:
            correlates: New QC scores, shape (M,) or (M, D) but flattened.
            tau:        Quality threshold (e.g. Dice).
            target_recall_bad: Desired recall of bad cases (e.g. 0.95).
            recompute:  Force threshold recomputation even if cached.

        Returns:
            decisions: boolean array of shape (M,), True = accept
            thr:       the score threshold used (or None if impossible)
        """
        thr = self.get_threshold(
            tau=tau,
            target_recall_bad=target_recall_bad,
            recompute=recompute,
        )

        scores = self._to_numpy(correlates).reshape(-1)

        if thr is None:
            # Can't define threshold; default: accept nothing
            return np.zeros_like(scores, dtype=bool), None

        if self.higher_is_better:
            # reject if s < thr
            decisions = scores >= thr
        else:
            # reject if s > thr
            decisions = scores <= thr

        return decisions.astype(bool), thr

    def cdf_tau(
        self,
        correlates: Union[np.ndarray, 'torch.Tensor'],
        tau: float,
        target_recall_bad: float = 0.95,
    ) -> np.ndarray:
        """
        Compute binary "CDF" P(y < tau | score) for KCCD API compatibility.

        Returns 1.0 for rejected cases (predicted bad) and 0.0 for accepted cases
        (predicted good), based on threshold-based QC decisions.

        This provides a fake CDF for recall metric computation: instead of smooth
        probabilities, we return hard binary decisions that work with the existing
        threshold logic (proba_bad > RISK_CUTOFF).

        Args:
            correlates: QC scores, shape (M,) or (M, D).
            tau:        Quality threshold (e.g. Dice).
            target_recall_bad: Desired recall of bad cases (e.g. 0.95).

        Returns:
            Binary probabilities, shape (M,): 1.0 = reject (bad), 0.0 = accept (good)
        """
        accept_decisions, _ = self.accept(
            correlates=correlates,
            tau=tau,
            target_recall_bad=target_recall_bad,
            recompute=False,
        )

        # Convert boolean to binary probability:
        # accept=True (good) -> 0.0 (low probability of being bad)
        # accept=False (bad) -> 1.0 (high probability of being bad)
        proba_bad = (~accept_decisions).astype(np.float32)

        return proba_bad

    def save(self, filepath: str) -> None:
        """Persist calibrator state to disk."""
        if not filepath.endswith('.pt'):
            filepath += '.pt'

        state = {
            'class': 'ThresholdCalibrator',
            'higher_is_better': self.higher_is_better,
            'n_cal': self.n_cal,
            's_cal': self.s_cal,
            'y_cal': self.y_cal,
            'threshold_cache': self._threshold_cache,
        }

        torch.save(state, filepath)

    def load(self, filepath: str) -> 'ThresholdCalibrator':
        """Load calibrator state from disk."""
        if not filepath.endswith('.pt'):
            filepath += '.pt'

        state = torch.load(filepath, map_location='cpu')

        if state.get('class') != 'ThresholdCalibrator':
            raise ValueError(f"Invalid calibrator type: {state.get('class')}")

        self.higher_is_better = state.get('higher_is_better', True)
        self.n_cal = state.get('n_cal', 0)
        self.s_cal = state.get('s_cal')
        self.y_cal = state.get('y_cal')
        self._threshold_cache = state.get('threshold_cache', {}) or {}

        # Ensure numpy arrays with consistent dtype
        if self.s_cal is not None:
            self.s_cal = np.asarray(self.s_cal, dtype=np.float32)
        if self.y_cal is not None:
            self.y_cal = np.asarray(self.y_cal, dtype=np.float32)

        return self

    @staticmethod
    def _to_numpy(x: Union[np.ndarray, 'torch.Tensor']) -> np.ndarray:
        """Convert torch tensors or arrays to numpy arrays."""
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
