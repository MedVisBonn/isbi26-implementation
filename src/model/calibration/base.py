# Base calibrator and distribution adapter protocols

from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np
import torch


class DistributionAdapter(ABC):
    """
    Abstract interface for distribution-specific operations.
    
    Isolates mathematical operations behind a unified API to enable
    calibration algorithms to work with different probability distributions
    (Beta, Gaussian, etc.) without distribution-specific logic.
    
    Key mathematical concepts:
    - CDF F(y): Probability that random variable ≤ y
    - PPF F^(-1)(u): Inverse CDF, returns y such that F(y) = u
    - Temperature scaling: Modify dispersion while preserving location
    - CRPS: Continuous Ranked Probability Score for distribution evaluation
    """
    
    @abstractmethod
    def cdf(self, y: Union[np.ndarray, torch.Tensor], 
            location: Union[np.ndarray, torch.Tensor], 
            concentration: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Cumulative Distribution Function: P(Y ≤ y | location, concentration)
        
        Args:
            y: Evaluation points, shape (...,)
            location: Location parameter (mean/center), shape (...,) or broadcastable
            concentration: Concentration parameter (dispersion), shape (...,) or broadcastable
            
        Returns:
            CDF values in [0,1], same shape as broadcasted inputs
            
        Mathematical interpretation:
            Returns the probability that a random variable from this distribution
            is less than or equal to y, given the distribution parameters.
        """
        pass
    
    @abstractmethod
    def ppf(self, u: Union[np.ndarray, torch.Tensor],
            location: Union[np.ndarray, torch.Tensor],
            concentration: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Percent Point Function (Quantile Function): F^(-1)(u)
        
        Args:
            u: Probability values in [0,1], shape (...,)
            location: Location parameter, shape (...,) or broadcastable  
            concentration: Concentration parameter, shape (...,) or broadcastable
            
        Returns:
            Quantile values, same shape as broadcasted inputs
            
        Mathematical interpretation:
            Returns the value y such that P(Y ≤ y) = u.
            This is the inverse of the CDF operation.
        """
        pass
    
    @abstractmethod
    def apply_temperature(self, location: Union[np.ndarray, torch.Tensor],
                         concentration: Union[np.ndarray, torch.Tensor],
                         temperature: float) -> tuple:
        """
        Apply temperature scaling to distribution parameters.
        
        Args:
            location: Location parameter (unchanged by temperature)
            concentration: Concentration parameter (modified by temperature)
            temperature: Temperature scaling factor T > 0
            
        Returns:
            (location_new, concentration_new) where location_new = location
            
        Mathematical interpretation:
            Temperature scaling modifies the dispersion (spread) of a distribution:
            - T > 1: Increases dispersion (wider distribution, less confident)
            - T < 1: Decreases dispersion (narrower distribution, more confident)  
            - T = 1: No change (original distribution)
            
            The location (mean/mode) is preserved to maintain the central tendency.
        """
        pass
    
    @abstractmethod
    def crps(self, y: Union[np.ndarray, torch.Tensor],
             location: Union[np.ndarray, torch.Tensor], 
             concentration: Union[np.ndarray, torch.Tensor],
             temperature: float = 1.0) -> Union[np.ndarray, torch.Tensor]:
        """
        Continuous Ranked Probability Score for this distribution.
        
        Args:
            y: Observed values, shape (N,)
            location: Location parameters, shape (N,) or broadcastable
            concentration: Concentration parameters, shape (N,) or broadcastable  
            temperature: Temperature scaling factor, default 1.0
            
        Returns:
            CRPS values, shape (N,)
            
        Mathematical interpretation:
            CRPS measures the integral squared difference between the predicted CDF
            and the empirical CDF (step function at observed value):
            
            CRPS(F, y) = ∫_{-∞}^{∞} [F(z) - 𝟙{z ≥ y}]² dz
            
            Lower CRPS indicates better probabilistic predictions.
            CRPS reduces to absolute error for deterministic forecasts.
        """
        pass


class Calibrator(ABC):
    """
    Abstract base class for calibration algorithms.
    
    Calibration improves the reliability of probabilistic predictions by
    adjusting them so that predicted probabilities match observed frequencies.
    
    Mathematical motivation:
    - Uncalibrated model: P(Y=1 | predicted_prob=0.8) ≠ 0.8
    - Calibrated model: P(Y=1 | predicted_prob=0.8) ≈ 0.8
    """
    
    @abstractmethod  
    def fit(self, *args, **kwargs) -> 'Calibrator':
        """
        Fit the calibration mapping using validation data.
        
        The calibration is learned from validation data and then applied
        to test/inference data to improve probability calibration.
        
        Returns:
            self (for method chaining)
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save calibrator state to file for later use."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> 'Calibrator':
        """Load calibrator state from file.""" 
        pass