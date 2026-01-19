# Distribution adapters: BetaAdapter, GaussianAdapter
# Provides: cdf, ppf, apply_temperature, crps methods

import logging

import numpy as np
import torch
from scipy import stats, special
from typing import Union
from .base import DistributionAdapter


logger = logging.getLogger(__name__)


class BetaAdapter(DistributionAdapter):
    """
    Distribution adapter for Beta distribution.
    
    Mathematical parameterization:
    - location = μ ∈ (0,1): mean of the Beta distribution
    - concentration = κ > 0: concentration parameter controlling spread
    
    Relationship to standard Beta(α, β):
    - α = μ * κ  
    - β = (1 - μ) * κ
    - Mean = α/(α+β) = μ
    - Variance = αβ/[(α+β)²(α+β+1)] = μ(1-μ)/(κ+1)
    
    Higher κ → lower variance (more concentrated around mean)
    Lower κ → higher variance (more spread out)
    """
    
    MIN_TEMPERATURE = 1e-3
    MAX_TEMPERATURE = 1e3

    def __init__(self, eps: float = 1e-6):
        """
        Args:
            eps: Small value to prevent numerical issues at boundaries
        """
        self.eps = eps
    
    def _to_alpha_beta(self, location: Union[np.ndarray, torch.Tensor], 
                       concentration: Union[np.ndarray, torch.Tensor]) -> tuple:
        """
        Convert (location, concentration) to standard Beta(α, β) parameters.
        
        Mathematical conversion:
        α = location * concentration  
        β = (1 - location) * concentration
        
        This preserves the mean while scaling the concentration.
        """
        alpha = location * concentration
        beta = (1 - location) * concentration
        return alpha, beta
    
    def _ensure_numpy(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert torch tensors to numpy for scipy compatibility."""
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    
    def _clip_parameters(self, location: Union[np.ndarray, torch.Tensor],
                        concentration: Union[np.ndarray, torch.Tensor]) -> tuple:
        """
        Clip parameters to valid ranges for numerical stability.
        
        Beta distribution requires:
        - location ∈ (0, 1) 
        - concentration > 0
        """
        clipped = False

        if torch.is_tensor(location):
            orig_location = location.clone()
            orig_concentration = concentration.clone()
            location = torch.clamp(location, self.eps, 1 - self.eps)
            concentration = torch.clamp(concentration, self.eps, float('inf'))
            clipped = (not torch.equal(location, orig_location)) or (
                not torch.equal(concentration, orig_concentration)
            )
        else:
            orig_location = np.asarray(location)
            orig_concentration = np.asarray(concentration)
            location = np.clip(orig_location, self.eps, 1 - self.eps)
            concentration = np.maximum(orig_concentration, self.eps)
            clipped = (
                np.any(location != orig_location)
                or np.any(concentration != orig_concentration)
            )

        if clipped:
            logger.warning(
                "BetaAdapter parameters clipped to maintain numerical stability. "
                "location∈[%s,%s], concentration≥%s.",
                self.eps,
                1 - self.eps,
                self.eps,
            )

        return location, concentration
    
    def cdf(self, y: Union[np.ndarray, torch.Tensor],
            location: Union[np.ndarray, torch.Tensor],
            concentration: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Beta cumulative distribution function.
        
        Mathematical formula:
        F(y; α, β) = I_y(α, β) where I_y is the regularized incomplete beta function
        
        For y ∈ [0,1], returns P(Y ≤ y) where Y ~ Beta(α, β)
        """
        location, concentration = self._clip_parameters(location, concentration)
        alpha, beta = self._to_alpha_beta(location, concentration)
        
        # Convert to numpy for scipy
        y_np = self._ensure_numpy(y)
        alpha_np = self._ensure_numpy(alpha)
        beta_np = self._ensure_numpy(beta)
        
        # Clip y to [0,1] for Beta support
        y_clipped = np.clip(y_np, 0.0, 1.0)
        
        # Compute CDF using scipy
        result = stats.beta.cdf(y_clipped, alpha_np, beta_np)
        
        # Convert back to original tensor type if needed
        if torch.is_tensor(y):
            return torch.from_numpy(result).to(y.device).to(y.dtype)
        return result
    
    def ppf(self, u: Union[np.ndarray, torch.Tensor],
            location: Union[np.ndarray, torch.Tensor],
            concentration: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Beta percent point function (inverse CDF).
        
        Mathematical interpretation:
        Given u ∈ [0,1], find y such that F(y; α, β) = u
        This is the u-th quantile of the Beta distribution.
        """
        location, concentration = self._clip_parameters(location, concentration)
        alpha, beta = self._to_alpha_beta(location, concentration)
        
        # Convert to numpy for scipy
        u_np = self._ensure_numpy(u)
        alpha_np = self._ensure_numpy(alpha)
        beta_np = self._ensure_numpy(beta)
        
        # Clip u to [0,1] for probability validity
        u_clipped = np.clip(u_np, self.eps, 1 - self.eps)
        
        # Compute PPF using scipy
        result = stats.beta.ppf(u_clipped, alpha_np, beta_np)
        
        # Convert back to original tensor type if needed  
        if torch.is_tensor(u):
            return torch.from_numpy(result).to(u.device).to(u.dtype)
        return result
    
    def apply_temperature(self, location: Union[np.ndarray, torch.Tensor],
                         concentration: Union[np.ndarray, torch.Tensor],
                         temperature: float) -> tuple:
        """
        Apply temperature scaling to Beta distribution.
        
        Mathematical effect:
        - location' = location (mean preserved)
        - concentration' = concentration / T
        
        Intuition:
        - T > 1: Reduces concentration → increases variance (less confident)
        - T < 1: Increases concentration → decreases variance (more confident)
        - T = 1: No change
        
        The variance scales as: Var' = Var * T / (T + concentration - 1)
        For large concentration, this approximates Var' ≈ Var * T

                Notes:
                        * Temperatures must reside in the inclusive range
                            [MIN_TEMPERATURE, MAX_TEMPERATURE]; values outside raise
                            ``ValueError`` rather than silently producing unstable updates.
                        * Parameters are clipped into ``[eps, 1-eps]`` for ``location`` and
                            ``[eps, ∞)`` for ``concentration``; a warning is emitted whenever
                            clipping occurs to aid debugging.
        """
        location, concentration = self._clip_parameters(location, concentration)
        
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        if temperature < self.MIN_TEMPERATURE or temperature > self.MAX_TEMPERATURE:
            raise ValueError(
                "Temperature %.3e outside supported range [%.3e, %.3e] for BetaAdapter"
                % (temperature, self.MIN_TEMPERATURE, self.MAX_TEMPERATURE)
            )
        
        new_concentration = concentration / temperature
        return location, new_concentration
    
    def crps(self, y: Union[np.ndarray, torch.Tensor],
             location: Union[np.ndarray, torch.Tensor],
             concentration: Union[np.ndarray, torch.Tensor], 
             temperature: float = 1.0, 
             n_grid: int = 256) -> Union[np.ndarray, torch.Tensor]:
        """
        Continuous Ranked Probability Score for Beta distribution.
        
        Mathematical formula:
        CRPS(F, y) = ∫₀¹ [F(z) - 𝟙{z ≥ y}]² dz
        
        This is computed using numerical quadrature to maintain the strictly proper
        scoring property essential for unbiased optimization.
        
        Args:
            y: Observed values
            location: Mean parameter μ ∈ (0,1) 
            concentration: Concentration parameter κ > 0
            temperature: Temperature scaling factor
            n_grid: Number of quadrature points for numerical integration
            
        Returns:
            CRPS values (always non-negative, strictly proper scoring rule)
        """
        # Apply temperature scaling: only concentration is scaled by 1/T
        location_temp, concentration_temp = self.apply_temperature(
            location, concentration, temperature)
        
        location_temp, concentration_temp = self._clip_parameters(
            location_temp, concentration_temp)
        alpha, beta = self._to_alpha_beta(location_temp, concentration_temp)
        
        # Convert everything to torch tensors for numerical integration
        if torch.is_tensor(y):
            device, dtype = y.device, y.dtype
        else:
            device, dtype = torch.device('cpu'), torch.float32
            y = torch.as_tensor(y, device=device, dtype=dtype)
            
        alpha = torch.as_tensor(alpha, device=device, dtype=dtype)
        beta = torch.as_tensor(beta, device=device, dtype=dtype)
        
        # Clip y to [0,1] for Beta support
        eps = 1e-6
        y_clipped = torch.clamp(y, eps, 1 - eps)
        
        # Create quadrature grid over [0,1]
        z = torch.linspace(0.0, 1.0, n_grid, device=device, dtype=dtype)
        step = z[1] - z[0]
        
        # Compute F(z) for all grid points
        # Broadcast: alpha/beta shape (...,) -> (..., 1), z shape (n_grid,) -> (1, n_grid)
        # Result: F_z shape (..., n_grid)
        
        # For torch.special.betainc, we need to be careful about shapes and numerical stability
        alpha_expanded = alpha.unsqueeze(-1)  # (..., 1)
        beta_expanded = beta.unsqueeze(-1)    # (..., 1)
        z_expanded = z.unsqueeze(0).expand(alpha.shape + (n_grid,))  # (..., n_grid)
        
        # Compute Beta CDF F(z) = I_z(α, β) using regularized incomplete beta function
        try:
            # torch.special.betainc(a, b, x) computes I_x(a, b) 
            F_z = torch.special.betainc(alpha_expanded, beta_expanded, z_expanded)
        except:
            # Fallback to scipy if torch special functions not available
            alpha_np = alpha.detach().cpu().numpy()
            beta_np = beta.detach().cpu().numpy()
            z_np = z.detach().cpu().numpy()
            
            # Broadcast and compute with scipy
            F_z_np = np.zeros(alpha.shape + (n_grid,))
            for idx in np.ndindex(alpha.shape):
                F_z_np[idx] = stats.beta.cdf(z_np, alpha_np[idx], beta_np[idx])
            
            F_z = torch.from_numpy(F_z_np).to(device=device, dtype=dtype)
        
        # Compute indicator function: 1{z >= y}
        # y_clipped shape (...,) -> (..., 1), z shape (n_grid,) -> (1, n_grid)
        y_expanded = y_clipped.unsqueeze(-1)  # (..., 1)
        indicator = (z >= y_expanded).to(dtype)  # (..., n_grid)
        
        # Compute integrand: [F(z) - 1{z >= y}]²
        integrand = (F_z - indicator) ** 2
        
        # Numerical integration using trapezoidal rule
        crps_values = torch.trapz(integrand, dx=step, dim=-1)
        
        # Ensure non-negative (should be guaranteed mathematically)
        crps_values = torch.clamp(crps_values, min=0.0)
        
        # Convert back to original input type if needed
        if isinstance(location, np.ndarray) or isinstance(concentration, np.ndarray):
            return crps_values.detach().cpu().numpy()
        return crps_values


class GaussianAdapter(DistributionAdapter):
    """
    Distribution adapter for Gaussian (Normal) distribution.
    
    Mathematical parameterization:
    - location = μ ∈ ℝ: mean of the Gaussian distribution
    - concentration = σ > 0: standard deviation (dispersion parameter)
    
    Standard Normal(μ, σ²) parameterization where:
    - Mean = μ  
    - Variance = σ²
    - PDF: f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
    
    Higher σ → higher variance (more spread out)
    Lower σ → lower variance (more concentrated around mean)
    """
    
    MIN_TEMPERATURE = 1e-3
    MAX_TEMPERATURE = 1e3

    def __init__(self, eps: float = 1e-6):
        """
        Args:
            eps: Small value to prevent numerical issues with σ = 0
        """
        self.eps = eps
    
    def _ensure_numpy(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert torch tensors to numpy for scipy compatibility."""
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    
    def _clip_parameters(self, location: Union[np.ndarray, torch.Tensor],
                        concentration: Union[np.ndarray, torch.Tensor]) -> tuple:
        """
        Clip parameters to valid ranges for numerical stability.
        
        Gaussian distribution requires:
        - location ∈ ℝ (no clipping needed)
        - concentration > 0 (standard deviation)
        """
        clipped = False

        if torch.is_tensor(concentration):
            orig_concentration = concentration.clone()
            concentration = torch.clamp(concentration, self.eps, float('inf'))
            clipped = not torch.equal(concentration, orig_concentration)
        else:
            orig_concentration = np.asarray(concentration)
            concentration = np.maximum(orig_concentration, self.eps)
            clipped = np.any(concentration != orig_concentration)

        if clipped:
            logger.warning(
                "GaussianAdapter concentration clipped to be ≥ %s for stability.",
                self.eps,
            )

        return location, concentration
    
    def cdf(self, y: Union[np.ndarray, torch.Tensor],
            location: Union[np.ndarray, torch.Tensor],
            concentration: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Gaussian cumulative distribution function.
        
        Mathematical formula:
        F(y; μ, σ) = Φ((y - μ)/σ) 
        where Φ is the standard normal CDF
        
        Returns P(Y ≤ y) where Y ~ Normal(μ, σ²)
        """
        location, concentration = self._clip_parameters(location, concentration)
        
        # Convert to numpy for scipy
        y_np = self._ensure_numpy(y)
        location_np = self._ensure_numpy(location)
        concentration_np = self._ensure_numpy(concentration)
        
        # Standardize: z = (y - μ) / σ
        z = (y_np - location_np) / concentration_np
        
        # Compute CDF of standard normal
        result = stats.norm.cdf(z)
        
        # Convert back to original tensor type if needed
        if torch.is_tensor(y):
            return torch.from_numpy(result).to(y.device).to(y.dtype)
        return result
    
    def ppf(self, u: Union[np.ndarray, torch.Tensor],
            location: Union[np.ndarray, torch.Tensor],
            concentration: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Gaussian percent point function (inverse CDF).
        
        Mathematical formula:
        F⁻¹(u; μ, σ) = μ + σ * Φ⁻¹(u)
        where Φ⁻¹ is the standard normal quantile function
        
        Given u ∈ [0,1], find y such that F(y; μ, σ) = u
        """
        location, concentration = self._clip_parameters(location, concentration)
        
        # Convert to numpy for scipy
        u_np = self._ensure_numpy(u)
        location_np = self._ensure_numpy(location)
        concentration_np = self._ensure_numpy(concentration)
        
        # Clip u to valid probability range
        u_clipped = np.clip(u_np, self.eps, 1 - self.eps)
        
        # Compute quantile: y = μ + σ * Φ⁻¹(u)
        z = stats.norm.ppf(u_clipped)  # Standard normal quantile
        result = location_np + concentration_np * z
        
        # Convert back to original tensor type if needed
        if torch.is_tensor(u):
            return torch.from_numpy(result).to(u.device).to(u.dtype)
        return result
    
    def apply_temperature(self, location: Union[np.ndarray, torch.Tensor],
                         concentration: Union[np.ndarray, torch.Tensor],
                         temperature: float) -> tuple:
        """
        Apply temperature scaling to Gaussian distribution.
        
        Mathematical effect:
        - location' = location (mean preserved)
        - concentration' = concentration * T
        
        Intuition:
        - T > 1: Increases standard deviation → increases variance (less confident)
        - T < 1: Decreases standard deviation → decreases variance (more confident)  
        - T = 1: No change
        
        The variance scales as: Var' = Var * T²

                Notes:
                        * Supported temperatures lie within
                            [MIN_TEMPERATURE, MAX_TEMPERATURE]; passing a value outside this
                            band triggers a ``ValueError`` instead of silently continuing.
                        * Concentration is clipped to be at least ``eps`` and a warning is
                            emitted when such clipping is applied so callers can correct the
                            source data.
        """
        location, concentration = self._clip_parameters(location, concentration)
        
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        if temperature < self.MIN_TEMPERATURE or temperature > self.MAX_TEMPERATURE:
            raise ValueError(
                "Temperature %.3e outside supported range [%.3e, %.3e] for GaussianAdapter"
                % (temperature, self.MIN_TEMPERATURE, self.MAX_TEMPERATURE)
            )
        
        new_concentration = concentration * temperature
        return location, new_concentration
    
    def crps(self, y: Union[np.ndarray, torch.Tensor],
             location: Union[np.ndarray, torch.Tensor],
             concentration: Union[np.ndarray, torch.Tensor],
             temperature: float = 1.0) -> Union[np.ndarray, torch.Tensor]:
        """
        Continuous Ranked Probability Score for Gaussian distribution.
        
        Mathematical formula for Normal(μ, σ²):
        CRPS(Normal(μ, σ²), y) = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
        
        where:
        - z = (y - μ) / σ (standardized observation)
        - Φ(z) = standard normal CDF
        - φ(z) = standard normal PDF  
        - 1/√π ≈ 0.5641895835 is a normalizing constant
        
        This has a closed-form solution for the Gaussian case.
        """
        # Apply temperature scaling
        location_temp, concentration_temp = self.apply_temperature(
            location, concentration, temperature)
        
        location_temp, concentration_temp = self._clip_parameters(
            location_temp, concentration_temp)
        
        # Convert to numpy for computation
        y_np = self._ensure_numpy(y)
        location_np = self._ensure_numpy(location_temp)
        concentration_np = self._ensure_numpy(concentration_temp)
        
        # Standardize observation: z = (y - μ) / σ
        z = (y_np - location_np) / concentration_np
        
        # Compute components
        Phi_z = stats.norm.cdf(z)  # Standard normal CDF
        phi_z = stats.norm.pdf(z)  # Standard normal PDF
        
        # CRPS formula: σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
        sqrt_pi_inv = 1.0 / np.sqrt(np.pi)  # ≈ 0.5641895835
        
        crps_values = concentration_np * (
            z * (2 * Phi_z - 1) + 2 * phi_z - sqrt_pi_inv
        )
        
        # Convert back to original tensor type if needed
        if torch.is_tensor(y):
            return torch.from_numpy(crps_values).to(y.device).to(y.dtype)
        return crps_values