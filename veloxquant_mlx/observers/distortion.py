from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from veloxquant_mlx.core.abstractions import QuantizationObserver
from veloxquant_mlx.core.constants import LOWER_MSE_FACTOR, UPPER_MSE_FACTOR
from veloxquant_mlx.observers.base import QuantizationEvent


@dataclass
class DistortionReport:
    """Summary of empirical distortion vs. theoretical bounds.

    Attributes:
        empirical_mse: Observed mean squared reconstruction error.
        theoretical_mse_upper: TurboQuant MSE upper bound √(3π)/2 · 4^(-b).
        theoretical_mse_lower: Information-theoretic lower bound 4^(-b).
        mse_ratio: empirical_mse / theoretical_mse_upper (should be <= 1 + ε).
        empirical_ip_distortion: Mean squared inner-product error.
        n_samples: Number of vectors observed.
    """

    empirical_mse: float
    theoretical_mse_upper: float
    theoretical_mse_lower: float
    mse_ratio: float
    empirical_ip_distortion: float
    n_samples: int

    def __repr__(self) -> str:
        return (
            f"DistortionReport("
            f"mse={self.empirical_mse:.6f}, "
            f"upper={self.theoretical_mse_upper:.6f}, "
            f"lower={self.theoretical_mse_lower:.6f}, "
            f"ratio={self.mse_ratio:.3f}, "
            f"n={self.n_samples})"
        )


class DistortionObserver(QuantizationObserver):
    """Computes running MSE and inner-product distortion against theory.

    Expects events to carry 'x_original' and 'x_reconstructed' arrays
    in the metadata dict.  The observer accumulates squared errors and
    computes theoretical bounds when report() is called.

    Theoretical bounds (TurboQuant paper, Theorem 1):
        D_mse_upper(b) = √(3π)/2 · 4^(-b)
        D_mse_lower(b) = 4^(-b)
        D_ip_upper(b, d, ‖y‖²) = √(3π)/2 · ‖y‖²/d · 4^(-b)

    Args:
        b: Bit-width used (for bound computation).
        d: Vector dimension.
        query: Optional fixed query vector for IP distortion tracking (numpy).
    """

    def __init__(self, b: int = 2, d: int = 128, query: Optional[np.ndarray] = None) -> None:
        self._b = b
        self._d = d
        self._query = query
        self._mse_sum: float = 0.0
        self._ip_sq_sum: float = 0.0
        self._n: int = 0

    def on_event(self, event: QuantizationEvent) -> None:
        """Accumulate distortion from a pipeline event.

        Args:
            event: Must have metadata keys 'x_original' and 'x_reconstructed'
                   as numpy arrays of shape (batch, d).
        """
        x_orig = event.metadata.get("x_original")
        x_recon = event.metadata.get("x_reconstructed")
        if x_orig is None or x_recon is None:
            return

        x_orig = np.asarray(x_orig, dtype=np.float64)
        x_recon = np.asarray(x_recon, dtype=np.float64)
        diff = x_orig - x_recon
        self._mse_sum += float(np.sum(diff**2, axis=-1).mean())
        self._n += 1

        if self._query is not None:
            true_ip = x_orig @ self._query
            approx_ip = x_recon @ self._query
            self._ip_sq_sum += float(np.mean((true_ip - approx_ip) ** 2))

    @staticmethod
    def theoretical_mse_upper(b: int) -> float:
        """Upper bound on MSE: √(3π)/2 · 4^(-b)."""
        return UPPER_MSE_FACTOR * 4.0 ** (-b)

    @staticmethod
    def theoretical_mse_lower(b: int) -> float:
        """Lower bound on MSE: 4^(-b)."""
        return LOWER_MSE_FACTOR * 4.0 ** (-b)

    @staticmethod
    def theoretical_ip_upper(b: int, d: int, y_norm_sq: float) -> float:
        """Upper bound on IP distortion: √(3π)/2 · ‖y‖²/d · 4^(-b)."""
        return UPPER_MSE_FACTOR * y_norm_sq / d * 4.0 ** (-b)

    @staticmethod
    def theoretical_ip_lower(b: int, d: int, y_norm_sq: float) -> float:
        """Lower bound on IP distortion: ‖y‖²/d · 4^(-b)."""
        return LOWER_MSE_FACTOR * y_norm_sq / d * 4.0 ** (-b)

    def report(self) -> DistortionReport:
        """Compute and return the distortion report.

        Returns:
            DistortionReport with empirical and theoretical values.
        """
        if self._n == 0:
            empirical_mse = 0.0
            ip_dist = 0.0
        else:
            empirical_mse = self._mse_sum / self._n
            ip_dist = self._ip_sq_sum / self._n

        upper = self.theoretical_mse_upper(self._b)
        lower = self.theoretical_mse_lower(self._b)
        ratio = empirical_mse / upper if upper > 0 else float("inf")

        return DistortionReport(
            empirical_mse=empirical_mse,
            theoretical_mse_upper=upper,
            theoretical_mse_lower=lower,
            mse_ratio=ratio,
            empirical_ip_distortion=ip_dist,
            n_samples=self._n,
        )

    def plot(self, save_path: str) -> None:
        """Plot MSE distortion vs. bit-width, reproducing Figure 3 from TurboQuant.

        Requires matplotlib.

        Args:
            save_path: File path to save the generated figure.
        """
        import matplotlib.pyplot as plt

        bit_widths = [1, 2, 3, 4, 5]
        upper_bounds = [self.theoretical_mse_upper(b) for b in bit_widths]
        lower_bounds = [self.theoretical_mse_lower(b) for b in bit_widths]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(bit_widths, upper_bounds, "r--", label="Upper bound (√(3π)/2·4⁻ᵇ)")
        ax.semilogy(bit_widths, lower_bounds, "g--", label="Lower bound (4⁻ᵇ)")
        if self._n > 0:
            current_report = self.report()
            ax.semilogy(
                [self._b],
                [current_report.empirical_mse],
                "bo",
                markersize=8,
                label=f"Empirical (b={self._b}, n={self._n})",
            )
        ax.set_xlabel("Bit-width b")
        ax.set_ylabel("MSE Distortion D_mse")
        ax.set_title("TurboQuant MSE Distortion vs. Bit-width")
        ax.legend()
        ax.grid(True, which="both", alpha=0.4)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._mse_sum = 0.0
        self._ip_sq_sum = 0.0
        self._n = 0

    def __repr__(self) -> str:
        return f"DistortionObserver(b={self._b}, d={self._d}, n={self._n})"
