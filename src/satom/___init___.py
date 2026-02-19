"""
SAtom – Sensitivity Analysis using the Kolmogorov-Smirnov 2-sample test.

Usage
-----
>>> from satom import SAtom
>>> KS2_mean, KS2 = SAtom(X, Y, J=2000, seed=42)
"""

from satom._core import SAtom

__all__ = ["SAtom"]
__version__ = "1.0.0"