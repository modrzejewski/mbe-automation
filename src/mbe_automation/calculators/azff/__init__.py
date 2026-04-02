"""
AZFF package initialization.

This package provides a fully ASE compatible implementation of the
AZ FF (AstraZeneca Force Field) for evaluating crystal structures
directly from CIF/ASE Atoms objects using OpenMM.

Only the main calculator class is exported at package level to keep
the public interface clean.
"""

from .calculator import AZFFCalculator

__all__ = ["AZFFCalculator"]
