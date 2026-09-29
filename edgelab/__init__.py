"""Edge Lab — event studies for trade ideas."""

__version__ = "0.1.0"

from .spec import Spec, Zone, Trigger, Scoring, Stop, Target, EXAMPLE_SPEC
from .library import Library

__all__ = ["Spec", "Zone", "Trigger", "Scoring", "Stop", "Target",
           "EXAMPLE_SPEC", "Library", "__version__"]
