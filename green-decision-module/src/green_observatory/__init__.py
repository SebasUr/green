"""Green Window Observatory (V1.0).

Carbon temporal intelligence for *when-to-run* decisions on the French grid.

Milestones 0-2 (this codebase so far) cover the **carbon track**: the
ODRE/eCO2mix provider, the baseline ladder (persistence, climatology, corrected
climatology), low-carbon windows and the project forecast model with its
evaluation against an oracle. The CERN CDC **facility track** (Milestones 3-4)
and the **simulation** / **API** layers are scaffolded but not yet implemented.

See ``IMPLEMENTATION_PLAN_V1.md`` for the full plan and scope boundaries.
"""

from __future__ import annotations

from green_observatory.models import (
    CarbonBasis,
    CarbonForecast,
    CarbonSignal,
    DataQualityReport,
    FacilitySignal,
    GenerationMix,
    GreenWindow,
    ModelName,
    PolicyName,
    SeriesQuality,
    SimulationResult,
    WindowType,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CarbonBasis",
    "CarbonForecast",
    "CarbonSignal",
    "DataQualityReport",
    "FacilitySignal",
    "GenerationMix",
    "GreenWindow",
    "ModelName",
    "PolicyName",
    "SeriesQuality",
    "SimulationResult",
    "WindowType",
]
