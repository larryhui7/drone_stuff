"""Scripted swarm policies, one per file. Add a policy: write <name>.py with a Policy
subclass (see base.py) and register it in POLICIES below; swarm_policies.py --run uses it.
"""
from .base import Policy
from .defend import DefendPolicy
from .formation import FormationPolicy
from .hover import HoverPolicy
from .levels import LevelsPolicy
from .rl import RLPolicy
from .spin import SpinPolicy
from .umbrella import Umbrella

POLICIES = {"spin": SpinPolicy, "levels": LevelsPolicy, "hover": HoverPolicy,
            "formation": FormationPolicy, "rl": RLPolicy,
            "defend": DefendPolicy}

__all__ = ["POLICIES", "Policy", "DefendPolicy", "FormationPolicy", "HoverPolicy", "LevelsPolicy", "RLPolicy", "SpinPolicy",
           "Umbrella"]
