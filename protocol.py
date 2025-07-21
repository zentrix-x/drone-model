from dataclasses import dataclass
from typing import Tuple, Optional
import random

@dataclass
class RPMCmd:
    t: float
    rpm: Tuple[float, float, float, float]

class MapTask:
    """A class to define the map task (start, goal, horizon, etc.)."""
    
    def __init__(self, start: Tuple[float, float, float], goal: Tuple[float, float, float], horizon: float, sim_dt: float, map_seed: Optional[int] = None):
        self.start = start  # (x, y, z) start position of the drone
        self.goal = goal    # (x, y, z) goal position of the drone
        self.horizon = horizon  # The time limit for the task
        self.sim_dt = sim_dt  # Time step for the simulation
        self.map_seed = map_seed if map_seed is not None else random.randint(1, 10000000000)
