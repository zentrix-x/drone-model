import math
import random
from typing import Optional, Tuple

import pybullet as p

from swarm.constants import WORLD_RANGE, HEIGHT_SCALE, N_OBSTACLES

SAFE_ZONE_RADIUS = 2.0
MAX_ATTEMPTS_PER_OBS = 10

def _add_box(cli: int, pos, size, yaw) -> None:
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[s / 2 for s in size], physicsClientId=cli
    )
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[s / 2 for s in size],
        rgbaColor=[0.2, 0.6, 0.8, 1.0],
        physicsClientId=cli,
    )
    quat = p.getQuaternionFromEuler([0, 0, yaw])
    p.createMultiBody(
        0,
        col,
        vis,
        basePosition=pos,
        baseOrientation=quat,
        physicsClientId=cli,
    )

def build_world(
    seed: int,
    cli: int,
    *,
    start: Optional[Tuple[float, float, float]] = None,
    goal: Optional[Tuple[float, float, float]] = None,
) -> None:
    if seed is None:
        seed = random.randint(1, 10000000000)
    
    rng = random.Random(seed)

    sx, sy = (start[0], start[1]) if start is not None else (None, None)
    gx, gy = (goal[0], goal[1]) if goal is not None else (None, None)

    placed = 0
    placed_obstacles = []
    MIN_OBSTACLE_DISTANCE = 0.6
    
    while placed < N_OBSTACLES:
        for _ in range(MAX_ATTEMPTS_PER_OBS):
            kind = rng.choice(["wall", "pillar", "box"])
            x = rng.uniform(-WORLD_RANGE, WORLD_RANGE)
            y = rng.uniform(-WORLD_RANGE, WORLD_RANGE)
            yaw = rng.uniform(0, math.pi)

            # Obstacle size and radius calculation
            if kind == "box":
                sx_len, sy_len, sz_len = (rng.uniform(1, 4) for _ in range(3))
                sz_len *= HEIGHT_SCALE
                obj_r = math.hypot(sx_len / 2, sy_len / 2)
            elif kind == "wall":
                length = rng.uniform(5, 15)
                height = rng.uniform(2, 5) * HEIGHT_SCALE
                sx_len, sy_len, sz_len = length, 0.3, height
                obj_r = length / 2.0
            else:  # pillar
                r = rng.uniform(0.3, 1.0)
                h = rng.uniform(2, 7) * HEIGHT_SCALE
                sx_len = sy_len = r * 2
                sz_len = h
                obj_r = r

            # Safe zone checks for obstacles
            def _violates(cx, cy):
                if cx is None:
                    return False
                required_clearance = obj_r + SAFE_ZONE_RADIUS + 0.5
                return math.hypot(x - cx, y - cy) < required_clearance

            if _violates(sx, sy) or _violates(gx, gy):
                continue

            # Obstacle placement logic
            if kind == "box":
                _add_box(cli, [x, y, sz_len / 2], [sx_len, sy_len, sz_len], yaw)
            elif kind == "wall":
                col = p.createCollisionShape(p.GEOM_BOX, sx_len / 2, sy_len / 2, sz_len / 2, physicsClientId=cli)
                vis = p.createVisualShape(p.GEOM_BOX, sx_len / 2, sy_len / 2, sz_len / 2, rgbaColor=[0.9, 0.8, 0.1], physicsClientId=cli)
                quat = p.getQuaternionFromEuler([0, 0, yaw])
                p.createMultiBody(0, col, vis, [x, y, sz_len / 2], quat, physicsClientId=cli)
            else:  # pillar
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=obj_r, height=sz_len, physicsClientId=cli)
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=obj_r, length=sz_len, rgbaColor=[0.8, 0.2, 0.2], physicsClientId=cli)
                p.createMultiBody(0, col, vis, [x, y, sz_len / 2], physicsClientId=cli)

            placed_obstacles.append((x, y, obj_r))
            placed += 1
            break
        else:
            if placed < N_OBSTACLES * 0.7:
                MIN_OBSTACLE_DISTANCE = max(0.8, MIN_OBSTACLE_DISTANCE - 0.1)
            break
