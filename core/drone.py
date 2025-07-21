import numpy as np
import pybullet as p

def track_drone(cli, drone_id) -> None:
    """Keep the PyBullet spectator camera locked on the drone."""
    pos, _ = p.getBasePositionAndOrientation(drone_id, physicsClientId=cli)
    tgt = np.add(pos, [0.0, 0.0, 0.4])  # Look ~0.4m above CG
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0, cameraPitch=-25, cameraTargetPosition=tgt, physicsClientId=cli)
