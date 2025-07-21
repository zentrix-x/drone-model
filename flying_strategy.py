from __future__ import annotations
import time
from typing import List, Sequence, Tuple
import numpy as np
from utils.gui_isolation import run_isolated
from utils.env_factory import make_env
from protocol import MapTask, RPMCmd
from core.drone import track_drone
from core.env_builder import build_world  # Import build_world function
from swarm.constants import (
    SAFE_Z,
    GOAL_TOL,
    HOVER_SEC,
    CAM_HZ,
    N_OBSTACLES,
    WORLD_RANGE
)
import random


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def compute(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative


# ---------- Energy-efficient Control for Dynamic Trajectory -------------------
def mpc_control(prev_pos: np.ndarray, target: np.ndarray, pid_x, pid_y, pid_z, dt) -> Tuple[np.ndarray, float, float]:
    error_x = target[0] - prev_pos[0]
    error_y = target[1] - prev_pos[1]
    error_z = target[2] - prev_pos[2]

    # Simple linear model for thrust (thrust proportional to distance to goal)
    thrust_x = error_x * pid_x.Kp
    thrust_y = error_y * pid_y.Kp
    thrust_z = error_z * pid_z.Kp
    
    total_thrust = thrust_x + thrust_y + thrust_z

    # Calculate a simplified energy consumption model (based on thrust)
    energy_penalty = np.abs(total_thrust) ** 2  # Penalty increases with thrust (simplified model)
    
    return np.array([thrust_x, thrust_y, thrust_z, total_thrust]), energy_penalty, total_thrust


# ---------- Public API ---------------------------------------------------
def flying_strategy(task: MapTask, *, gui: bool = False) -> List[RPMCmd]:
    return run_isolated(_flying_strategy_impl, task, gui=gui)


# ---------- Dynamic Waypoints Implementation --------------------------------
def dynamic_waypoints(prev_pos: np.ndarray, target: np.ndarray, safe_z: float) -> List[np.ndarray]:
    """
    Dynamically adjust waypoints based on current position and goal.
    The drone will generate intermediate waypoints if it is far from the target.
    """
    distance_to_goal = np.linalg.norm(target - prev_pos)

    if distance_to_goal > 15:  
        # Reduce number of intermediate waypoints
        mid_point = (target + prev_pos) / 3  # Space out waypoints dynamically
        return [prev_pos, mid_point, target]
    else:
        return [prev_pos, target]


# ---------- Implementation ----------------------------------------------
def _flying_strategy_impl(task: MapTask, *, gui: bool = False) -> List[RPMCmd]:
    # Replace PID controller with energy-efficient MPC-based control
    pid_x = PIDController(Kp=2.5, Ki=0.1, Kd=0.7)  
    pid_y = PIDController(Kp=2.5, Ki=0.1, Kd=0.7)
    pid_z = PIDController(Kp=3.0, Ki=0.3, Kd=1.0)  
    
    pid_x.setpoint = task.goal[0]
    pid_y.setpoint = task.goal[1]
    pid_z.setpoint = task.goal[2]

    env = make_env(task, gui=gui)
    cli = env.getPyBulletClient()

    build_world(seed=task.map_seed, cli=cli, start=task.start, goal=task.goal)

    start_xyz = np.array(task.start, dtype=float)
    gx, gy, gz = task.goal
    safe_z = max(SAFE_Z, start_xyz[2], gz)

    # Get dynamic waypoints
    wps = dynamic_waypoints(start_xyz, task.goal, safe_z)
    wp_idx = 0

    t_sim = 0.0
    hover_elapsed = 0.0
    rpm_log: List[RPMCmd] = []
    total_energy = 0
    prev_pos = np.array([0.0, 0.0, 0.0])

    while t_sim < task.horizon:
        target = wps[wp_idx]
        
        # Generate RPM commands using MPC and energy-efficient control
        target_rpm, energy_penalty, total_thrust = mpc_control(prev_pos, target, pid_x, pid_y, pid_z, task.sim_dt)

        # Step in the environment
        obs, *_ = env.step(target_rpm.reshape(1, 4))  # Pass the action with 4 RPM values
        pos = obs[0, :3]

        _record_cmd(rpm_log, env.last_clipped_action[0], t_sim)

        dist = np.linalg.norm(pos - target)
        if wp_idx < len(wps) - 1:
            if dist < GOAL_TOL:
                wp_idx += 1
        else:
            # Goal reached condition with tighter tolerance
            if dist < HOVER_SEC * GOAL_TOL:  # Reduced tolerance to stop earlier
                hover_elapsed += task.sim_dt
                if hover_elapsed >= HOVER_SEC:  # Minimized hover time
                    break
            else:
                hover_elapsed = 0.0

        total_energy += np.sum(np.abs(target_rpm)) + energy_penalty  # Include energy penalty
        t_sim += task.sim_dt
        prev_pos = pos

    if not gui:
        env.close()

    score = _evaluate_flight_plan_with_energy(rpm_log, total_energy, task.horizon, total_thrust)
    print(f"Final Score: {score}")
    return rpm_log

def _record_cmd(buffer: List[RPMCmd], rpm_vec: Sequence[float], t: float) -> None:
    rpm_tuple: Tuple[float, float, float, float] = tuple(float(x) for x in rpm_vec)
    buffer.append(RPMCmd(t=t, rpm=rpm_tuple))


def _evaluate_flight_plan_with_energy(rpm_log: List[RPMCmd], total_energy: float, horizon: float, total_thrust: float) -> float:
    success = 0
    time_score = max(0, 1 - len(rpm_log) / horizon)
    
    # Update energy penalty based on the actual total thrust
    energy_penalty = np.abs(total_thrust) ** 1.2  # Using thrust penalty to calculate the energy score
    energy_score = 1 - (energy_penalty / total_energy)

    last_cmd = rpm_log[-1] if rpm_log else None
    if last_cmd and np.linalg.norm(np.array(last_cmd.rpm)[:3]) < GOAL_TOL:
        success = 1

    score = (0.70 * success) + (0.15 * time_score) + (0.15 * energy_score)
    return score
