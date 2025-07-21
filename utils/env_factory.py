import pybullet as p
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

def make_env(task, gui=False):
    """
    Function to create the environment for simulation.
    We are removing the 'raw_rpm' argument since HoverAviary may not support it.
    """
    # Create the environment using the HoverAviary class from gym_pybullet_drones
    env = HoverAviary(gui=gui)

    # Return the environment and associated PyBullet client
    return env
