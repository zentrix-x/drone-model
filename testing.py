import unittest
from unittest.mock import MagicMock
from flying_strategy import flying_strategy
from protocol import RPMCmd, MapTask

class TestFlyingStrategy(unittest.TestCase):

    def test_waypoint_navigation(self):
        mock_task = MagicMock(spec=MapTask)
        mock_task.start = (0, 0, 0)
        mock_task.goal = (10, 10, 5)
        mock_task.horizon = 10
        mock_task.sim_dt = 0.05
        mock_task.map_seed = None

        rpm_cmds = flying_strategy(mock_task, gui=False)

        self.assertGreater(len(rpm_cmds), 0)
        self.assertTrue(all(isinstance(cmd, RPMCmd) for cmd in rpm_cmds))

if __name__ == '__main__':
    unittest.main()
