import unittest
from unittest.mock import MagicMock
from flying_strategy import flying_strategy
from protocol import RPMCmd

class TestFlyingStrategy(unittest.TestCase):

    def test_waypoint_navigation(self):
        # Mocking the task object
        mock_task = MagicMock()
        mock_task.start = (0, 0, 0)
        mock_task.goal = (10, 10, 5)
        mock_task.horizon = 10
        mock_task.sim_dt = 0.05

        # Assuming flying_strategy is correctly imported
        rpm_cmds = flying_strategy(mock_task, gui=False)

        # Test that the RPM commands are generated
        self.assertGreater(len(rpm_cmds), 0)
        self.assertTrue(all(isinstance(cmd, RPMCmd) for cmd in rpm_cmds))

if __name__ == '__main__':
    unittest.main()
