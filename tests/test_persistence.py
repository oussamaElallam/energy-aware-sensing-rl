"""
Tests for persistence logic in RL environment.
Verifies that event flags persist when sensors are OFF (realistic partial observability).
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from framework.rl_env import EnergyAwareSensingEnv, HealthWearableEnv


class TestPersistenceLogic:
    """Test suite for verifying persistence behavior (realistic partial observability)."""
    
    def test_arr_flag_persists_when_ecg_off(self):
        """Verify arr_flag persists when ECG sensor is OFF."""
        # Create data where arr_flag changes from 1 to 0
        data = [
            {'arr_flag': 1, 'bp_flag': 0, 'fever_flag': 0},  # t=0: arrhythmia present
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},  # t=1: arrhythmia gone (ground truth)
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},  # t=2
        ]
        
        env = HealthWearableEnv(data=data, max_time_steps=3)
        state = env.reset()
        
        # Initial state should have arr_flag=1
        assert state[2] == 1, "Initial arr_flag should be 1"
        
        # Take action with ECG OFF (action 0b000 = 0)
        next_state, _, _, _ = env.step(0)
        
        # arr_flag should PERSIST as 1 (not update to ground truth 0)
        assert next_state[2] == 1, "arr_flag should persist when ECG is OFF"
    
    def test_arr_flag_updates_when_ecg_on(self):
        """Verify arr_flag updates when ECG sensor is ON."""
        data = [
            {'arr_flag': 1, 'bp_flag': 0, 'fever_flag': 0},  # t=0
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},  # t=1
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},  # t=2
        ]
        
        env = HealthWearableEnv(data=data, max_time_steps=3)
        env.reset()
        
        # Take action with ECG ON (action 0b100 = 4)
        next_state, _, _, _ = env.step(4)
        
        # arr_flag should UPDATE to ground truth (0)
        assert next_state[2] == 0, "arr_flag should update to 0 when ECG is ON"
    
    def test_bp_flag_persists_when_ppg_off(self):
        """Verify bp_flag persists when PPG sensor is OFF."""
        data = [
            {'arr_flag': 0, 'bp_flag': 1, 'fever_flag': 0},  # t=0
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},  # t=1
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},  # t=2
        ]
        
        env = HealthWearableEnv(data=data, max_time_steps=3)
        state = env.reset()
        
        assert state[3] == 1, "Initial bp_flag should be 1"
        
        # Take action with PPG OFF (action 0b101 = 5, ECG and Temp ON)
        next_state, _, _, _ = env.step(5)
        
        # bp_flag should PERSIST as 1
        assert next_state[3] == 1, "bp_flag should persist when PPG is OFF"
    
    def test_bp_flag_updates_when_ppg_on(self):
        """Verify bp_flag updates when PPG sensor is ON."""
        data = [
            {'arr_flag': 0, 'bp_flag': 1, 'fever_flag': 0},
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},
        ]
        
        env = HealthWearableEnv(data=data, max_time_steps=3)
        env.reset()
        
        # Take action with PPG ON (action 0b010 = 2)
        next_state, _, _, _ = env.step(2)
        
        assert next_state[3] == 0, "bp_flag should update to 0 when PPG is ON"
    
    def test_fever_flag_persists_when_temp_off(self):
        """Verify fever_flag persists when Temp sensor is OFF."""
        data = [
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 1},
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},
        ]
        
        env = HealthWearableEnv(data=data, max_time_steps=3)
        state = env.reset()
        
        assert state[4] == 1, "Initial fever_flag should be 1"
        
        # Take action with Temp OFF (action 0b110 = 6, ECG and PPG ON)
        next_state, _, _, _ = env.step(6)
        
        assert next_state[4] == 1, "fever_flag should persist when Temp is OFF"
    
    def test_fever_flag_updates_when_temp_on(self):
        """Verify fever_flag updates when Temp sensor is ON."""
        data = [
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 1},
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},
        ]
        
        env = HealthWearableEnv(data=data, max_time_steps=3)
        env.reset()
        
        # Take action with Temp ON (action 0b001 = 1)
        next_state, _, _, _ = env.step(1)
        
        assert next_state[4] == 0, "fever_flag should update to 0 when Temp is ON"
    
    def test_all_flags_persist_when_all_sensors_off(self):
        """Verify all flags persist when all sensors are OFF."""
        data = [
            {'arr_flag': 1, 'bp_flag': 1, 'fever_flag': 1},
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},
        ]
        
        env = HealthWearableEnv(data=data, max_time_steps=3)
        state = env.reset()
        
        # All flags should be 1 initially
        assert state[2:5] == (1, 1, 1), "Initial flags should all be 1"
        
        # Take action with all sensors OFF (action 0)
        next_state, _, _, _ = env.step(0)
        
        # All flags should PERSIST
        assert next_state[2:5] == (1, 1, 1), "All flags should persist when all sensors OFF"
    
    def test_all_flags_update_when_all_sensors_on(self):
        """Verify all flags update when all sensors are ON."""
        data = [
            {'arr_flag': 1, 'bp_flag': 1, 'fever_flag': 1},
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},
        ]
        
        env = HealthWearableEnv(data=data, max_time_steps=3)
        env.reset()
        
        # Take action with all sensors ON (action 7 = 0b111)
        next_state, _, _, _ = env.step(7)
        
        # All flags should update to ground truth (all 0)
        assert next_state[2:5] == (0, 0, 0), "All flags should update when all sensors ON"
    
    def test_stale_flag_detection_scenario(self):
        """
        Test realistic scenario: Agent sees stale arrhythmia flag when ECG is OFF,
        then detects new event when ECG turns ON.
        """
        data = [
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},  # t=0: no event
            {'arr_flag': 1, 'bp_flag': 0, 'fever_flag': 0},  # t=1: arrhythmia starts (hidden if ECG OFF)
            {'arr_flag': 1, 'bp_flag': 0, 'fever_flag': 0},  # t=2: arrhythmia continues
            {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},  # t=3: arrhythmia ends
        ]
        
        env = HealthWearableEnv(data=data, max_time_steps=4)
        state = env.reset()
        
        # t=0: ECG OFF - agent starts with arr_flag=0
        assert state[2] == 0
        
        # Step with ECG OFF - arrhythmia at t=1 is HIDDEN
        state, _, _, _ = env.step(0)  # All sensors OFF
        assert state[2] == 0, "Arrhythmia should be HIDDEN when ECG is OFF"
        
        # Step with ECG ON - arrhythmia at t=2 is DETECTED
        state, _, _, _ = env.step(4)  # ECG ON
        assert state[2] == 1, "Arrhythmia should be DETECTED when ECG is ON"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
