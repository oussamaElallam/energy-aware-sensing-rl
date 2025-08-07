"""
Tests for the RL environment framework.
"""

import pytest
import numpy as np
from framework.rl_env import EnergyAwareSensingEnv, HealthWearableEnv


def create_test_data(steps=100):
    """Create synthetic test data."""
    np.random.seed(42)
    return [
        {
            'arr_flag': np.random.choice([0, 1], p=[0.7, 0.3]),
            'bp_flag': np.random.choice([0, 1], p=[0.6, 0.4]),
            'fever_flag': np.random.choice([0, 1], p=[0.8, 0.2])
        }
        for _ in range(steps)
    ]


class TestEnergyAwareSensingEnv:
    
    def test_initialization(self):
        """Test environment initialization."""
        data = create_test_data(10)
        env = EnergyAwareSensingEnv(
            data=data,
            sensor_costs=[10, 4, 1],
            alpha=15.0,
            beta=0.008,
            lambda_risk=0.5
        )
        
        assert env.n_sensors == 3
        assert env.action_space_size == 8
        assert env.lambda_risk == 0.5
        assert len(env.event_flags) == 3
    
    def test_reset(self):
        """Test environment reset."""
        data = create_test_data(10)
        env = EnergyAwareSensingEnv(data=data, sensor_costs=[10, 4, 1])
        
        state = env.reset()
        
        assert env.t == 0
        assert env.battery == env.max_battery
        assert not env.done
        assert len(state) == 5  # battery, time, 3 event flags
    
    def test_action_decoding(self):
        """Test action decoding to sensor activations."""
        data = create_test_data(10)
        env = EnergyAwareSensingEnv(data=data, sensor_costs=[10, 4, 1])
        
        # Test action 0 (000) - no sensors
        activations = env._decode_action(0)
        assert np.array_equal(activations, [0, 0, 0])
        
        # Test action 7 (111) - all sensors
        activations = env._decode_action(7)
        assert np.array_equal(activations, [1, 1, 1])
        
        # Test action 5 (101) - first and third sensors
        activations = env._decode_action(5)
        assert np.array_equal(activations, [1, 0, 1])
    
    def test_step_basic(self):
        """Test basic step functionality."""
        data = create_test_data(10)
        env = EnergyAwareSensingEnv(data=data, sensor_costs=[10, 4, 1])
        
        env.reset()
        initial_battery = env.battery
        
        # Take action 7 (all sensors on)
        next_state, reward, done, info = env.step(7)
        
        # Check battery decreased
        assert env.battery < initial_battery
        assert env.battery == initial_battery - 15  # 10+4+1
        
        # Check time advanced
        assert env.t == 1
        
        # Check info dict
        assert 'energy_cost' in info
        assert 'detection_success' in info
        assert 'missed_events' in info
        assert info['energy_cost'] == 15
    
    def test_risk_aware_reward(self):
        """Test risk-aware reward calculation."""
        # Create data with known event pattern
        data = [
            {'arr_flag': 1, 'bp_flag': 0, 'fever_flag': 1},  # arr and fever events
            {'arr_flag': 0, 'bp_flag': 1, 'fever_flag': 0}   # bp event
        ]
        
        env = EnergyAwareSensingEnv(
            data=data, 
            sensor_costs=[10, 4, 1],
            alpha=15.0,
            beta=0.008,
            lambda_risk=2.0
        )
        
        env.reset()
        
        # Action 0: no sensors active (should miss arr and fever events)
        _, reward, _, info = env.step(0)
        
        expected_reward = 15.0 * 0 - 0.008 * 0 - 2.0 * 2  # miss 2 events
        assert abs(reward - expected_reward) < 1e-6
        assert info['missed_events'] == 2
        assert info['detection_success'] == 0
    
    def test_termination_conditions(self):
        """Test episode termination conditions."""
        data = create_test_data(5)  # Short episode
        env = EnergyAwareSensingEnv(
            data=data, 
            sensor_costs=[10, 4, 1],
            max_time_steps=3
        )
        
        env.reset()
        
        # Should not be done initially
        assert not env.done
        
        # Take steps until termination
        for i in range(3):
            _, _, done, _ = env.step(0)
            if i < 2:
                assert not done
            else:
                assert done


class TestHealthWearableEnv:
    
    def test_health_wearable_initialization(self):
        """Test health wearable specific initialization."""
        data = create_test_data(10)
        env = HealthWearableEnv(data=data)
        
        # Should use default sensor costs
        assert np.array_equal(env.sensor_costs, [10, 4, 1])
        
        # Should have expected event flags
        expected_flags = ['arr_flag', 'bp_flag', 'fever_flag']
        assert all(flag in env.event_flags for flag in expected_flags)
    
    def test_custom_sensor_costs(self):
        """Test custom sensor costs for health wearable."""
        data = create_test_data(10)
        custom_costs = [15, 6, 2]
        env = HealthWearableEnv(data=data, sensor_costs=custom_costs)
        
        assert np.array_equal(env.sensor_costs, custom_costs)


if __name__ == "__main__":
    pytest.main([__file__])
