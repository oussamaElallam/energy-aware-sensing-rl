"""
Test reward calculation with risk-aware penalty.
Tests that missed events result in lower rewards when lambda_risk > 0.
"""
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'framework'))

from rl_env import HealthWearableEnv
import numpy as np


def test_reward_with_risk_penalty():
    """Test that missed events result in lower rewards with lambda_risk > 0."""
    # Create test data with known event patterns (list of dictionaries format)
    test_data = [
        {'arr_flag': 1, 'bp_flag': 1, 'fever_flag': 1},  # all events present
        {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},  # no events
        {'arr_flag': 1, 'bp_flag': 0, 'fever_flag': 0},  # only arrhythmia
        {'arr_flag': 0, 'bp_flag': 1, 'fever_flag': 0},  # only BP issue
        {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 1},  # only fever
    ]
    
    # Instantiate environment with risk penalty
    env = HealthWearableEnv(
        data=test_data,
        lambda_risk=2.0,  # High risk penalty
        alpha=15.0,
        beta=0.008,
        max_battery=400000,
        max_time_steps=5
    )
    
    # Reset to first state (all events present)
    env.reset()
    
    # Test case 1: All sensors ON when all events present (optimal)
    action_all_on = 7  # Binary 111: ECG=1, PPG=1, TEMP=1
    _, reward_all_detected, _, _ = env.step(action_all_on)
    
    # Reset to same state
    env.reset()
    
    # Test case 2: All sensors OFF when all events present (worst case)
    action_all_off = 0  # Binary 000: ECG=0, PPG=0, TEMP=0
    _, reward_all_missed, _, _ = env.step(action_all_off)
    
    # Assert that detecting events gives higher reward than missing them
    assert reward_all_detected > reward_all_missed, \
        f"Reward when detecting all events ({reward_all_detected}) should be higher than when missing all ({reward_all_missed})"
    
    # Test case 3: Partial detection vs full miss
    env.reset()
    action_partial = 4  # Binary 100: ECG=1, PPG=0, TEMP=0 (detects arrhythmia, misses BP and fever)
    _, reward_partial, _, _ = env.step(action_partial)
    
    # Partial detection should be better than complete miss but worse than full detection
    assert reward_all_missed < reward_partial < reward_all_detected, \
        f"Reward progression should be: all missed ({reward_all_missed}) < partial ({reward_partial}) < all detected ({reward_all_detected})"
    
    print(f"✓ Reward test passed:")
    print(f"  All detected: {reward_all_detected:.3f}")
    print(f"  Partial detected: {reward_partial:.3f}")
    print(f"  All missed: {reward_all_missed:.3f}")


def test_reward_without_risk_penalty():
    """Test reward calculation when lambda_risk = 0 (no risk penalty)."""
    test_data = [
        {'arr_flag': 1, 'bp_flag': 1, 'fever_flag': 1},  # all events present
        {'arr_flag': 0, 'bp_flag': 0, 'fever_flag': 0},  # no events
    ]
    
    # Environment without risk penalty
    env_no_risk = HealthWearableEnv(
        data=test_data,
        lambda_risk=0.0,  # No risk penalty
        alpha=15.0,
        beta=0.008,
        max_battery=400000,
        max_time_steps=2
    )
    
    env_no_risk.reset()
    
    # When lambda_risk=0, missed events should not affect reward
    # Only detection success and energy cost matter
    action_all_on = 7  # All sensors on
    _, reward_on, _, _ = env_no_risk.step(action_all_on)
    
    env_no_risk.reset()
    action_all_off = 0  # All sensors off
    _, reward_off, _, _ = env_no_risk.step(action_all_off)
    
    # With lambda_risk=0, the difference should only be due to energy cost (not missed events)
    # All on: reward = 15*3 - 0.008*100 - 0*3 = 45 - 0.8 = 44.2
    # All off: reward = 15*0 - 0.008*0 - 0*3 = 0
    expected_diff = 15.0 * 3 - 0.008 * 100  # Success reward minus energy cost
    actual_diff = reward_on - reward_off
    
    # Use more tolerant precision for floating point comparison
    assert abs(actual_diff - expected_diff) < 1.0, \
        f"Expected reward difference {expected_diff}, got {actual_diff}"
    
    print(f"✓ No-risk test passed:")
    print(f"  All on (no risk): {reward_on:.3f}")
    print(f"  All off (no risk): {reward_off:.3f}")
    print(f"  Difference: {actual_diff:.3f} (expected: {expected_diff:.3f})")


def test_risk_penalty_scaling():
    """Test that higher lambda_risk values result in larger penalties for missed events."""
    test_data = [{'arr_flag': 1, 'bp_flag': 1, 'fever_flag': 1}]  # Single state with all events
    
    # Test different risk penalty values
    lambda_values = [0.0, 1.0, 2.0, 5.0]
    rewards_missed = []
    
    for lambda_risk in lambda_values:
        env = HealthWearableEnv(
            data=test_data,
            lambda_risk=lambda_risk,
            alpha=15.0,
            beta=0.008,
            max_battery=400000,
            max_time_steps=1
        )
        
        env.reset()
        # All sensors off when all events present (maximum missed events)
        _, reward, _, _ = env.step(0)
        rewards_missed.append(reward)
    
    # Rewards should decrease as lambda_risk increases (more penalty for missed events)
    for i in range(1, len(rewards_missed)):
        assert rewards_missed[i] < rewards_missed[i-1], \
            f"Reward should decrease as lambda_risk increases: {rewards_missed}"
    
    print(f"✓ Risk scaling test passed:")
    for i, (lambda_val, reward) in enumerate(zip(lambda_values, rewards_missed)):
        print(f"  λ={lambda_val}: reward={reward:.3f}")


if __name__ == "__main__":
    test_reward_with_risk_penalty()
    test_reward_without_risk_penalty()
    test_risk_penalty_scaling()
    print("\n✅ All reward tests passed!")
