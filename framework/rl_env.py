"""
Core RL environment for energy-aware sensing applications.

This module provides a reusable base class for multi-sensor RL environments
with configurable reward functions including risk-aware components.
"""

import random
from typing import Dict, List, Tuple, Any
import numpy as np


class EnergyAwareSensingEnv:
    """
    Base environment for energy-aware multi-sensor RL applications.
    
    This environment models:
    - Battery-constrained operation
    - Multiple sensors with different energy costs
    - Event detection with success/failure outcomes
    - Risk-aware reward functions
    
    State space: (battery_level, time_step, *event_flags)
    Action space: Binary mask indicating which sensors to activate
    Reward: α·detection_success - β·energy_cost - λ·missed_events
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        sensor_costs: List[float],
        alpha: float = 15.0,
        beta: float = 0.008,
        lambda_risk: float = 0.0,
        max_battery: int = 400_000,
        max_time_steps: int = 12_000,
        battery_discretization: int = 10,
        time_discretization: int = 60,
    ):
        """
        Initialize the energy-aware sensing environment.
        
        Args:
            data: List of timestep data with event flags
            sensor_costs: Energy cost per sensor per timestep [mA/timestep]
            alpha: Reward weight for successful detections
            beta: Penalty weight for energy consumption
            lambda_risk: Penalty weight for missed events (risk-aware component)
            max_battery: Maximum battery capacity [mA·timestep units]
            max_time_steps: Maximum episode length
            battery_discretization: Number of discrete battery levels
            time_discretization: Number of discrete time buckets
        """
        self.data = data
        self.sensor_costs = np.array(sensor_costs)
        self.alpha = alpha
        self.beta = beta
        self.lambda_risk = lambda_risk
        self.max_battery = max_battery
        self.max_time_steps = max_time_steps
        self.battery_discretization = battery_discretization
        self.time_discretization = time_discretization
        
        self.n_sensors = len(sensor_costs)
        self.action_space_size = 2 ** self.n_sensors
        
        # Extract event flag names from first data point
        self.event_flags = [k for k in data[0].keys() if k.endswith('_flag')]
        
        self.reset()
    
    def reset(self) -> Tuple:
        """Reset environment to initial state."""
        self.t = 0
        self.battery = self.max_battery
        self.done = False
        
        # Initialize event flags from first timestep
        first_data = self.data[0]
        self.current_events = {flag: int(first_data[flag]) for flag in self.event_flags}
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[Tuple, float, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Integer representing binary sensor activation mask
        
        Returns:
            (next_state, reward, done, info)
        """
        # Decode action to sensor activations
        sensor_activations = self._decode_action(action)
        
        # Calculate energy cost
        energy_cost = np.sum(self.sensor_costs * sensor_activations)
        self.battery = max(0, self.battery - energy_cost)
        
        # Calculate detection success
        detection_success = self._calculate_detection_success(sensor_activations)
        
        # Calculate missed events (for risk-aware reward)
        missed_events = self._calculate_missed_events(sensor_activations)
        
        # Calculate reward
        reward = (self.alpha * detection_success - 
                 self.beta * energy_cost - 
                 self.lambda_risk * missed_events)
        
        # Advance time
        self.t += 1
        
        # Check termination conditions
        if self.t >= self.max_time_steps or self.battery == 0:
            self.done = True
        else:
            # Update event flags for next timestep
            next_data = self.data[self.t]
            self.current_events = {flag: int(next_data[flag]) for flag in self.event_flags}
        
        info = {
            'energy_cost': energy_cost,
            'detection_success': detection_success,
            'missed_events': missed_events,
            'battery_remaining': self.battery
        }
        
        return self._get_state(), reward, self.done, info
    
    def _decode_action(self, action: int) -> np.ndarray:
        """Decode integer action to binary sensor activation array."""
        activations = np.zeros(self.n_sensors, dtype=int)
        for i in range(self.n_sensors):
            activations[i] = (action >> (self.n_sensors - 1 - i)) & 1
        return activations
    
    def _calculate_detection_success(self, sensor_activations: np.ndarray) -> float:
        """Calculate detection success based on active sensors and current events."""
        success = 0
        for i, (flag, active) in enumerate(zip(self.event_flags, sensor_activations)):
            if self.current_events[flag] and active:
                success += 1
        return success
    
    def _calculate_missed_events(self, sensor_activations: np.ndarray) -> float:
        """Calculate number of missed events (events present but sensor inactive)."""
        missed = 0
        for i, (flag, active) in enumerate(zip(self.event_flags, sensor_activations)):
            if self.current_events[flag] and not active:
                missed += 1
        return missed
    
    def _get_state(self) -> Tuple:
        """Get current state representation."""
        # Discretize battery level
        battery_discrete = min(self.battery_discretization, 
                             self.battery * self.battery_discretization // self.max_battery)
        
        # Discretize time (cyclical)
        time_discrete = self.t % self.time_discretization
        
        # Add event flags to state
        event_values = tuple(self.current_events[flag] for flag in self.event_flags)
        
        return (int(battery_discrete), int(time_discrete)) + event_values


class HealthWearableEnv(EnergyAwareSensingEnv):
    """
    Specialized environment for health wearable case study.
    
    Sensors: ECG, PPG, Temperature
    Events: Arrhythmia, Blood pressure anomaly, Fever
    """
    
    def __init__(self, data: List[Dict], **kwargs):
        # Default sensor costs for health wearable [ECG, PPG, Temp] in mA/5s
        default_costs = kwargs.pop('sensor_costs', [10, 4, 1])
        super().__init__(data, sensor_costs=default_costs, **kwargs)
        
        # Ensure we have the expected event flags
        expected_flags = ['arr_flag', 'bp_flag', 'fever_flag']
        assert all(flag in self.event_flags for flag in expected_flags), \
            f"Expected flags {expected_flags}, got {self.event_flags}"
