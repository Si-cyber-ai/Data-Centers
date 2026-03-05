"""
Digital Twin Thermal Environment for Data Center Simulation

This module implements a Gymnasium-compatible environment for simulating
data center thermal dynamics with reinforcement learning.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import yaml

from simulator.heat_transfer_model import HeatTransferModel


class DataCenterThermalEnv(gym.Env):
    """
    Data Center Thermal Environment for RL training.
    
    Simulates a 2D grid of server racks with thermal dynamics,
    workload variations, and cooling control.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        workload_generator=None
    ):
        """
        Initialize the data center thermal environment.
        
        Args:
            config_path: Path to configuration file
            workload_generator: Optional workload generator instance
        """
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Grid dimensions
        self.rows, self.cols = self.config['simulation']['grid_size']
        self.num_racks = self.rows * self.cols
        
        # Simulation parameters
        self.timestep = self.config['simulation']['timestep']
        self.max_steps = self.config['simulation']['max_steps']
        self.ambient_temp = self.config['simulation']['ambient_temperature']
        
        # Safety parameters
        self.max_temp = self.config['safety']['max_temperature']
        self.min_temp = self.config['safety']['min_temperature']
        self.critical_temp = self.config['safety']['critical_temperature']
        self.max_cooling_change = self.config['safety']['max_cooling_change']
        
        # Initialize heat transfer model
        self.heat_model = HeatTransferModel(
            grid_size=(self.rows, self.cols),
            alpha=self.config['simulation']['alpha'],
            beta=self.config['simulation']['beta'],
            gamma=self.config['simulation']['gamma'],
            delta=self.config['simulation']['delta'],
            noise_std=self.config['simulation']['noise_std']
        )
        
        # Workload generator
        self.workload_generator = workload_generator
        
        # Define action space (discrete cooling levels)
        # Actions: 0=decrease cooling, 1=maintain, 2=increase low, 3=increase med, 4=increase high
        self.action_space = spaces.Discrete(self.config['rl']['action_dim'])
        
        # Define observation space
        # State includes: temperatures, cpu_workload, cooling_levels, ambient_temp
        state_size = self.num_racks * 3 + 1  # temp + workload + cooling + ambient
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_size,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.temperatures: Optional[np.ndarray] = None
        self.cpu_workload: Optional[np.ndarray] = None
        self.cooling_levels: Optional[np.ndarray] = None
        self.current_step: int = 0
        
        # History for tracking
        self.temperature_history = []
        self.cooling_history = []
        self.workload_history = []
        self.reward_history = []
        self.violation_count = 0
        
        # Previous cooling for rate limiting
        self.prev_cooling_levels: Optional[np.ndarray] = None
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Initialize temperatures near ambient
        self.temperatures = np.random.uniform(
            self.ambient_temp + 10,
            self.ambient_temp + 20,
            size=(self.rows, self.cols)
        )
        
        # Initialize CPU workload
        if self.workload_generator is not None:
            self.cpu_workload = self.workload_generator.generate(0)
        else:
            self.cpu_workload = np.random.uniform(0.2, 0.5, size=(self.rows, self.cols))
        
        # Initialize cooling at moderate level
        self.cooling_levels = np.full((self.rows, self.cols), 0.5)
        self.prev_cooling_levels = self.cooling_levels.copy()
        
        # Reset counters
        self.current_step = 0
        self.violation_count = 0
        
        # Clear history
        self.temperature_history = [self.temperatures.copy()]
        self.cooling_history = [self.cooling_levels.copy()]
        self.workload_history = [self.cpu_workload.copy()]
        self.reward_history = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Cooling action to take
            
        Returns:
            observation: New state
            reward: Reward for this transition
            terminated: Whether episode has ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Map action to cooling adjustment
        cooling_change = self._action_to_cooling_change(action)
        
        # Apply cooling change with rate limiting
        new_cooling = np.clip(
            self.cooling_levels + cooling_change,
            0.0,
            1.0
        )
        
        # Enforce maximum cooling change rate (safety constraint)
        cooling_delta = new_cooling - self.prev_cooling_levels
        cooling_delta = np.clip(
            cooling_delta,
            -self.max_cooling_change,
            self.max_cooling_change
        )
        self.cooling_levels = self.prev_cooling_levels + cooling_delta
        self.prev_cooling_levels = self.cooling_levels.copy()
        
        # Update CPU workload
        if self.workload_generator is not None:
            self.cpu_workload = self.workload_generator.generate(self.current_step)
        else:
            # Random walk workload
            workload_change = np.random.uniform(-0.05, 0.05, size=(self.rows, self.cols))
            self.cpu_workload = np.clip(self.cpu_workload + workload_change, 0.0, 1.0)
        
        # Update temperatures using heat transfer model
        self.temperatures = self.heat_model.update_temperatures(
            self.temperatures,
            self.cpu_workload,
            self.cooling_levels,
            self.ambient_temp,
            dt=1.0
        )
        
        # Check for safety violations
        violations = np.sum(self.temperatures > self.max_temp)
        critical_violations = np.sum(self.temperatures > self.critical_temp)
        self.violation_count += violations
        
        # Compute reward
        reward = self._compute_reward(violations, critical_violations)
        
        # Update history
        self.temperature_history.append(self.temperatures.copy())
        self.cooling_history.append(self.cooling_levels.copy())
        self.workload_history.append(self.cpu_workload.copy())
        self.reward_history.append(reward)
        
        # Check termination conditions
        self.current_step += 1
        terminated = critical_violations > 0  # Terminate if critical violation
        truncated = self.current_step >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _action_to_cooling_change(self, action: int) -> np.ndarray:
        """
        Convert discrete action to cooling level change.
        
        Args:
            action: Discrete action index
            
        Returns:
            Cooling change matrix
        """
        # Action mapping:
        # 0: Decrease cooling globally
        # 1: Maintain current cooling
        # 2: Increase cooling slightly
        # 3: Increase cooling moderately
        # 4: Increase cooling significantly
        
        action_map = {
            0: -0.1,   # Decrease
            1: 0.0,    # Maintain
            2: 0.05,   # Slight increase
            3: 0.1,    # Moderate increase
            4: 0.2     # Significant increase
        }
        
        change = action_map.get(action, 0.0)
        
        # Apply change more aggressively to hotter racks
        temperature_normalized = (self.temperatures - self.ambient_temp) / 50.0
        temperature_weight = np.clip(temperature_normalized, 0.5, 1.5)
        
        cooling_change = change * temperature_weight
        
        return cooling_change
    
    def _compute_reward(self, violations: int, critical_violations: int) -> float:
        """
        Compute reward based on energy efficiency and safety.
        
        Args:
            violations: Number of temperature violations
            critical_violations: Number of critical violations
            
        Returns:
            Reward value
        """
        # Energy consumption penalty (proportional to cooling effort)
        energy_cost = np.mean(self.cooling_levels) * self.config['reward']['energy_weight']
        
        # Temperature violation penalty
        violation_penalty = (
            violations * self.config['reward']['temperature_violation_weight']
            + critical_violations * self.config['reward']['temperature_violation_weight'] * 5
        )
        
        # Temperature stability reward (prefer temperatures near optimal)
        optimal_temp = self.config['reward']['comfort_zone_temp']
        temp_deviation = np.mean(np.abs(self.temperatures - optimal_temp))
        stability_penalty = temp_deviation * self.config['reward']['stability_weight']
        
        # Cooling instability penalty (avoid rapid changes)
        if len(self.cooling_history) > 1:
            cooling_change = np.mean(np.abs(
                self.cooling_history[-1] - self.cooling_history[-2]
            ))
            instability_penalty = cooling_change * 50.0
        else:
            instability_penalty = 0.0
        
        # Total reward
        reward = -(energy_cost + violation_penalty + stability_penalty + instability_penalty)
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector from current state.
        
        Returns:
            Flattened observation array
        """
        temp_flat = self.temperatures.flatten()
        workload_flat = self.cpu_workload.flatten()
        cooling_flat = self.cooling_levels.flatten()
        ambient = np.array([self.ambient_temp])
        
        observation = np.concatenate([
            temp_flat,
            workload_flat,
            cooling_flat,
            ambient
        ]).astype(np.float32)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional environment information.
        
        Returns:
            Info dictionary
        """
        return {
            'step': self.current_step,
            'avg_temperature': np.mean(self.temperatures),
            'max_temperature': np.max(self.temperatures),
            'min_temperature': np.min(self.temperatures),
            'avg_cooling': np.mean(self.cooling_levels),
            'avg_workload': np.mean(self.cpu_workload),
            'violations': self.violation_count,
            'hotspots': np.sum(self.temperatures > self.max_temp)
        }
    
    def get_state_grid(self) -> Dict[str, np.ndarray]:
        """
        Get current state as 2D grids for visualization.
        
        Returns:
            Dictionary of state grids
        """
        return {
            'temperatures': self.temperatures.copy(),
            'cpu_workload': self.cpu_workload.copy(),
            'cooling_levels': self.cooling_levels.copy()
        }
    
    def apply_safety_override(self) -> None:
        """
        Apply emergency cooling if temperatures exceed safety threshold.
        """
        danger_mask = self.temperatures > self.config['safety']['temperature_warning']
        if np.any(danger_mask):
            # Force maximum cooling on overheating racks
            self.cooling_levels[danger_mask] = 1.0
    
    def render(self, mode: str = 'human'):
        """
        Render the environment state.
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Temperatures:\n{self.temperatures}")
            print(f"CPU Workload:\n{self.cpu_workload}")
            print(f"Cooling Levels:\n{self.cooling_levels}")
            print(f"Avg Temp: {np.mean(self.temperatures):.2f}°C")
            print(f"Max Temp: {np.max(self.temperatures):.2f}°C")
