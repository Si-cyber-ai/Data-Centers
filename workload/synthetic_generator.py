"""
Synthetic Workload Generator for Data Center Simulation

Generates realistic CPU workload patterns for server racks.
"""

import numpy as np
from typing import Tuple, Optional


class SyntheticWorkloadGenerator:
    """
    Generates synthetic CPU workload patterns for data center racks.
    
    Supports multiple workload patterns:
    - Sinusoidal (daily/weekly cycles)
    - Random spikes
    - Burst patterns
    - Mixed patterns
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int],
        pattern: str = "mixed",
        base_load: float = 0.3,
        peak_load: float = 0.9,
        seed: Optional[int] = None
    ):
        """
        Initialize workload generator.
        
        Args:
            grid_size: (rows, cols) dimensions of rack grid
            pattern: Workload pattern type
            base_load: Minimum workload level [0, 1]
            peak_load: Maximum workload level [0, 1]
            seed: Random seed for reproducibility
        """
        self.rows, self.cols = grid_size
        self.pattern = pattern
        self.base_load = base_load
        self.peak_load = peak_load
        
        if seed is not None:
            np.random.seed(seed)
        
        # Internal state for temporal patterns
        self.time_step = 0
        self.spike_probability = 0.02  # 2% chance of spike per step
        self.burst_duration = 0
        self.burst_remaining = 0
        
        # Per-rack workload bias (some racks naturally busier)
        self.rack_bias = np.random.uniform(0.8, 1.2, size=(self.rows, self.cols))
        
    def generate(self, step: Optional[int] = None) -> np.ndarray:
        """
        Generate workload for current time step.
        
        Args:
            step: Optional time step (if None, uses internal counter)
            
        Returns:
            CPU workload grid [rows, cols] in range [0, 1]
        """
        if step is not None:
            self.time_step = step
        
        if self.pattern == "sinusoidal":
            workload = self._generate_sinusoidal()
        elif self.pattern == "spikes":
            workload = self._generate_spikes()
        elif self.pattern == "burst":
            workload = self._generate_burst()
        elif self.pattern == "mixed":
            workload = self._generate_mixed()
        else:
            workload = self._generate_random()
        
        self.time_step += 1
        return workload
    
    def _generate_sinusoidal(self) -> np.ndarray:
        """
        Generate sinusoidal workload pattern (daily cycles).
        
        Simulates typical diurnal patterns in data centers.
        """
        # Daily cycle (24-hour period, assuming 60s timesteps -> 1440 steps/day)
        period = 1440
        phase = 2 * np.pi * self.time_step / period
        
        # Base sinusoidal pattern
        amplitude = (self.peak_load - self.base_load) / 2
        mean_load = (self.peak_load + self.base_load) / 2
        temporal_load = mean_load + amplitude * np.sin(phase)
        
        # Add per-rack variation
        workload = temporal_load * self.rack_bias
        
        # Add small random noise
        noise = np.random.uniform(-0.05, 0.05, size=(self.rows, self.cols))
        workload = workload + noise
        
        return np.clip(workload, 0.0, 1.0)
    
    def _generate_spikes(self) -> np.ndarray:
        """
        Generate random workload spikes.
        
        Simulates sudden traffic bursts or batch job arrivals.
        """
        # Base load with rack bias
        workload = self.base_load * self.rack_bias
        
        # Random spikes
        spike_mask = np.random.random(size=(self.rows, self.cols)) < self.spike_probability
        spike_intensity = np.random.uniform(0.5, 1.0, size=(self.rows, self.cols))
        workload[spike_mask] = spike_intensity[spike_mask]
        
        return np.clip(workload, 0.0, 1.0)
    
    def _generate_burst(self) -> np.ndarray:
        """
        Generate burst workload pattern.
        
        Simulates batch processing with sustained high load periods.
        """
        # Check if we should start a new burst
        if self.burst_remaining <= 0:
            if np.random.random() < 0.05:  # 5% chance to start burst
                self.burst_duration = np.random.randint(10, 50)  # 10-50 steps
                self.burst_remaining = self.burst_duration
        
        # Generate workload based on burst state
        if self.burst_remaining > 0:
            # High load during burst
            workload = self.peak_load * self.rack_bias
            # Some racks more affected than others
            burst_mask = np.random.random(size=(self.rows, self.cols)) < 0.7
            workload[~burst_mask] = self.base_load * self.rack_bias[~burst_mask]
            self.burst_remaining -= 1
        else:
            # Normal base load
            workload = self.base_load * self.rack_bias
        
        # Add noise
        noise = np.random.uniform(-0.05, 0.05, size=(self.rows, self.cols))
        workload = workload + noise
        
        return np.clip(workload, 0.0, 1.0)
    
    def _generate_mixed(self) -> np.ndarray:
        """
        Generate mixed workload pattern.
        
        Combines sinusoidal base with random spikes.
        """
        # Start with sinusoidal base
        period = 1440
        phase = 2 * np.pi * self.time_step / period
        amplitude = (self.peak_load - self.base_load) / 3
        mean_load = (self.peak_load + self.base_load) / 2
        temporal_load = mean_load + amplitude * np.sin(phase)
        
        workload = temporal_load * self.rack_bias
        
        # Add random spikes
        spike_mask = np.random.random(size=(self.rows, self.cols)) < self.spike_probability
        spike_intensity = np.random.uniform(0.6, 1.0, size=(self.rows, self.cols))
        workload[spike_mask] = np.maximum(workload[spike_mask], spike_intensity[spike_mask])
        
        # Add noise
        noise = np.random.uniform(-0.03, 0.03, size=(self.rows, self.cols))
        workload = workload + noise
        
        return np.clip(workload, 0.0, 1.0)
    
    def _generate_random(self) -> np.ndarray:
        """
        Generate purely random workload.
        """
        workload = np.random.uniform(
            self.base_load,
            self.peak_load,
            size=(self.rows, self.cols)
        )
        return workload
    
    def reset(self):
        """Reset generator to initial state."""
        self.time_step = 0
        self.burst_remaining = 0


class WorkloadScenario:
    """
    Predefined workload scenarios for testing.
    """
    
    @staticmethod
    def create_hotspot_scenario(grid_size: Tuple[int, int]) -> np.ndarray:
        """
        Create a scenario with localized hotspot (uneven load distribution).
        
        Args:
            grid_size: (rows, cols) dimensions
            
        Returns:
            Workload grid with hotspot
        """
        rows, cols = grid_size
        workload = np.full((rows, cols), 0.3)
        
        # Create hotspot in center region
        center_r, center_c = rows // 2, cols // 2
        workload[center_r, center_c] = 0.95
        if center_r > 0:
            workload[center_r - 1, center_c] = 0.85
        if center_r < rows - 1:
            workload[center_r + 1, center_c] = 0.85
        if center_c > 0:
            workload[center_r, center_c - 1] = 0.85
        if center_c < cols - 1:
            workload[center_r, center_c + 1] = 0.85
        
        return workload
    
    @staticmethod
    def create_edge_heavy_scenario(grid_size: Tuple[int, int]) -> np.ndarray:
        """
        Create scenario with high load on edge racks.
        
        Args:
            grid_size: (rows, cols) dimensions
            
        Returns:
            Workload grid with edge emphasis
        """
        rows, cols = grid_size
        workload = np.full((rows, cols), 0.3)
        
        # High load on edges
        workload[0, :] = 0.8
        workload[-1, :] = 0.8
        workload[:, 0] = 0.8
        workload[:, -1] = 0.8
        
        return workload
    
    @staticmethod
    def create_gradient_scenario(grid_size: Tuple[int, int]) -> np.ndarray:
        """
        Create scenario with workload gradient.
        
        Args:
            grid_size: (rows, cols) dimensions
            
        Returns:
            Workload grid with gradient
        """
        rows, cols = grid_size
        workload = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                # Linear gradient from top-left to bottom-right
                workload[i, j] = 0.2 + 0.7 * (i + j) / (rows + cols - 2)
        
        return workload
