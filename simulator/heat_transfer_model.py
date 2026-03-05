"""
Heat Transfer Model for Data Center Thermal Simulation

This module implements physical heat transfer equations for modeling
thermal dynamics in a data center environment.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import convolve


class HeatTransferModel:
    """
    Models heat transfer in a 2D grid of server racks.
    
    Implements convection, conduction, and ambient temperature effects.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int],
        alpha: float = 0.15,  # Heat generation coefficient
        beta: float = 0.20,   # Cooling efficiency
        gamma: float = 0.05,  # Heat diffusion coefficient
        delta: float = 0.02,  # Ambient effect coefficient
        noise_std: float = 0.1
    ):
        """
        Initialize heat transfer model.
        
        Args:
            grid_size: (rows, cols) dimensions of rack grid
            alpha: Heat generation coefficient from CPU workload
            beta: Cooling efficiency coefficient
            gamma: Heat diffusion from neighboring racks
            delta: Ambient temperature influence
            noise_std: Standard deviation of thermal noise
        """
        self.rows, self.cols = grid_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.noise_std = noise_std
        
        # Convolution kernel for neighbor heat diffusion
        # Models airflow interaction between adjacent racks
        self.neighbor_kernel = np.array([
            [0.05, 0.10, 0.05],
            [0.10, 0.00, 0.10],
            [0.05, 0.10, 0.05]
        ])
        
    def compute_neighbor_heat(self, temperatures: np.ndarray) -> np.ndarray:
        """
        Compute heat diffusion from neighboring racks using convolution.
        
        Args:
            temperatures: Current temperature grid [rows, cols]
            
        Returns:
            Heat contribution from neighbors [rows, cols]
        """
        # Use convolution to efficiently compute neighbor influence
        neighbor_heat = convolve(
            temperatures,
            self.neighbor_kernel,
            mode='constant',
            cval=0.0
        )
        return neighbor_heat
    
    def update_temperatures(
        self,
        temperatures: np.ndarray,
        cpu_workload: np.ndarray,
        cooling_levels: np.ndarray,
        ambient_temp: float,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Update rack temperatures based on thermal dynamics.
        
        Thermal equation:
        T(t+1) = T(t) + dt * [
            α * CPU_usage - β * Cooling 
            + γ * Neighbor_heat + δ * (Ambient - T)
        ] + noise
        
        Args:
            temperatures: Current temperatures [rows, cols]
            cpu_workload: CPU utilization per rack [rows, cols] in [0, 1]
            cooling_levels: Cooling intensity per rack [rows, cols] in [0, 1]
            ambient_temp: Ambient temperature (scalar)
            dt: Time step multiplier
            
        Returns:
            Updated temperatures [rows, cols]
        """
        # Heat generation from CPU workload
        heat_generation = self.alpha * cpu_workload * 100.0  # Scale to degrees
        
        # Heat removal from cooling
        heat_removal = self.beta * cooling_levels * 100.0
        
        # Heat diffusion from neighbors
        neighbor_heat = self.compute_neighbor_heat(temperatures)
        heat_diffusion = self.gamma * neighbor_heat
        
        # Ambient temperature effect (thermal equilibrium)
        ambient_effect = self.delta * (ambient_temp - temperatures)
        
        # Stochastic thermal noise (sensor noise, airflow turbulence)
        noise = np.random.normal(0, self.noise_std, size=temperatures.shape)
        
        # Update temperatures using forward Euler integration
        temperature_change = (
            heat_generation 
            - heat_removal 
            + heat_diffusion 
            + ambient_effect 
            + noise
        )
        
        new_temperatures = temperatures + dt * temperature_change
        
        # Physical bounds (temperatures cannot be negative)
        new_temperatures = np.maximum(new_temperatures, ambient_temp - 10.0)
        
        return new_temperatures
    
    def compute_thermal_gradient(self, temperatures: np.ndarray) -> float:
        """
        Compute thermal gradient magnitude (hotspot detection).
        
        Args:
            temperatures: Temperature grid
            
        Returns:
            Average gradient magnitude across grid
        """
        grad_y, grad_x = np.gradient(temperatures)
        gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        return np.mean(gradient_magnitude)
    
    def find_hotspots(
        self,
        temperatures: np.ndarray,
        threshold: float = 75.0
    ) -> np.ndarray:
        """
        Identify racks exceeding temperature threshold.
        
        Args:
            temperatures: Temperature grid
            threshold: Temperature threshold in Celsius
            
        Returns:
            Binary mask of hotspot locations
        """
        return temperatures > threshold
    
    def compute_cooling_effectiveness(
        self,
        temp_before: np.ndarray,
        temp_after: np.ndarray,
        cooling_applied: np.ndarray
    ) -> float:
        """
        Measure cooling system effectiveness.
        
        Args:
            temp_before: Temperatures before cooling
            temp_after: Temperatures after cooling
            cooling_applied: Cooling levels applied
            
        Returns:
            Effectiveness score (temperature reduction per cooling unit)
        """
        temp_reduction = np.mean(temp_before - temp_after)
        cooling_effort = np.mean(cooling_applied) + 1e-6  # Avoid division by zero
        return temp_reduction / cooling_effort
