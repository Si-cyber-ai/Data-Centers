"""
PID Controller for Data Center Cooling

Classical Proportional-Integral-Derivative controller for baseline comparison.
"""

import numpy as np
from typing import Tuple, Optional


class PIDController:
    """
    PID controller for data center cooling management.
    
    Maintains rack temperatures near setpoint by adjusting cooling levels.
    """
    
    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
        setpoint: float = 65.0,
        output_limits: Tuple[float, float] = (0.0, 1.0),
        integral_limits: Tuple[float, float] = (-10.0, 10.0)
    ):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            setpoint: Target temperature in Celsius
            output_limits: (min, max) cooling output bounds
            integral_limits: Anti-windup limits for integral term
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        
        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_temperatures: Optional[np.ndarray] = None
        
    def compute(
        self,
        current_temperatures: np.ndarray,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Compute cooling control output using PID.
        
        Args:
            current_temperatures: Current rack temperatures [rows, cols]
            dt: Time step
            
        Returns:
            Cooling levels [rows, cols] in range [0, 1]
        """
        # Compute error (deviation from setpoint)
        error = current_temperatures - self.setpoint
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, *self.integral_limits)
        i_term = self.ki * self.integral
        
        # Derivative term
        if self.prev_temperatures is not None:
            temp_change = (current_temperatures - self.prev_temperatures) / dt
            d_term = self.kd * temp_change
        else:
            d_term = 0.0
        
        # Total control output
        output = p_term + i_term + d_term
        
        # Normalize to cooling level [0, 1]
        # Higher error (hotter temp) -> higher cooling
        cooling_levels = np.clip(output / 100.0, *self.output_limits)
        
        # Ensure positive cooling (cannot "anti-cool")
        cooling_levels = np.maximum(cooling_levels, 0.0)
        
        # Update state
        self.prev_error = error
        self.prev_temperatures = current_temperatures.copy()
        
        return cooling_levels
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_temperatures = None
    
    def set_setpoint(self, setpoint: float):
        """
        Update temperature setpoint.
        
        Args:
            setpoint: New target temperature
        """
        self.setpoint = setpoint
    
    def tune(self, kp: float, ki: float, kd: float):
        """
        Update PID gains.
        
        Args:
            kp: New proportional gain
            ki: New integral gain
            kd: New derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd


class AdaptivePIDController(PIDController):
    """
    Adaptive PID controller with gain scheduling.
    
    Adjusts gains based on system state for better performance.
    """
    
    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
        setpoint: float = 65.0,
        output_limits: Tuple[float, float] = (0.0, 1.0),
        integral_limits: Tuple[float, float] = (-10.0, 10.0)
    ):
        """Initialize adaptive PID controller."""
        super().__init__(kp, ki, kd, setpoint, output_limits, integral_limits)
        
        # Base gains for adaptation
        self.base_kp = kp
        self.base_ki = ki
        self.base_kd = kd
    
    def compute(
        self,
        current_temperatures: np.ndarray,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Compute adaptive PID control output.
        
        Gains are adjusted based on temperature error magnitude.
        
        Args:
            current_temperatures: Current rack temperatures [rows, cols]
            dt: Time step
            
        Returns:
            Cooling levels [rows, cols]
        """
        # Compute average error magnitude
        avg_temp = np.mean(current_temperatures)
        error_magnitude = abs(avg_temp - self.setpoint)
        
        # Gain scheduling: increase gains for larger errors
        if error_magnitude > 10.0:
            # Large error: aggressive control
            self.kp = self.base_kp * 1.5
            self.ki = self.base_ki * 1.2
            self.kd = self.base_kd * 1.3
        elif error_magnitude > 5.0:
            # Moderate error: nominal gains
            self.kp = self.base_kp
            self.ki = self.base_ki
            self.kd = self.base_kd
        else:
            # Small error: gentle control
            self.kp = self.base_kp * 0.7
            self.ki = self.base_ki * 0.8
            self.kd = self.base_kd * 0.9
        
        # Use parent class compute with adapted gains
        return super().compute(current_temperatures, dt)


class ZonePIDController:
    """
    Multi-zone PID controller for spatial control.
    
    Divides rack grid into zones with independent PID controllers.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int],
        num_zones: Tuple[int, int] = (2, 2),
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
        setpoint: float = 65.0
    ):
        """
        Initialize zone PID controller.
        
        Args:
            grid_size: (rows, cols) dimensions of rack grid
            num_zones: (zone_rows, zone_cols) number of zones
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            setpoint: Target temperature
        """
        self.rows, self.cols = grid_size
        self.zone_rows, self.zone_cols = num_zones
        
        # Create PID controller for each zone
        self.controllers = []
        for _ in range(self.zone_rows * self.zone_cols):
            controller = PIDController(kp, ki, kd, setpoint)
            self.controllers.append(controller)
        
        # Compute zone boundaries
        self.row_boundaries = np.linspace(0, self.rows, self.zone_rows + 1, dtype=int)
        self.col_boundaries = np.linspace(0, self.cols, self.zone_cols + 1, dtype=int)
    
    def compute(
        self,
        current_temperatures: np.ndarray,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Compute cooling for each zone independently.
        
        Args:
            current_temperatures: Current rack temperatures [rows, cols]
            dt: Time step
            
        Returns:
            Cooling levels [rows, cols]
        """
        cooling_levels = np.zeros((self.rows, self.cols))
        
        controller_idx = 0
        for i in range(self.zone_rows):
            for j in range(self.zone_cols):
                # Extract zone temperatures
                row_start = self.row_boundaries[i]
                row_end = self.row_boundaries[i + 1]
                col_start = self.col_boundaries[j]
                col_end = self.col_boundaries[j + 1]
                
                zone_temps = current_temperatures[row_start:row_end, col_start:col_end]
                
                # Compute cooling for this zone
                zone_cooling = self.controllers[controller_idx].compute(zone_temps, dt)
                
                # Assign to output grid
                cooling_levels[row_start:row_end, col_start:col_end] = zone_cooling
                
                controller_idx += 1
        
        return cooling_levels
    
    def reset(self):
        """Reset all zone controllers."""
        for controller in self.controllers:
            controller.reset()
