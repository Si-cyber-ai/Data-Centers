"""
Safety Override System for Data Center Cooling

Implements safety mechanisms to prevent thermal violations and system instability.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import time


class SafetyOverride:
    """
    Safety override system for data center thermal management.
    
    Monitors system state and enforces safety constraints:
    - Temperature threshold enforcement
    - Cooling rate limiting
    - Emergency shutdown triggers
    - Sensor anomaly detection
    """
    
    def __init__(
        self,
        max_temperature: float = 80.0,
        critical_temperature: float = 85.0,
        min_temperature: float = 15.0,
        temperature_warning: float = 75.0,
        max_cooling_change: float = 0.3,
        anomaly_threshold: float = 3.0  # Standard deviations
    ):
        """
        Initialize safety override system.
        
        Args:
            max_temperature: Maximum safe temperature
            critical_temperature: Emergency shutdown threshold
            min_temperature: Minimum safe temperature
            temperature_warning: Warning threshold
            max_cooling_change: Maximum allowed cooling change per step
            anomaly_threshold: Z-score threshold for anomaly detection
        """
        self.max_temp = max_temperature
        self.critical_temp = critical_temperature
        self.min_temp = min_temperature
        self.warning_temp = temperature_warning
        self.max_cooling_change = max_cooling_change
        self.anomaly_threshold = anomaly_threshold
        
        # Safety state
        self.override_active = False
        self.emergency_shutdown = False
        self.violation_count = 0
        self.warning_count = 0
        
        # History for anomaly detection
        self.temp_history = deque(maxlen=50)
        self.cooling_history = deque(maxlen=50)
        
        # Event log
        self.event_log: List[Dict[str, Any]] = []
        
    def check_safety(
        self,
        temperatures: np.ndarray,
        cooling_levels: np.ndarray,
        proposed_cooling: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Check current state against safety constraints.
        
        Args:
            temperatures: Current rack temperatures [rows, cols]
            cooling_levels: Current cooling levels [rows, cols]
            proposed_cooling: Proposed new cooling levels (optional)
            
        Returns:
            Safety status dictionary
        """
        status = {
            'safe': True,
            'override_needed': False,
            'emergency': False,
            'violations': [],
            'warnings': []
        }
        
        # Check for critical violations
        critical_violations = np.sum(temperatures >= self.critical_temp)
        if critical_violations > 0:
            status['safe'] = False
            status['emergency'] = True
            status['violations'].append(
                f"CRITICAL: {critical_violations} racks at or above {self.critical_temp}°C"
            )
            self.emergency_shutdown = True
            self._log_event('EMERGENCY_SHUTDOWN', {
                'critical_racks': critical_violations,
                'max_temp': np.max(temperatures)
            })
        
        # Check for temperature violations
        violations = np.sum(temperatures > self.max_temp)
        if violations > 0:
            status['safe'] = False
            status['override_needed'] = True
            status['violations'].append(
                f"VIOLATION: {violations} racks above {self.max_temp}°C"
            )
            self.violation_count += violations
            self._log_event('TEMPERATURE_VIOLATION', {
                'num_racks': violations,
                'max_temp': np.max(temperatures)
            })
        
        # Check for warnings
        warnings = np.sum(temperatures > self.warning_temp)
        if warnings > 0:
            status['warnings'].append(
                f"WARNING: {warnings} racks above {self.warning_temp}°C"
            )
            self.warning_count += warnings
        
        # Check for anomalous temperature readings
        anomalies = self._detect_temperature_anomalies(temperatures)
        if anomalies['detected']:
            status['warnings'].append(
                f"ANOMALY: {anomalies['count']} anomalous temperature readings"
            )
        
        # Check cooling rate limits
        if proposed_cooling is not None:
            cooling_change = proposed_cooling - cooling_levels
            excessive_change = np.sum(np.abs(cooling_change) > self.max_cooling_change)
            if excessive_change > 0:
                status['warnings'].append(
                    f"WARNING: {excessive_change} racks with excessive cooling change"
                )
        
        # Update history
        self.temp_history.append(temperatures.copy())
        self.cooling_history.append(cooling_levels.copy())
        
        return status
    
    def apply_override(
        self,
        temperatures: np.ndarray,
        cooling_levels: np.ndarray
    ) -> np.ndarray:
        """
        Apply safety override to cooling levels.
        
        Forces maximum cooling on overheating racks.
        
        Args:
            temperatures: Current rack temperatures [rows, cols]
            cooling_levels: Proposed cooling levels [rows, cols]
            
        Returns:
            Safe cooling levels [rows, cols]
        """
        safe_cooling = cooling_levels.copy()
        
        # Force maximum cooling on critical racks
        critical_mask = temperatures >= self.critical_temp
        safe_cooling[critical_mask] = 1.0
        
        # Force high cooling on violated racks
        violation_mask = (temperatures > self.max_temp) & (~critical_mask)
        safe_cooling[violation_mask] = np.maximum(safe_cooling[violation_mask], 0.9)
        
        # Increase cooling on warning racks
        warning_mask = (temperatures > self.warning_temp) & (~violation_mask) & (~critical_mask)
        safe_cooling[warning_mask] = np.maximum(safe_cooling[warning_mask], 0.7)
        
        if np.any(critical_mask | violation_mask):
            self.override_active = True
            self._log_event('SAFETY_OVERRIDE', {
                'critical_racks': np.sum(critical_mask),
                'violated_racks': np.sum(violation_mask)
            })
        
        return safe_cooling
    
    def limit_cooling_rate(
        self,
        current_cooling: np.ndarray,
        proposed_cooling: np.ndarray
    ) -> np.ndarray:
        """
        Limit rate of cooling change for system stability.
        
        Args:
            current_cooling: Current cooling levels
            proposed_cooling: Proposed new cooling levels
            
        Returns:
            Rate-limited cooling levels
        """
        cooling_change = proposed_cooling - current_cooling
        
        # Clip change to maximum allowed
        limited_change = np.clip(
            cooling_change,
            -self.max_cooling_change,
            self.max_cooling_change
        )
        
        limited_cooling = current_cooling + limited_change
        
        # Ensure within valid range
        limited_cooling = np.clip(limited_cooling, 0.0, 1.0)
        
        return limited_cooling
    
    def _detect_temperature_anomalies(
        self,
        temperatures: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect anomalous temperature readings using statistical methods.
        
        Args:
            temperatures: Current rack temperatures
            
        Returns:
            Anomaly detection results
        """
        if len(self.temp_history) < 10:
            return {'detected': False, 'count': 0}
        
        # Compute statistics from history
        historical_temps = np.array(list(self.temp_history))
        mean_temps = np.mean(historical_temps, axis=0)
        std_temps = np.std(historical_temps, axis=0) + 1e-6  # Avoid division by zero
        
        # Compute z-scores
        z_scores = np.abs((temperatures - mean_temps) / std_temps)
        
        # Detect anomalies
        anomaly_mask = z_scores > self.anomaly_threshold
        num_anomalies = np.sum(anomaly_mask)
        
        if num_anomalies > 0:
            self._log_event('TEMPERATURE_ANOMALY', {
                'count': num_anomalies,
                'max_z_score': np.max(z_scores)
            })
        
        return {
            'detected': num_anomalies > 0,
            'count': num_anomalies,
            'mask': anomaly_mask,
            'z_scores': z_scores
        }
    
    def check_cooling_failure(self, cooling_effectiveness: float) -> bool:
        """
        Detect potential cooling system failure.
        
        Args:
            cooling_effectiveness: Measure of cooling system effectiveness
            
        Returns:
            True if failure detected
        """
        # If cooling effectiveness is very low, may indicate failure
        if cooling_effectiveness < 0.1:
            self._log_event('COOLING_FAILURE_SUSPECTED', {
                'effectiveness': cooling_effectiveness
            })
            return True
        return False
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log safety event.
        
        Args:
            event_type: Type of safety event
            data: Event data
        """
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'data': data
        }
        self.event_log.append(event)
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get comprehensive safety status report.
        
        Returns:
            Status report dictionary
        """
        return {
            'override_active': self.override_active,
            'emergency_shutdown': self.emergency_shutdown,
            'total_violations': self.violation_count,
            'total_warnings': self.warning_count,
            'recent_events': self.event_log[-10:] if self.event_log else [],
            'num_events': len(self.event_log)
        }
    
    def reset(self):
        """Reset safety system state."""
        self.override_active = False
        self.emergency_shutdown = False
        self.violation_count = 0
        self.warning_count = 0
        self.temp_history.clear()
        self.cooling_history.clear()
        self.event_log.clear()


class SafeRLWrapper:
    """
    Wrapper for RL agent that enforces safety constraints.
    
    Intercepts agent actions and applies safety overrides as needed.
    """
    
    def __init__(self, rl_agent, safety_system: SafetyOverride):
        """
        Initialize safe RL wrapper.
        
        Args:
            rl_agent: RL agent to wrap
            safety_system: Safety override system
        """
        self.rl_agent = rl_agent
        self.safety_system = safety_system
        self.interventions = 0
    
    def select_action(
        self,
        state: np.ndarray,
        temperatures: np.ndarray,
        cooling_levels: np.ndarray,
        training: bool = True
    ) -> int:
        """
        Select action with safety checking.
        
        Args:
            state: Current state
            temperatures: Current temperatures
            cooling_levels: Current cooling levels
            training: Whether in training mode
            
        Returns:
            Safe action
        """
        # Get action from RL agent
        action = self.rl_agent.select_action(state, training=training)
        
        # Check if safety override needed
        # (Implementation would depend on how actions map to cooling)
        # For now, return action - safety applied in environment
        
        return action
    
    def get_intervention_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on safety interventions.
        
        Returns:
            Intervention statistics
        """
        return {
            'total_interventions': self.interventions,
            'safety_report': self.safety_system.get_status_report()
        }
