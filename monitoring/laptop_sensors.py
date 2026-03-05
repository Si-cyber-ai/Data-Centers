"""
Laptop/System Sensor Monitoring

Monitors real system metrics using psutil for demonstration purposes.
This provides real-world data alongside the simulated environment.
"""

import psutil
import time
import numpy as np
from typing import Dict, Any, Optional, List
from collections import deque
from datetime import datetime
import platform


class LaptopSensorMonitor:
    """
    Monitors laptop/desktop system sensors.
    
    Collects:
    - CPU utilization
    - CPU temperature (if available)
    - CPU frequency
    - Fan speed (platform-dependent)
    - System power draw (if available)
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize sensor monitor.
        
        Args:
            history_size: Number of readings to keep in history
        """
        self.history_size = history_size
        
        # History buffers
        self.cpu_usage_history = deque(maxlen=history_size)
        self.cpu_temp_history = deque(maxlen=history_size)
        self.cpu_freq_history = deque(maxlen=history_size)
        self.power_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        
        # System info
        self.system_info = self._get_system_info()
        
        # Check sensor availability
        self.sensors_available = self._check_sensor_availability()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """
        Get static system information.
        
        Returns:
            System info dictionary
        """
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'processor': platform.processor(),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }
    
    def _check_sensor_availability(self) -> Dict[str, bool]:
        """
        Check which sensors are available on this system.
        
        Returns:
            Dictionary of sensor availability
        """
        available = {
            'cpu_usage': True,  # Always available
            'cpu_freq': False,
            'cpu_temp': False,
            'fan_speed': False,
            'battery': False,
            'power': False
        }
        
        # Check CPU frequency
        try:
            freq = psutil.cpu_freq()
            if freq is not None:
                available['cpu_freq'] = True
        except:
            pass
        
        # Check temperature sensors
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                available['cpu_temp'] = True
        except:
            pass
        
        # Check fan sensors
        try:
            fans = psutil.sensors_fans()
            if fans:
                available['fan_speed'] = True
        except:
            pass
        
        # Check battery
        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                available['battery'] = True
        except:
            pass
        
        return available
    
    def read_sensors(self) -> Dict[str, Any]:
        """
        Read current sensor values.
        
        Returns:
            Dictionary of current sensor readings
        """
        readings = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage_percent': self._read_cpu_usage(),
            'cpu_freq_mhz': self._read_cpu_freq(),
            'cpu_temp_celsius': self._read_cpu_temp(),
            'fan_rpm': self._read_fan_speed(),
            'memory_usage_percent': self._read_memory_usage(),
            'power_watts': self._read_power(),
            'battery_percent': self._read_battery()
        }
        
        # Add to history
        self.cpu_usage_history.append(readings['cpu_usage_percent'])
        if readings['cpu_temp_celsius'] is not None:
            self.cpu_temp_history.append(readings['cpu_temp_celsius'])
        if readings['cpu_freq_mhz'] is not None:
            self.cpu_freq_history.append(readings['cpu_freq_mhz'])
        if readings['power_watts'] is not None:
            self.power_history.append(readings['power_watts'])
        self.timestamp_history.append(time.time())
        
        return readings
    
    def _read_cpu_usage(self) -> float:
        """
        Read CPU utilization percentage.
        
        Returns:
            CPU usage in percent [0, 100]
        """
        return psutil.cpu_percent(interval=0.1)
    
    def _read_cpu_freq(self) -> Optional[float]:
        """
        Read current CPU frequency.
        
        Returns:
            CPU frequency in MHz (None if unavailable)
        """
        if not self.sensors_available['cpu_freq']:
            return None
        
        try:
            freq = psutil.cpu_freq()
            return freq.current if freq else None
        except:
            return None
    
    def _read_cpu_temp(self) -> Optional[float]:
        """
        Read CPU temperature.
        
        Returns:
            CPU temperature in Celsius (None if unavailable)
        """
        if not self.sensors_available['cpu_temp']:
            return None
        
        try:
            temps = psutil.sensors_temperatures()
            
            # Try to find CPU temperature
            # Different systems use different sensor names
            for name, entries in temps.items():
                if 'coretemp' in name.lower() or 'cpu' in name.lower():
                    if entries:
                        return entries[0].current
            
            # If not found, return first available temperature
            if temps:
                first_sensor = list(temps.values())[0]
                if first_sensor:
                    return first_sensor[0].current
            
            return None
        except:
            return None
    
    def _read_fan_speed(self) -> Optional[float]:
        """
        Read fan speed.
        
        Returns:
            Fan speed in RPM (None if unavailable)
        """
        if not self.sensors_available['fan_speed']:
            return None
        
        try:
            fans = psutil.sensors_fans()
            if fans:
                first_fan = list(fans.values())[0]
                if first_fan:
                    return first_fan[0].current
            return None
        except:
            return None
    
    def _read_memory_usage(self) -> float:
        """
        Read memory utilization percentage.
        
        Returns:
            Memory usage in percent
        """
        return psutil.virtual_memory().percent
    
    def _read_power(self) -> Optional[float]:
        """
        Estimate power consumption.
        
        Note: Actual power measurement is platform-specific and often unavailable.
        This provides a rough estimate based on CPU usage.
        
        Returns:
            Estimated power in watts (None if unavailable)
        """
        # On Windows, some laptops expose power info
        # On Linux, might be in /sys/class/power_supply/
        # This is a simplified estimation
        
        try:
            if self.sensors_available['battery']:
                battery = psutil.sensors_battery()
                if battery and battery.power_plugged:
                    # Rough estimate: assume 15W base + CPU usage scaling
                    cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
                    estimated_power = 15.0 + cpu_usage * 50.0  # Up to 65W
                    return estimated_power
        except:
            pass
        
        return None
    
    def _read_battery(self) -> Optional[float]:
        """
        Read battery percentage.
        
        Returns:
            Battery percentage (None if unavailable or on AC power)
        """
        if not self.sensors_available['battery']:
            return None
        
        try:
            battery = psutil.sensors_battery()
            return battery.percent if battery else None
        except:
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics from sensor history.
        
        Returns:
            Statistics dictionary
        """
        stats = {}
        
        if self.cpu_usage_history:
            stats['cpu_usage'] = {
                'current': self.cpu_usage_history[-1],
                'avg': np.mean(self.cpu_usage_history),
                'max': np.max(self.cpu_usage_history),
                'min': np.min(self.cpu_usage_history),
                'std': np.std(self.cpu_usage_history)
            }
        
        if self.cpu_temp_history:
            stats['cpu_temp'] = {
                'current': self.cpu_temp_history[-1],
                'avg': np.mean(self.cpu_temp_history),
                'max': np.max(self.cpu_temp_history),
                'min': np.min(self.cpu_temp_history)
            }
        
        if self.cpu_freq_history:
            stats['cpu_freq'] = {
                'current': self.cpu_freq_history[-1],
                'avg': np.mean(self.cpu_freq_history),
                'max': np.max(self.cpu_freq_history),
                'min': np.min(self.cpu_freq_history)
            }
        
        if self.power_history:
            stats['power'] = {
                'current': self.power_history[-1],
                'avg': np.mean(self.power_history)
            }
        
        return stats
    
    def monitor_continuous(
        self,
        duration_seconds: int = 60,
        interval_seconds: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Continuously monitor sensors for specified duration.
        
        Args:
            duration_seconds: How long to monitor
            interval_seconds: Sampling interval
            
        Returns:
            List of sensor readings
        """
        print(f"Monitoring system sensors for {duration_seconds} seconds...")
        print(f"Available sensors: {self.sensors_available}")
        
        readings_list = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            readings = self.read_sensors()
            readings_list.append(readings)
            
            # Print current readings
            print(f"\rCPU: {readings['cpu_usage_percent']:.1f}% | "
                  f"Temp: {readings['cpu_temp_celsius'] if readings['cpu_temp_celsius'] else 'N/A'} | "
                  f"Memory: {readings['memory_usage_percent']:.1f}%", end='')
            
            time.sleep(interval_seconds)
        
        print("\nMonitoring complete.")
        return readings_list
    
    def get_sensor_availability_report(self) -> str:
        """
        Get human-readable sensor availability report.
        
        Returns:
            Formatted report string
        """
        report = "=== System Sensor Availability ===\n"
        report += f"Platform: {self.system_info['platform']}\n"
        report += f"Processor: {self.system_info['processor']}\n"
        report += f"CPU Cores: {self.system_info['cpu_count_physical']} physical, "
        report += f"{self.system_info['cpu_count_logical']} logical\n\n"
        
        report += "Available Sensors:\n"
        for sensor, available in self.sensors_available.items():
            status = "✓ Available" if available else "✗ Not Available"
            report += f"  {sensor}: {status}\n"
        
        return report


# Example usage and testing
if __name__ == "__main__":
    monitor = LaptopSensorMonitor()
    
    print(monitor.get_sensor_availability_report())
    
    # Take a few readings
    print("\nTaking sensor readings...")
    for i in range(5):
        readings = monitor.read_sensors()
        print(f"\nReading {i+1}:")
        for key, value in readings.items():
            if value is not None:
                print(f"  {key}: {value}")
        time.sleep(1)
    
    # Print statistics
    print("\n=== Statistics ===")
    stats = monitor.get_statistics()
    for category, values in stats.items():
        print(f"\n{category}:")
        for metric, value in values.items():
            print(f"  {metric}: {value:.2f}")
