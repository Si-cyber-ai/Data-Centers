"""
Dataset Loader for Real Workload Traces

Loads and processes real data center workload traces from CSV files.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os


class WorkloadTraceLoader:
    """
    Loads and replays real workload traces from datasets.
    
    Supports formats from:
    - Google cluster traces
    - Alibaba cluster traces
    - Bitbrains cloud traces
    """
    
    def __init__(
        self,
        trace_file: str,
        grid_size: Tuple[int, int],
        cpu_column: str = 'cpu_usage',
        timestamp_column: Optional[str] = 'timestamp',
        normalize: bool = True
    ):
        """
        Initialize trace loader.
        
        Args:
            trace_file: Path to CSV file containing workload trace
            grid_size: (rows, cols) dimensions of rack grid
            cpu_column: Name of CPU usage column
            timestamp_column: Name of timestamp column (optional)
            normalize: Whether to normalize CPU values to [0, 1]
        """
        self.trace_file = trace_file
        self.rows, self.cols = grid_size
        self.num_racks = self.rows * self.cols
        self.cpu_column = cpu_column
        self.timestamp_column = timestamp_column
        self.normalize = normalize
        
        # Load trace data
        self.data = None
        self.current_index = 0
        self.trace_length = 0
        
        if os.path.exists(trace_file):
            self._load_trace()
        else:
            print(f"Warning: Trace file {trace_file} not found. Using synthetic data.")
            self.data = None
    
    def _load_trace(self):
        """Load trace data from CSV file."""
        try:
            self.data = pd.read_csv(self.trace_file)
            
            # Check if required column exists
            if self.cpu_column not in self.data.columns:
                print(f"Warning: Column '{self.cpu_column}' not found. Available columns: {list(self.data.columns)}")
                self.data = None
                return
            
            # Normalize if needed
            if self.normalize:
                cpu_values = self.data[self.cpu_column].values
                min_val = np.min(cpu_values)
                max_val = np.max(cpu_values)
                if max_val > min_val:
                    self.data[self.cpu_column] = (cpu_values - min_val) / (max_val - min_val)
                else:
                    self.data[self.cpu_column] = 0.5  # Default if all values same
            
            self.trace_length = len(self.data)
            print(f"Loaded workload trace: {self.trace_length} samples")
            
        except Exception as e:
            print(f"Error loading trace file: {e}")
            self.data = None
    
    def generate(self, step: Optional[int] = None) -> np.ndarray:
        """
        Generate workload from trace data.
        
        Args:
            step: Optional time step (if None, uses internal counter)
            
        Returns:
            CPU workload grid [rows, cols] in range [0, 1]
        """
        if step is not None:
            self.current_index = step % self.trace_length if self.trace_length > 0 else 0
        
        if self.data is None or self.trace_length == 0:
            # Fallback to random workload
            return np.random.uniform(0.3, 0.7, size=(self.rows, self.cols))
        
        # Get CPU values from trace
        # If trace has fewer entries than racks, cycle through
        workload_flat = []
        for i in range(self.num_racks):
            idx = (self.current_index + i) % self.trace_length
            cpu_val = self.data[self.cpu_column].iloc[idx]
            workload_flat.append(cpu_val)
        
        workload = np.array(workload_flat).reshape(self.rows, self.cols)
        
        # Add small noise to make it more realistic
        noise = np.random.uniform(-0.02, 0.02, size=(self.rows, self.cols))
        workload = np.clip(workload + noise, 0.0, 1.0)
        
        self.current_index = (self.current_index + 1) % self.trace_length
        
        return workload
    
    def reset(self):
        """Reset to beginning of trace."""
        self.current_index = 0
    
    @staticmethod
    def create_sample_trace(
        output_file: str,
        num_samples: int = 1000,
        pattern: str = "realistic"
    ):
        """
        Create a sample workload trace CSV file.
        
        Args:
            output_file: Path to output CSV file
            num_samples: Number of samples to generate
            pattern: Pattern type ('realistic', 'variable', 'stable')
        """
        timestamps = pd.date_range(
            start='2024-01-01',
            periods=num_samples,
            freq='5min'
        )
        
        if pattern == "realistic":
            # Realistic daily pattern
            hours = np.arange(num_samples) % 288  # 288 5-min intervals per day
            base_load = 0.3
            daily_pattern = base_load + 0.4 * np.sin(2 * np.pi * hours / 288)
            noise = np.random.normal(0, 0.1, size=num_samples)
            spikes = np.random.random(num_samples) < 0.05
            cpu_usage = daily_pattern + noise
            cpu_usage[spikes] += np.random.uniform(0.3, 0.5, size=np.sum(spikes))
            cpu_usage = np.clip(cpu_usage, 0.0, 1.0)
            
        elif pattern == "variable":
            # Highly variable load
            cpu_usage = np.random.beta(2, 2, size=num_samples)
            
        else:  # stable
            # Relatively stable load
            cpu_usage = np.random.normal(0.5, 0.1, size=num_samples)
            cpu_usage = np.clip(cpu_usage, 0.0, 1.0)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_usage': cpu_usage,
            'memory_usage': np.random.uniform(0.4, 0.8, size=num_samples),
            'network_traffic': np.random.exponential(0.3, size=num_samples)
        })
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Created sample workload trace: {output_file}")


def create_google_cluster_adapter(trace_file: str) -> pd.DataFrame:
    """
    Adapter for Google cluster trace format.
    
    Google traces have format: timestamp, machine_id, cpu_usage, memory_usage
    
    Args:
        trace_file: Path to Google trace file
        
    Returns:
        Standardized DataFrame
    """
    try:
        df = pd.read_csv(trace_file)
        
        # Google traces may have different column names
        # Adapt to standard format
        if 'cpuUsage' in df.columns:
            df['cpu_usage'] = df['cpuUsage']
        elif 'cpu' in df.columns:
            df['cpu_usage'] = df['cpu']
        
        return df
    except Exception as e:
        print(f"Error loading Google cluster trace: {e}")
        return pd.DataFrame()


def create_alibaba_cluster_adapter(trace_file: str) -> pd.DataFrame:
    """
    Adapter for Alibaba cluster trace format.
    
    Args:
        trace_file: Path to Alibaba trace file
        
    Returns:
        Standardized DataFrame
    """
    try:
        df = pd.read_csv(trace_file)
        
        # Alibaba traces may have different format
        # Adapt to standard format
        if 'cpu_util' in df.columns:
            df['cpu_usage'] = df['cpu_util']
        
        return df
    except Exception as e:
        print(f"Error loading Alibaba cluster trace: {e}")
        return pd.DataFrame()
