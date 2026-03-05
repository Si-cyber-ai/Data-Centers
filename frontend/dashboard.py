"""
Interactive Streamlit Dashboard for Data Center Cooling Optimization

Provides real-time visualization and control of the cooling simulation.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import yaml
import os
import time

from simulator.thermal_environment import DataCenterThermalEnv
from workload.synthetic_generator import SyntheticWorkloadGenerator, WorkloadScenario
from rl_agent.dqn_agent import DQNAgent
from controllers.pid_controller import PIDController
from monitoring.laptop_sensors import LaptopSensorMonitor
from safety.safety_override import SafetyOverride


# Page configuration
st.set_page_config(
    page_title="AI Data Center Cooling",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    """Load configuration file."""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)


@st.cache_resource
def initialize_environment(config):
    """Initialize simulation environment."""
    grid_size = tuple(config['simulation']['grid_size'])
    workload_gen = SyntheticWorkloadGenerator(
        grid_size=grid_size,
        pattern=config['workload']['synthetic_pattern'],
        base_load=config['workload']['base_load'],
        peak_load=config['workload']['peak_load']
    )
    env = DataCenterThermalEnv(
        config_path="config.yaml",
        workload_generator=workload_gen
    )
    return env, workload_gen


@st.cache_resource
def initialize_controllers(config, env):
    """Initialize RL and PID controllers."""
    # RL Agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    rl_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['rl']['hidden_dim']
    )
    
    # Try to load checkpoint if exists
    checkpoint_path = "checkpoints/dqn_final.pth"
    if os.path.exists(checkpoint_path):
        rl_agent.load_checkpoint(checkpoint_path)
    
    # PID Controller
    pid_controller = PIDController(
        kp=config['pid']['kp'],
        ki=config['pid']['ki'],
        kd=config['pid']['kd'],
        setpoint=config['pid']['setpoint']
    )
    
    return rl_agent, pid_controller


def create_heatmap(data, title, colorscale='Hot'):
    """Create interactive heatmap using Plotly."""
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale=colorscale,
        colorbar=dict(title="Value")
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Rack Column",
        yaxis_title="Rack Row",
        height=300
    )
    return fig


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<div class="main-header">❄️ AI-Based Data Center Cooling Optimization</div>', 
                unsafe_allow_html=True)
    st.markdown("**Safe Reinforcement Learning with Digital Twin Simulation**")
    st.divider()
    
    # Load configuration
    config = load_config()
    
    # Sidebar controls
    st.sidebar.title("⚙️ Control Panel")
    
    # Simulation mode
    mode = st.sidebar.radio(
        "Simulation Mode",
        ["Digital Twin", "Real System Monitor", "Comparison"]
    )
    
    st.sidebar.divider()
    
    # Controller selection
    controller_type = st.sidebar.selectbox(
        "Controller Type",
        ["RL (DQN)", "PID", "Adaptive PID"]
    )
    
    # Simulation parameters
    st.sidebar.subheader("Simulation Parameters")
    
    ambient_temp = st.sidebar.slider(
        "Ambient Temperature (°C)",
        15.0, 35.0,
        float(config['simulation']['ambient_temperature']),
        0.5
    )
    
    workload_pattern = st.sidebar.selectbox(
        "Workload Pattern",
        ["mixed", "sinusoidal", "spikes", "burst"]
    )
    
    alpha = st.sidebar.slider(
        "Heat Generation (α)",
        0.05, 0.30,
        float(config['simulation']['alpha']),
        0.01
    )
    
    beta = st.sidebar.slider(
        "Cooling Efficiency (β)",
        0.10, 0.40,
        float(config['simulation']['beta']),
        0.01
    )
    
    # PID tuning (if PID selected)
    if "PID" in controller_type:
        st.sidebar.subheader("PID Tuning")
        kp = st.sidebar.slider("Kp", 0.0, 2.0, float(config['pid']['kp']), 0.1)
        ki = st.sidebar.slider("Ki", 0.0, 1.0, float(config['pid']['ki']), 0.05)
        kd = st.sidebar.slider("Kd", 0.0, 0.5, float(config['pid']['kd']), 0.01)
    
    # Scenario selection
    st.sidebar.subheader("Test Scenarios")
    scenario = st.sidebar.selectbox(
        "Load Scenario",
        ["Normal", "Hotspot", "Edge Heavy", "Gradient"]
    )
    
    # Control buttons
    st.sidebar.divider()
    run_simulation = st.sidebar.button("▶️ Run Simulation", type="primary")
    stop_simulation = st.sidebar.button("⏹️ Stop")
    reset_simulation = st.sidebar.button("🔄 Reset")
    
    # Main content area
    if mode == "Digital Twin":
        display_digital_twin(
            config, controller_type, ambient_temp, workload_pattern,
            alpha, beta, scenario, run_simulation
        )
    
    elif mode == "Real System Monitor":
        display_real_system_monitor()
    
    elif mode == "Comparison":
        display_comparison_mode(config)


def display_digital_twin(
    config, controller_type, ambient_temp, workload_pattern,
    alpha, beta, scenario, run_simulation
):
    """Display digital twin simulation."""
    
    # Initialize session state
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'step_count' not in st.session_state:
        st.session_state.step_count = 0
    if 'history' not in st.session_state:
        st.session_state.history = {
            'temperatures': [],
            'cooling': [],
            'workload': [],
            'rewards': []
        }
    
    # Create environment
    env, workload_gen = initialize_environment(config)
    rl_agent, pid_controller = initialize_controllers(config, env)
    
    # Update environment parameters
    env.ambient_temp = ambient_temp
    env.heat_model.alpha = alpha
    env.heat_model.beta = beta
    workload_gen.pattern = workload_pattern
    
    # Apply scenario
    if scenario != "Normal":
        grid_size = tuple(config['simulation']['grid_size'])
        if scenario == "Hotspot":
            workload = WorkloadScenario.create_hotspot_scenario(grid_size)
        elif scenario == "Edge Heavy":
            workload = WorkloadScenario.create_edge_heavy_scenario(grid_size)
        elif scenario == "Gradient":
            workload = WorkloadScenario.create_gradient_scenario(grid_size)
        env.cpu_workload = workload
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    if run_simulation and not st.session_state.simulation_running:
        st.session_state.simulation_running = True
        state, _ = env.reset()
        
        # Placeholder for live updates
        metrics_placeholder = st.empty()
        heatmap_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # Run simulation steps
        for step in range(100):
            if not st.session_state.simulation_running:
                break
            
            # Get state grids
            state_grids = env.get_state_grid()
            temperatures = state_grids['temperatures']
            cooling_levels = state_grids['cooling_levels']
            workload = state_grids['cpu_workload']
            
            # Select action based on controller
            if controller_type == "RL (DQN)":
                action = rl_agent.select_action(state, training=False)
            else:
                # PID control
                proposed_cooling = pid_controller.compute(temperatures)
                env.cooling_levels = np.clip(proposed_cooling, 0.0, 1.0)
                action = 1  # Dummy action
            
            # Step environment
            state, reward, terminated, truncated, info = env.step(action)
            
            # Update history
            st.session_state.history['temperatures'].append(temperatures.copy())
            st.session_state.history['cooling'].append(cooling_levels.copy())
            st.session_state.history['workload'].append(workload.copy())
            st.session_state.history['rewards'].append(reward)
            st.session_state.step_count += 1
            
            # Update metrics
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Temperature", f"{info['avg_temperature']:.1f} °C")
                with col2:
                    st.metric("Max Temperature", f"{info['max_temperature']:.1f} °C")
                with col3:
                    st.metric("Avg Cooling", f"{info['avg_cooling']:.2f}")
                with col4:
                    st.metric("Violations", str(info['hotspots']))
            
            # Update heatmaps
            with heatmap_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    fig_temp = create_heatmap(temperatures, "Temperature (°C)", "Hot")
                    st.plotly_chart(fig_temp, use_container_width=True)
                with col2:
                    fig_cooling = create_heatmap(cooling_levels, "Cooling Level", "Blues")
                    st.plotly_chart(fig_cooling, use_container_width=True)
                with col3:
                    fig_workload = create_heatmap(workload, "CPU Workload", "Viridis")
                    st.plotly_chart(fig_workload, use_container_width=True)
            
            # Update time series
            with chart_placeholder.container():
                if len(st.session_state.history['temperatures']) > 1:
                    temps_mean = [np.mean(t) for t in st.session_state.history['temperatures']]
                    cooling_mean = [np.mean(c) for c in st.session_state.history['cooling']]
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("Average Temperature", "Average Cooling Level")
                    )
                    
                    fig.add_trace(
                        go.Scatter(y=temps_mean, mode='lines', name='Temperature'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(y=cooling_mean, mode='lines', name='Cooling'),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            time.sleep(0.1)  # Simulation speed
            
            if terminated:
                break
        
        st.session_state.simulation_running = False
    else:
        st.info("Click 'Run Simulation' to start")


def display_real_system_monitor():
    """Display real system monitoring."""
    st.subheader("🖥️ Real System Monitoring")
    
    # Initialize monitor
    if 'monitor' not in st.session_state:
        st.session_state.monitor = LaptopSensorMonitor()
    
    monitor = st.session_state.monitor
    
    # Display sensor availability
    with st.expander("Sensor Availability", expanded=True):
        st.text(monitor.get_sensor_availability_report())
    
    # Live monitoring
    st.subheader("Live Sensor Readings")
    
    if st.button("Start Monitoring"):
        placeholder = st.empty()
        
        for _ in range(60):  # Monitor for 60 seconds
            readings = monitor.read_sensors()
            
            with placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("CPU Usage", f"{readings['cpu_usage_percent']:.1f}%")
                with col2:
                    temp = readings['cpu_temp_celsius']
                    st.metric("CPU Temp", f"{temp:.1f} °C" if temp else "N/A")
                with col3:
                    freq = readings['cpu_freq_mhz']
                    st.metric("CPU Freq", f"{freq:.0f} MHz" if freq else "N/A")
                with col4:
                    st.metric("Memory", f"{readings['memory_usage_percent']:.1f}%")
                
                # Plot history
                if len(monitor.cpu_usage_history) > 1:
                    df = pd.DataFrame({
                        'CPU Usage': list(monitor.cpu_usage_history)
                    })
                    st.line_chart(df)
            
            time.sleep(1)


def display_comparison_mode(config):
    """Display side-by-side comparison of RL vs PID."""
    st.subheader("📊 Controller Comparison: RL vs PID")
    
    if st.button("Run Comparison Experiment"):
        from evaluation.experiments import ExperimentRunner
        
        runner = ExperimentRunner()
        
        with st.spinner("Running comparison experiment..."):
            results = runner.run_comparison_experiment(
                rl_checkpoint="checkpoints/dqn_final.pth",
                workload_pattern="mixed",
                num_episodes=5
            )
        
        st.success("Comparison complete!")
        
        # Display comparison table
        st.dataframe(results['comparison'])
        
        # Display key metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RL Controller")
            rl_metrics = results['rl_metrics']
            st.json(rl_metrics)
        
        with col2:
            st.subheader("PID Controller")
            pid_metrics = results['pid_metrics']
            st.json(pid_metrics)


if __name__ == "__main__":
    main()
