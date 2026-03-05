# Quick Start Guide

## AI-Based Data Center Cooling Optimization

### Installation (5 minutes)

1. **Install Python 3.8+** (if not already installed)
   ```bash
   python --version  # Should be 3.8 or higher
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   If you encounter issues, install PyTorch first:
   ```bash
   # CPU version
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

### Running Your First Simulation (2 minutes)

**Option 1: Train RL Agent (10 minutes)**
```bash
python train_model.py --episodes 100
```

**Option 2: Run Pre-configured Simulation**
```bash
python run_simulation.py --controller pid --steps 200
```

**Option 3: Launch Interactive Dashboard**
```bash
streamlit run frontend/dashboard.py
```
Then open your browser to `http://localhost:8501`

### Understanding the Simulation

#### What's Being Simulated?
- A **3x4 grid of server racks** (12 racks total)
- Each rack generates **heat** based on CPU workload
- A **cooling system** removes heat
- **Goal**: Keep temperatures safe while minimizing energy

#### Key Metrics to Watch
- **Temperature**: Should stay around 65°C
- **Max Safe Temp**: 80°C (violations are bad!)
- **Cooling Level**: How much energy we're using (0-1 scale)
- **Violations**: Number of racks that overheat

### Common Workflows

#### 1. Quick Demo
```bash
# Run PID controller for 200 steps
python run_simulation.py --controller pid --steps 200 --workload mixed
```

#### 2. Train Your Own RL Agent
```bash
# Train for 500 episodes (takes ~30 minutes)
python train_model.py --episodes 500

# Evaluate it
python train_model.py --evaluate --resume checkpoints/dqn_final.pth
```

#### 3. Compare Controllers
```bash
# Compare RL vs PID
python run_simulation.py --controller compare
```

#### 4. Test Different Scenarios
```bash
# Hotspot scenario (one rack gets very hot)
python run_simulation.py --controller rl --scenario hotspot

# Spike workload (random bursts)
python run_simulation.py --controller pid --workload spikes
```

### Customizing Parameters

Edit `config.yaml`:

```yaml
simulation:
  grid_size: [3, 4]           # Change rack grid size
  ambient_temperature: 25.0    # Change base temperature
  
safety:
  max_temperature: 80.0        # Change safety threshold

pid:
  kp: 0.5                      # Tune PID controller
  ki: 0.1
  kd: 0.05
```

### Interpreting Results

**Good Performance**:
- Average temperature: 60-70°C
- Zero violations
- Cooling level: 0.4-0.6
- Stable temperature (low oscillation)

**Poor Performance**:
- Temperature violations (>80°C)
- High cooling level (>0.8) = wasting energy
- Large temperature swings

### Next Steps

1. **Experiment with workload patterns**:
   - Try `--workload sinusoidal` for daily cycles
   - Try `--workload burst` for batch jobs

2. **Tune the controllers**:
   - Adjust PID gains in `config.yaml`
   - Train RL agent longer for better performance

3. **Create custom scenarios**:
   - Edit `workload/synthetic_generator.py`
   - Add your own workload patterns

4. **Explore the dashboard**:
   - Real-time visualization
   - Interactive parameter tuning
   - System monitoring

### Troubleshooting

**"No module named 'torch'"**
```bash
pip install torch
```

**"CUDA not available"**
- This is fine! The code works on CPU
- GPU speeds up training but isn't required

**"Checkpoint not found"**
- Train a model first: `python train_model.py --episodes 100`
- Or use PID controller: `--controller pid`

**Dashboard won't open**
```bash
# Make sure streamlit is installed
pip install streamlit

# Run with explicit port
streamlit run frontend/dashboard.py --server.port 8501
```

### Getting Help

- Check the full **README.md** for detailed documentation
- Review code comments in each module
- Open an issue on GitHub

### Example Output

```
==================================================================
Running PID Controller Simulation
==================================================================
Applying scenario: None
Running simulation for 200 steps...

==================================================================
SIMULATION RESULTS
==================================================================
Controller: PID

Energy Metrics:
  Average Cooling: 0.523
  Peak Cooling: 0.847
  Total Energy: 2.34 kWh

Temperature Metrics:
  Average: 67.23°C
  Max: 78.45°C
  Deviation from Target: 4.87°C
  Violations: 0
  Comfort Zone Ratio: 76.5%

Stability Metrics:
  Avg Temperature Change: 0.142°C
  Settling Time: 45 steps
```

**Happy Simulating!** 🎉
