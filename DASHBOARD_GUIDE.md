# Dashboard Usage Guide

## ✅ Dashboard is Now Working!

The interactive Streamlit dashboard is fully functional and running.

---

## 🚀 How to Run the Dashboard

### Option 1: PowerShell Script (Recommended)
```powershell
cd "E:\Sidharth\Websites\Data Centers"
.\run_dashboard.ps1
```

### Option 2: Manual Launch
```powershell
cd "E:\Sidharth\Websites\Data Centers"
.\venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD"
streamlit run frontend/dashboard.py
```

---

## 🌐 Access the Dashboard

Once running, open your browser and navigate to:
- **Local URL**: http://localhost:8501
- **Network URL**: http://192.168.x.x:8501 (shown in terminal)

The dashboard will automatically open in your default browser.

---

## 🎛️ Dashboard Features

### 1. **Digital Twin Simulation Mode**
- Real-time thermal heatmap visualization
- Live workload and cooling action display
- Temperature distribution across data center racks
- Safety threshold indicators
- Step-by-step simulation control

### 2. **Real System Monitor Mode**
- Live laptop/system sensor readings
- CPU temperature monitoring
- CPU usage tracking
- Fan speed monitoring (if available)
- Real-time metrics dashboard
- Historical trends and statistics

### 3. **Controller Comparison Mode**
- Side-by-side RL vs PID comparison
- Parallel execution of both controllers
- Synchronized thermal state visualization
- Real-time performance metrics:
  - Average temperature
  - Energy consumption
  - Temperature violations
  - Hotspot detection
- Cumulative statistics comparison

---

## 🎮 Using the Dashboard

### Digital Twin Mode
1. Select "🔬 Digital Twin Simulation" from sidebar
2. Choose controller type (RL or PID)
3. For RL: Select checkpoint file (defaults to `dqn_final.pth`)
4. Click "▶️ Start Simulation" to begin
5. Use "⏹️ Stop Simulation" to pause
6. Click "🔄 Reset" to restart with new settings

### Real System Monitor
1. Select "💻 Real System Monitor" from sidebar
2. Set refresh interval (1-10 seconds)
3. Click "Start Monitoring" to begin
4. View live sensor data and historical trends
5. Click "Stop Monitoring" to pause

### Comparison Mode
1. Select "⚖️ Controller Comparison" from sidebar
2. Set checkpoint path for RL agent
3. Click "Start Comparison"
4. Watch both controllers operate simultaneously
5. Compare real-time metrics and performance

---

## 🔧 Troubleshooting

### Dashboard Won't Start
**Problem**: Import errors or module not found
**Solution**: 
```powershell
cd "E:\Sidharth\Websites\Data Centers"
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Port Already in Use
**Problem**: Port 8501 is occupied
**Solution**: 
```powershell
# Find and kill process on port 8501
netstat -ano | Select-String "8501"
taskkill /PID <PID_NUMBER> /F

# Or use a different port
streamlit run frontend/dashboard.py --server.port 8502
```

### Environment Not Activating
**Problem**: Virtual environment not found
**Solution**: 
```powershell
# Create new virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Browser Doesn't Open Automatically
**Problem**: Browser not launching
**Solution**: Manually navigate to http://localhost:8501

---

## 📊 What You'll See

### Heatmaps
- **Temperature Heatmap**: Shows thermal distribution (20-85°C range)
- **Workload Heatmap**: CPU utilization per rack (0-100%)
- **Cooling Heatmap**: Cooling effort per zone (0-1 scale)
- Color-coded visualization (cool blue → hot red)

### Metrics
- **Energy**: Average cooling level, peak cooling, total consumption
- **Temperature**: Average, max, violations, comfort zone ratio
- **Safety**: Violation count, hotspot detection, anomaly alerts
- **Performance**: Response time, stability, settling time

### Real-time Updates
- Simulations update every step (configurable interval)
- Sensor monitoring refreshes at set intervals
- Metrics calculated continuously
- Plots and statistics auto-update

---

## ⚙️ Configuration

Edit `config.yaml` to customize:
- Grid size (racks layout)
- Thermal parameters (heat transfer coefficients)
- Safety thresholds (max temperature, rate limits)
- Workload patterns (synthetic generation)
- RL hyperparameters (learning rate, network size)
- PID gains (Kp, Ki, Kd values)

Changes require dashboard restart to take effect.

---

## 🛑 Stopping the Dashboard

Press **Ctrl+C** in the terminal where Streamlit is running.

---

## 💡 Tips for Best Experience

1. **Use RL Agent**: Load the trained checkpoint (`checkpoints/dqn_final.pth`) for best performance
2. **Comparison Mode**: Great for research presentations and demonstrations
3. **Adjust Refresh Rate**: Lower intervals (1-2s) for smooth visualization
4. **Monitor Real System**: Useful for correlating simulation with actual hardware
5. **Screenshot Feature**: Use browser tools to capture dashboard states for reports

---

## 📝 Notes

- Dashboard runs in your local browser (no internet required)
- All simulations are real-time computations (no dummy data)
- Checkpoint files must exist before loading RL agent
- System monitoring requires sensor access (may vary by hardware)
- Multiple users can view dashboard on same network

---

## 🎯 Current Status

✅ **FULLY OPERATIONAL**
- All dependencies installed in virtual environment
- Streamlit caching issues resolved
- NumPy compatibility fixed
- Dashboard running on port 8501
- Ready for demonstration and research use

---

**Enjoy exploring your AI-powered data center cooling optimization system!** ❄️🤖
