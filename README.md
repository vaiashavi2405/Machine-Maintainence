# ⚙️ Predictive Machine Maintenance Using Markov Process

## 📌 Project Description
Predictive Machine Maintenance using Markov Process is a Python-based project that predicts possible machine conditions using transition probabilities. It helps estimate machine failure risks and supports preventive maintenance planning.

The project uses a Markov transition matrix and a Streamlit dashboard to visualize machine-state probabilities, simulate future conditions, and provide maintenance recommendations.

## 🎯 Objectives
- Predict possible future machine conditions.
- Estimate machine failure probability.
- Analyze machine-state transitions using a Markov Process.
- Visualize forecasts using interactive graphs.
- Recommend suitable maintenance actions.

## 🛠️ Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Plotly
- Streamlit
- Pickle

## 📊 Machine States
The model considers four machine conditions:
1. Operational
2. Minor Wear
3. Major Fault
4. Failure

## ✨ Features
- Machine condition selection.
- Adjustable forecast horizon.
- Multiple simulation runs.
- Failure probability estimation.
- Interactive line and pie charts.
- Markov transition matrix visualization.
- Maintenance recommendations.
- Pickle model saving and loading.

## 📁 Project Structure
```text
Predictive-Machine-Maintenance/
│
├── app.py
├── predictive_machine_maintenance_with_pickle.py
├── machine_maintenance_markov_model.pkl
└── README.md
```

## 🚀 Installation and Execution

### Step 1: Install Python Libraries
```bash
pip install numpy pandas matplotlib plotly streamlit
```

### Step 2: Generate the Pickle Model
Run the Python file that creates and saves the Markov model:

```bash
python predictive_machine_maintenance_with_pickle.py
```

This generates `machine_maintenance_markov_model.pkl`.

### Step 3: Run the Streamlit Application
```bash
streamlit run "app (7).py"
```

If you rename `app (7).py` to `app.py`, run:

```bash
streamlit run app.py
```

### Step 4: Open the Dashboard
Open the local URL displayed in the terminal by Streamlit.

## 🔍 How It Works
1. The Markov transition matrix defines the probability of moving between machine states.
2. The user selects the current machine condition.
3. The application calculates future state probabilities.
4. Monte Carlo simulation generates possible machine-state paths.
5. The dashboard displays failure risk and maintenance recommendations.

## 📈 Applications
- Industrial machine monitoring
- Preventive maintenance planning
- Manufacturing equipment analysis
- Machine failure risk estimation
- Industrial decision support systems

## ⚠️ Limitations
The current model uses predefined transition probabilities rather than live sensor data or actual machine maintenance records. Predictions are probabilistic estimates and should be validated before making real maintenance decisions.

## 🔮 Future Scope
- Integration with real-time IoT sensors.
- Use of historical maintenance datasets.
- Machine-learning-based failure prediction.
- Automated maintenance alerts.
- Cloud deployment of the dashboard.

## 👩‍💻 Author
**Vaishnavi Dawkhar**

## 📝 Conclusion
This project demonstrates how a Markov Process can estimate machine-state transitions and support predictive maintenance planning. The interactive Streamlit dashboard makes it easier to understand machine conditions, assess potential failure risks, and plan maintenance activities.
