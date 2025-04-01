import os
from neuron import h
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import find_peaks

h.load_file("stdrun.hoc")

# Load experimental data
experimental_data = pd.read_csv("experimental_data_example.csv")
exp_currents = experimental_data["Current"].values
exp_steady_state_voltages = experimental_data["SteadyStateVoltage"].values

# Get experimental AP features at 0.3 nA (rheobase)
chosen_current_for_ap = 0.3
row = experimental_data[experimental_data["Current"] == chosen_current_for_ap].iloc[0]
exp_ap_peak_voltage = row["AP_Peak"]
exp_ap_threshold = row["AP_Threshold"]
exp_ap_width = row["AP_Width"]

# Define soma parameters
totalcap = 20
somaarea = (totalcap * 1e-6)

def nstomho(x):
    return (1e-9 * x / somaarea)

# Create soma
soma = h.Section(name='soma')
soma.L = 15
soma.diam = 15
soma.Ra = 150
soma.cm = 1
soma.v = -70

soma.insert('leak')
soma.insert('HT')
soma.insert('LT')
soma.insert('NaCh')
soma.insert('IH')

soma.ek = -106.8
soma.ena = 62.77

# Stimulus
st = h.IClamp(0.5)
st.dur = 300
st.delay = 10
h.tstop = 510

# Recording
v_vec = h.Vector()
t_vec = h.Vector()
v_vec.record(soma(0.5)._ref_v)
t_vec.record(h._ref_t)

# Subthreshold component
def compute_ess_sub(params):
    gleak, gklt, gh, erev = params
    soma.g_leak = nstomho(gleak)
    soma.gkltbar_LT = nstomho(gklt)
    soma.ghbar_IH = nstomho(gh)
    soma.erev_leak = erev

    simulated_voltages = []
    for i in exp_currents:
        st.amp = i
        v_vec.resize(0)
        t_vec.resize(0)
        h.finitialize(-70)
        h.run()

        voltage_array = np.array(v_vec)
        if np.max(voltage_array) > 100 or np.min(voltage_array) < -150:
            return 1e6

        time_array = np.array(t_vec)
        steady_state_mask = (time_array >= 250) & (time_array <= 300)
        simulated_voltages.append(np.mean(voltage_array[steady_state_mask]))

    return np.sum((exp_steady_state_voltages - np.array(simulated_voltages)) ** 2)

# AP component
def compute_ess_ap(params):
    gna, gkht = params
    soma.gnabar_NaCh = nstomho(gna)
    soma.gkhtbar_HT = nstomho(gkht)

    st.amp = chosen_current_for_ap
    v_vec.resize(0)
    t_vec.resize(0)
    h.finitialize(-70)
    h.run()

    voltage_array = np.array(v_vec)
    if np.max(voltage_array) > 100 or np.min(voltage_array) < -150:
        return 1e6

    time_array = np.array(t_vec)
    peaks, _ = find_peaks(voltage_array, height=0)
    if len(peaks) < 1:
        return 1e6

    peak_voltage = voltage_array[peaks[0]]
    dvdt = np.gradient(voltage_array, time_array)
    try:
        threshold_idx = np.argmax(dvdt > 20)
        ap_threshold = voltage_array[threshold_idx]
    except:
        ap_threshold = -40

    half_max = (peak_voltage + ap_threshold) / 2
    crossings = np.where(np.diff(voltage_array > half_max))[0]
    if len(crossings) >= 2:
        ap_width = time_array[crossings[1]] - time_array[crossings[0]]
    else:
        ap_width = 1.0

    return ((peak_voltage - exp_ap_peak_voltage)**2 +
            (ap_threshold - exp_ap_threshold)**2 +
            (ap_width - exp_ap_width)**2)

# Combined cost
def combined_cost(params):
    gleak, gklt, gh, erev, gna, gkht = params
    cost_sub = compute_ess_sub([gleak, gklt, gh, erev])
    cost_ap = compute_ess_ap([gna, gkht])
    return cost_sub + cost_ap

# Optimization
initial_guess = [10, 100, 25, -70, 300, 300]
bounds = [(0, 20), (0, 200), (0, 50), (-90, -50), (100, 1000), (50, 800)]
result = minimize(combined_cost, initial_guess, bounds=bounds)
optimal_params = result.x
gleak, gklt, gh, erev, gna, gkht = optimal_params

# Apply optimal parameters
soma.g_leak = nstomho(gleak)
soma.gkltbar_LT = nstomho(gklt)
soma.ghbar_IH = nstomho(gh)
soma.erev_leak = erev
soma.gnabar_NaCh = nstomho(gna)
soma.gkhtbar_HT = nstomho(gkht)

# Final IV curve
simulated_voltages = []
for i in exp_currents:
    st.amp = i
    v_vec.resize(0)
    t_vec.resize(0)
    h.finitialize(-70)
    h.run()

    time_array = np.array(t_vec)
    voltage_array = np.array(v_vec)
    steady_state_mask = (time_array >= 250) & (time_array <= 300)
    simulated_voltages.append(np.mean(voltage_array[steady_state_mask]))

# Plot
plt.figure(figsize=(12, 7))
plt.scatter(exp_currents, exp_steady_state_voltages, color='r', label="Experimental Data")
plt.plot(exp_currents, simulated_voltages, 'o-', color='b', markersize=8, label="Best-Fit Simulation")
plt.xlabel("Injected Current (nA)", fontsize=16)
plt.ylabel("Steady-State Voltage (mV)", fontsize=16)
plt.title("Experimental vs. Simulated Steady-State Voltage", fontsize=16)
plt.legend()
plt.grid()
plt.show()
