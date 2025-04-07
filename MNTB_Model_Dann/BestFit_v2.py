import os

from neuron import h

h.load_file("stdrun.hoc")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
np.random.seed(1)
start_time = time.time()
# Load experimental data
script_dir = os.path.dirname(os.path.abspath("/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron"))
os.chdir(script_dir)

experimental_data = pd.read_csv("/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/MNTB_Model_Dann/experimental_data_P9.csv")
exp_currents = (experimental_data["Current"].values) * 1e-3  # Convert pA to nA
exp_steady_state_voltages = experimental_data["SteadyStateVoltage"].values

# Define soma parameters
totalcap = 20  # Total membrane capacitance in pF
somaarea = (totalcap * 1e-6) / 1  # Convert to cm^2 assuming 1 µF/cm²

def nstomho(x):
    return (1e-9 * x / somaarea)  # Convert conductance to mho/cm²

script_dir = os.path.dirname(os.path.abspath("/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron"))
os.chdir(script_dir)

# Create soma section
soma = h.Section(name='soma')
soma.L = 15  # Length in µm
soma.diam = 15  # Diameter in µm
soma.Ra = 150  # Axial resistance (Ohm*cm)
soma.cm = 1  # Membrane capacitance (µF/cm²)
soma.v = -70  # Initial membrane potential (mV)

# Insert channels to fit in the simulation
soma.insert('leak')
soma.insert('LT')  # Kv1 Potassium channel
soma.insert('IH')  # HCN channel

# Insert active conductances (Mainen & Sejnowski 1996)
soma.insert('HT')  # Kv3 Potassium channel
soma.gkhtbar_HT = nstomho(300)

soma.insert('NaCh')  # Sodium channel
soma.gnabar_NaCh = nstomho(300)

soma.ek = -106.8
soma.ena = 62.77

# Create current clamp stimulus
st = h.IClamp(0.5)  # Location at the center of the soma
st.dur = 300  # Duration (ms)
st.delay = 10  # Delay before stimulus (ms)
h.tstop = 510  # Simulation stop time (ms)

# Set up recording vectors
v_vec = h.Vector()
t_vec = h.Vector()
v_vec.record(soma(0.5)._ref_v)
t_vec.record(h._ref_t)


# Function to compute explained sum of squares (ESS)
def compute_ess(params):
    gleak, gklt, gh, erev= params
    soma.g_leak = nstomho(gleak)
    soma.gkltbar_LT = nstomho(gklt)
    soma.ghbar_IH = nstomho(gh)
    soma.erev_leak = erev


    simulated_voltages = []

    for i in exp_currents:
        st.amp = i
        v_vec.resize(0)
        t_vec.resize(0)
        v_vec.record(soma(0.5)._ref_v)
        t_vec.record(h._ref_t)
        h.finitialize(-70)
        h.run()

        # Compute steady-state voltage (average from 250-300 ms)
        time_array = np.array(t_vec)
        voltage_array = np.array(v_vec)
        steady_state_mask = (time_array >= 250) & (time_array <= 300)
        simulated_voltages.append(np.mean(voltage_array[steady_state_mask]))

    simulated_voltages = np.array(simulated_voltages)
    ess = np.sum((exp_steady_state_voltages - simulated_voltages) ** 2)
    return ess


# Optimize g_leak, gkltbar_LT, and ghbar_IH
initial_guess = [10, 100, 25, -70]  # Initial values in the middle of the range
bounds = [(0,20),(0,200),(0, 50), (-90,-50)]  # Set parameter bounds
result = minimize(compute_ess, initial_guess, bounds=bounds)

optimal_leak, optimal_gklt, optimal_gh, optimal_erev = result.x
print(f"Optimal Leak: {optimal_leak}, Optimal LT: {optimal_gklt}, Optimal ghbar_IH: {optimal_gh}"
      f"Optima erev: {optimal_erev}")

# Set optimized parameters
soma.g_leak = nstomho(optimal_leak)
#soma.gkhtbar_HT = nstomho(optimal_gkht)
soma.gkltbar_LT = nstomho(optimal_gklt)
#soma.gnabar_NaCh = nstomho(optimal_gna)
soma.ghbar_IH = nstomho(optimal_gh)
soma.erev_leak = optimal_erev


# Compute best-fit simulation results
simulated_voltages = []
for i in exp_currents:
    st.amp = i
    v_vec.resize(0)
    t_vec.resize(0)
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(h._ref_t)
    h.finitialize(-70)
    h.run()

    time_array = np.array(t_vec)
    voltage_array = np.array(v_vec)
    steady_state_mask = (time_array >= 250) & (time_array <= 300)
    simulated_voltages.append(np.mean(voltage_array[steady_state_mask]))

end_time = time.time()
print(f"⏱️ minimize() took {end_time - start_time:.2f} seconds")
# Plot experimental vs. best-fit simulation data
plt.figure(figsize=(12, 7))
plt.scatter(exp_currents, exp_steady_state_voltages, color='r', label="Experimental Data")
plt.plot(exp_currents, simulated_voltages, 'o-', color='b', markersize=8, label="Best-Fit Simulation")
plt.xlabel("Injected Current (nA)", fontsize=16)
plt.ylabel("Steady-State Voltage (mV)", fontsize=16)
plt.title("Experimental vs. Simulated Steady-State Voltage", fontsize=16)
plt.legend()
plt.grid()
plt.show()
