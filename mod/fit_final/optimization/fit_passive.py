import os
from neuron import h
h.load_file("stdrun.hoc")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from MNTB_PN_fit import MNTB, nstomho
import MNTB_PN_myFunctions as mFun
import subprocess
import sys
import time
np.random.seed(1)
start_time = time.time()

# Load experimental data
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "experimental_data_P9_TeNT.csv"))
experimental_data = pd.read_csv(data_path)
vconverter = 1000
exp_currents = (experimental_data["Current"].values) * 1e-3  # Convert pA to nA
exp_steady_state_voltages = (experimental_data["SteadyStateVoltage"].values) * vconverter

# Define soma parameters
totalcap = 25  # Total membrane capacitance in pF
somaarea = (totalcap * 1e-6) / 1  # Convert to cm^2 assuming 1 µF/cm²

def nstomho(x):
    return (1e-9 * x / somaarea)  # Convert conductance to mho/cm²

# Create soma section
v_init = -70
h.celsius = 35
soma = h.Section(name='soma')
soma.L = 20  # Length in µm
soma.diam = 15  # Diameter in µm
soma.Ra = 150  # Axial resistance (Ohm*cm)
soma.cm = 1  # Membrane capacitance (µF/cm²)


# Insert passive leak channel
soma.insert('leak')

# Insert active conductances (Mainen & Sejnowski 1996)
soma.insert('HT_dth')  # Kv3 Potassium channel
#soma.gkhtbar_HT_dth = nstomho(200)
soma.insert('LT_dth')  # Kv1 Potassium channel
soma.insert('NaCh_nmb')  # Sodium channel
#soma.gnabar_NaCh_nmb = nstomho(200)
soma.insert('IH_dth')  # HCN channel
#soma.insert('ka')
soma.ek = -106.8
soma.ena = 62.77

# Create current clamp stimulus
st = h.IClamp(0.5)  # Location at the center of the soma
st.dur = 300  # Duration (ms)
st.delay = 10  # Delay before stimulus (ms)
h.dt = 0.02  #

# Set up recording vectors
v_vec = h.Vector()
t_vec = h.Vector()
v_vec.record(soma(0.5)._ref_v)
t_vec.record(h._ref_t)

h.v_init = v_init
mFun.custom_init(v_init)
h.tstop = st.delay + st.dur
h.continuerun(510)


gkht = 200
gna = 200
# Function to compute explained sum of squares (ESS)
def compute_ess(params):
    (gleak,
     gklt,
     gh,
     #gka,
     erev,
     gkht, gna) = params
    soma.g_leak = nstomho(gleak)
    soma.gkltbar_LT_dth = nstomho(gklt)
    soma.ghbar_IH_dth = nstomho(gh)
    #soma.gkabar_ka = nstomho(gka)
    soma.erev_leak = erev
    soma.gkhtbar_HT_dth = nstomho(gkht)
    soma.gnabar_NaCh_nmb = nstomho(gna)
    simulated_voltages = []

    for i in exp_currents:
        st.amp = i
        v_vec.resize(0)
        t_vec.resize(0)
        v_vec.record(soma(0.5)._ref_v)
        t_vec.record(h._ref_t)

        h.v_init = v_init
        mFun.custom_init(v_init)
        # h.finitialize(-70)
        h.run()

        # Compute steady-state voltage (average from 250-300 ms)
        time_array = np.array(t_vec)
        voltage_array = np.array(v_vec)
        steady_state_mask = (time_array >= 250) & (time_array <= 300)
        simulated_voltages.append(np.mean(voltage_array[steady_state_mask]))

    simulated_voltages = np.array(simulated_voltages)
    ess = np.sum((exp_steady_state_voltages - simulated_voltages) ** 2)
    return ess
print(f"Sampling rate: {1 / h.dt:.1f} kHz")

# Optimize g_leak, gkltbar_LT, ghbar_IH, gka, erev
initial_guess = [20,
                 100,
                 25,
                 #50,
                 -75,
                 gkht,
                 gna]  # Initial values in the middle of the range
bounds = [(5,50),
          (0,200),
          (0, 50),
          #(0,100),
          (-80,-50),
          (gkht*0.5,gkht*1.5),
          (gna*0.5,gna*1.5)]  # Set parameter bounds
result = minimize(compute_ess, initial_guess, bounds=bounds)

(opt_leak, opt_gklt, opt_gh,
 #opt_gka,
 opt_erev, opt_gkht,opt_gna) = result.x
print(f"Optimal Leak: {opt_leak}, Optimal LT: {opt_gklt}, Optimal ghbar_IH: {opt_gh}, "
      #f"Optimal gka: {opt_gka},"
      f"Optimal erev: {opt_erev}, Optimal gkht: {opt_gkht}, Optimal gna: {opt_gna}")

# Set optimized parameters
soma.g_leak = nstomho(opt_leak)
#soma.gkhtbar_HT = nstomho(optimal_gkht)
soma.gkltbar_LT_dth = nstomho(opt_gklt)
#soma.gnabar_NaCh = nstomho(optimal_gna)
soma.ghbar_IH_dth = nstomho(opt_gh)
soma.erev_leak = opt_erev
#soma.gkabar_ka = nstomho(opt_gka)
soma.gkhtbar_HT_dth = nstomho(opt_gkht)
soma.gnabar_NaCh_nmb = nstomho(opt_gna)

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

print(f"Sampling rate: {1 / h.dt:.1f} kHz")

# Save optimal values for the simulation
#project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
script_dir = os.path.dirname(os.path.abspath(__file__))
param_file_path = os.path.join(script_dir, "best_fit_params.txt")
with open(param_file_path, "w") as f:
    f.write(f"{opt_leak},{opt_gklt},{opt_gh},{opt_erev},{opt_gkht},{opt_gna}\n")

# Print the optimal parameters
print("\n✅ Optimal Parameters Found:")
print(f"Leak conductance: {opt_leak:.2f} nS")
print(f"KLT conductance:  {opt_gklt:.2f} nS")
print(f"IH conductance:   {opt_gh:.2f} nS")
#print(f"KA conductance:   {opt_gka:.2f} nS")
print(f"Leak reversal:    {opt_erev:.2f} mV")
print(f"KHT conductance:  {opt_gkht:.2f} nS")
print(f"Na conductance:   {opt_gna:.2f} nS")
# === Optional: save human-readable version with timestamped folder ===
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(os.getcwd(), "figures", f"BestFit_P9_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "best_fit_params_readable.txt"), "w") as f:
    f.write("📐 Best-Fit Parameters\n")
    f.write(f"Leak:  {opt_leak:.2f} nS\n")
    f.write(f"KLT:   {opt_gklt:.2f} nS\n")
    f.write(f"IH:    {opt_gh:.2f} nS\n")
    #f.write(f"KA:    {opt_gka:.2f} nS\n")
    f.write(f"ELeak: {opt_erev:.2f} mV\n")
    f.write(f"KHT:   {opt_gkht:.2f} nS\n")
    f.write(f"Na:    {opt_gna:.2f} nS\n")

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





