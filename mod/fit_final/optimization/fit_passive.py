
'''fit_passive - Fit and optimize low-threshold potassium conductance (gklt), HCN-mediated conductance (gh),
and leak conductance (gkht) to experimental data using ess (explained sum of squares) optimization.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import time
import datetime
import sys
import subprocess
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42   # TrueType
from neuron import h
import MNTB_PN_myFunctions as mFun
from MNTB_PN_fit import MNTB, nstomho

# --- Initialization
np.random.seed(1)
start_time = time.time()

h.load_file("stdrun.hoc")
h.celsius = 35
h.dt = 0.02  # ms
v_init = -70  # mV

# --- Load experimental data
filename = "experimental_data_P9_TeNT_12172022_S3C1.csv"
file = filename.split(".")[0]
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", filename))
experimental_data = pd.read_csv(data_path)

vconverter = 1000
exp_currents = experimental_data["Current"].values * 1e-3  # pA to nA
exp_steady_state_voltages = experimental_data["SteadyStateVoltage"].values * vconverter  # V to mV

# --- Define soma parameters
totalcap = 25  # pF
somaarea = (totalcap * 1e-6) / 1  # cm² (assuming 1 µF/cm²)

def nstomho(x):
    return (1e-9 * x / somaarea)  # Conductance to mho/cm²

# --- Create soma section
soma = h.Section(name='soma')
soma.L = 20  # um
soma.diam = 15  # um
soma.Ra = 150  # Ohm.cm
soma.cm = 1  # uF/cm²

soma.insert('leak')
soma.insert('HT_dth')       # Kv3
soma.insert('LT_dth')       # Kv1
soma.insert('NaCh_nmb')     # Sodium
soma.insert('IH_nmb')       # HCN
soma.insert('ka')           # Kv4

soma.ek = -106.8  # mV
soma.ena = 62.77  # mV

# --- Create current clamp stimulus
st = h.IClamp(soma(0.5))
st.dur = 300  # ms
st.delay = 10  # ms

# --- Set up recording vectors
v_vec = h.Vector()
t_vec = h.Vector()
v_vec.record(soma(0.5)._ref_v)
t_vec.record(h._ref_t)

def run_simulation(current_injection):
    """Helper function to simulate voltage for a given current"""
    st.amp = current_injection
    v_vec.resize(0)
    t_vec.resize(0)

    h.v_init = v_init
    mFun.custom_init(v_init)
    h.tstop = st.delay + st.dur
    h.continuerun(h.tstop)

    time_array = np.array(t_vec)
    voltage_array = np.array(v_vec)
    steady_state_mask = (time_array >= 250) & (time_array <= 300)
    steady_voltage = np.mean(voltage_array[steady_state_mask])
    return steady_voltage

# --- Optimization target function
def compute_ess(params):
    gleak, gklt, gh, erev, gkht, gna, gka = params
    soma.g_leak = nstomho(gleak)
    soma.gkltbar_LT_dth = nstomho(gklt)
    soma.ghbar_IH_nmb = nstomho(gh)
    soma.erev_leak = erev
    soma.gkhtbar_HT_dth = nstomho(gkht)
    soma.gnabar_NaCh_nmb = nstomho(gna)
    soma.gkabar_ka = nstomho(gkht)
    simulated_voltages = np.array([run_simulation(i) for i in exp_currents])
    ess = np.sum((exp_steady_state_voltages - simulated_voltages) ** 2)
    return ess
print("Running optimization...")
# --- Initial parameter guesses and bounds
'''gkht, gna, and gka are optimized in the AP fit.'''
gkht = 10
gna = 10
gka = 10
initial_guess = [6.98,12.42,4.68,-76.66, gkht, gna, gka]
bounds = [(5.44, 7.48), (11, 13), (3, 5), (-80.00, -60.00), (gkht*0.999, gkht*1.001), (gna*0.999, gna*1.001), (gka*0.999, gka*1.001)]

# --- Run optimization
result = minimize(compute_ess, initial_guess, bounds=bounds)
opt_leak, opt_gklt, opt_gh, opt_erev, opt_gkht, opt_gna, opt_gka  = result.x

print(f'Optimal results got')
# --- Apply optimized parameters
soma.g_leak = nstomho(opt_leak)
soma.gkltbar_LT_dth = nstomho(opt_gklt)
soma.ghbar_IH_nmb = nstomho(opt_gh)
soma.erev_leak = opt_erev
soma.gkhtbar_HT_dth = nstomho(opt_gkht)
soma.gnabar_NaCh_nmb = nstomho(opt_gna)
soma.gkabar_ka = nstomho(opt_gka)

# --- Compute final simulation with best-fit parameters
print("Running final simulation with best-fit parameters...")
simulated_voltages = np.array([run_simulation(i) for i in exp_currents])

# --- Save optimal parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
param_file_path = os.path.join(script_dir, "best_fit_params.txt")
with open(param_file_path, "w") as f:
    f.write(f"{opt_leak},{opt_gklt},{opt_gh},{opt_erev},{opt_gkht},{opt_gna}, {opt_gka}\n")

# --- Output results
print("\n✅ Optimal Parameters Found:")
print(f"Leak conductance: {opt_leak:.2f} nS")
print(f"KLT conductance:  {opt_gklt:.2f} nS")
print(f"IH conductance:   {opt_gh:.2f} nS")
print(f"Leak reversal:    {opt_erev:.2f} mV")
print(f"KHT conductance:  {opt_gkht:.2f} nS")
print(f"Na conductance:   {opt_gna:.2f} nS")


# --- Optional: Save human-readable parameter file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(os.getcwd(), "..", "figures", f"fit_passive_{file}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "best_fit_params_readable.txt"), "w") as f:
    f.write("📐 Best-Fit Parameters\n")
    f.write(f"Leak:   {opt_leak:.2f} nS\n")
    f.write(f"KLT:    {opt_gklt:.2f} nS\n")
    f.write(f"IH:     {opt_gh:.2f} nS\n")
    f.write(f"ELeak:  {opt_erev:.2f} mV\n")
    f.write(f"KHT*:   {opt_gkht:.2f} nS\n")
    f.write(f"Na*:    {opt_gna:.2f} nS\n")
    f.write(f"KA*:    {opt_gka:.2f} nS\n")
    f.write("\n*These parameters are optimized in the AP fit.\n")

# --- 📈 Plot experimental vs simulated steady-state voltages
x_min = -0.15
x_max = 0.4
y_min = -110
y_max = -40
plt.figure(figsize=(8,5))
plt.scatter(exp_currents, exp_steady_state_voltages, color='r', label="Experimental Data")
plt.plot(exp_currents, simulated_voltages, 'o-', color='b', alpha=0.5, markersize=8, label="Best-Fit Simulation")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel("Injected Current (nA)", fontsize=16)
plt.ylabel("Steady-State Voltage (mV)", fontsize=16)
plt.title("Experimental vs. Simulated Steady-State Voltage", fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "passive_fit.pdf"),format='pdf', transparent=True,dpi=300)
plt.show()

print("📊 Saved passive fit plot.")

# --- 🔥 Calculate Input Resistance (slope around 0)
mask = (exp_currents >= -0.020) & (exp_currents <= 0.020)
selected_currents = exp_currents[mask]
selected_exp_voltages = exp_steady_state_voltages[mask]
selected_sim_voltages = simulated_voltages[mask]

# Experimental input resistance
coeff_exp = np.polyfit(selected_currents, selected_exp_voltages, 1)
rin_exp_mohm = coeff_exp[0]  # mV/nA = MOhm

# Simulated input resistance
coeff_sim = np.polyfit(selected_currents, selected_sim_voltages, 1)
rin_sim_mohm = coeff_sim[0]

print(f"🔍 Experimental Input Resistance (±20pA): {rin_exp_mohm:.2f} MΩ")
print(f"🔍 Simulated Input Resistance (±20pA): {rin_sim_mohm:.2f} MΩ")

# Plot local linear fits
plt.figure(figsize=(8, 5))
plt.plot(selected_currents, selected_exp_voltages, 'o', label="Experimental")
plt.plot(selected_currents, coeff_exp[0]*selected_currents + coeff_exp[1], '-', label=f"Exp Fit: {rin_exp_mohm:.2f} MΩ")
plt.plot(selected_currents, selected_sim_voltages, 's', label="Simulated")
plt.plot(selected_currents, coeff_sim[0]*selected_currents + coeff_sim[1], '--', label=f"Sim Fit: {rin_sim_mohm:.2f} MΩ")
plt.xlabel("Injected Current (nA)")
plt.ylabel("Steady-State Voltage (mV)")
plt.title("Input Resistance Estimation (±20pA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "input_resistance_fit.png"), dpi=300)
plt.show()

end_time = time.time()
print(f"⏱️ Total script time: {end_time - start_time:.2f} seconds")
