import os
from neuron import h
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MNTB_PN_myFunctions as mFun
import subprocess
import sys
import time


from MNTB_PN_mc import PN
from scipy.optimize import minimize, differential_evolution
import MNTB_PN_mc

h.load_file("stdrun.hoc")
np.random.seed(1)
start_time = time.time()

# Always use the base project directory for parameter file
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # one level up
param_file_path = os.path.join(project_root, "best_fit_params_mc.txt")

# Load experimental data
script_dir = os.path.dirname(os.path.abspath("/"))
os.chdir(script_dir)

experimental_data = pd.read_csv("/MNTB_Model_Dann/experimental_data_P9.csv")
exp_currents = (experimental_data["Current"].values) * 1e-3  # Convert pA to nA
exp_steady_state_voltages = experimental_data["SteadyStateVoltage"].values


totalcap = 25  # Total membrane capacitance in pF
somaarea = (totalcap * 1e-6) / 1  # Convert to cm^2 assuming 1 µF/cm²

def nstomho(x):
    return (1e-9 * x / somaarea)  # Convert conductance to mho/cm²

script_dir = os.path.dirname(os.path.abspath("/"))
os.chdir(script_dir)

# Create soma section
# v_init = -77
# soma = h.Section(name='soma')
# soma.L = 15  # Length in µm
# soma.diam = 15  # Diameter in µm
# soma.Ra = 150  # Axial resistance (Ohm*cm)
# soma.cm = 1  # Membrane capacitance (µF/cm²)
# soma.v = -70  # Initial membrane potential (mV)
#
# # Insert passive leak channel
# soma.insert('leak')
# #soma.g_leak = nstomho(5.5)
# #soma.erev_leak = -70
#
# # Insert active conductances (Mainen & Sejnowski 1996)
# soma.insert('HT')  # Kv3 Potassium channel
# soma.gkhtbar_HT = nstomho(300)
# soma.insert('LT')  # Kv1 Potassium channel
# soma.insert('NaCh')  # Sodium channel
# soma.gnabar_NaCh = nstomho(300)
# soma.insert('IH')  # HCN channel

############################# set first conductances #############################################
ek = -106.8
ena = 62.77
gna = 1014
gklt = 160
gkht = 1503
erev = -77
leakg = 12
gh = 20
############################################# MNTB_PN file imported ####################################################
# totalcap = 25
# somaarea = (totalcap * 1e-6) / 1  # in cm²

AIS_diam = 2
AIS_L = 25
dend_diam = 3
dend_L = 80

AISarea = np.pi * AIS_diam * AIS_L * 1e-8
dendarea = np.pi * dend_diam * dend_L * 1e-8
h.celsius = 35
my_cell = PN(0, somaarea, AISarea, dendarea, erev, ena, ek, leakg, gna, gh, gklt, gkht)

#######################################################################################################################

# stim = h.IClamp(my_cell.soma(0.5))
# stim_traces = h.Vector().record(stim._ref_i)
# soma_v = h.Vector().record(my_cell.soma(0.5)._ref_v)
# t = h.Vector().record(h._ref_t)
soma = my_cell.soma
AIS = my_cell.AIS
dend = my_cell.dend
# Create current clamp stimulus
st = h.IClamp(soma(0.5))  # Location at the center of the soma
st.dur = 300  # Duration (ms)
st.delay = 10  # Delay before stimulus (ms)
# h.tstop = 510  # Simulation stop time (ms)
h.dt = 0.02  # 0.01 ms time step → 100 kHz sampling
# Set up recording vectors
v_vec = h.Vector()
t_vec = h.Vector()
v_vec.record(soma(0.5)._ref_v)
t_vec.record(h._ref_t)

v_init = -77
h.v_init = v_init
mFun.custom_init(v_init)
h.tstop = st.delay + st.dur
h.continuerun(510)

# Function to compute explained sum of squares (ESS)
def compute_ess(params):
    gleak, gklt, gh, erev= params
    soma.g_leak = nstomho(gleak)
    soma.ghbar_IH_dth =  nstomho(gh)
    dend.ghbar_IH_dth =  nstomho(gh)

    AIS.gkltbar_LT_dth = nstomho(gklt)


    soma.erev_leak = erev
    AIS.erev_leak = erev
    dend.erev_leak = erev

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


# Optimize g_leak, gkltbar_LT, and ghbar_IH
# initial_guess = [10, 100, 30, -70]  # Initial values in the middle of the range
bounds = [(0,20),(1,400),(1, 50), (-90,-50)]  # Set parameter bounds

result = differential_evolution(compute_ess, bounds, strategy='best1bin', maxiter=20, popsize=15,polish=True)

# print(f"result_global: {result_global.x}")

# result = minimize(compute_ess, result_global, bounds=bounds, method='L-BFGS-B', options={'maxiter': 200})

print(result.x)

optimal_leak, optimal_gklt, optimal_gh, optimal_erev = result.x
print(f"Optimal Leak: {optimal_leak}, Optimal LT: {optimal_gklt}, Optimal ghbar_IH: {optimal_gh}"
      f"Optima erev: {optimal_erev}")

# Set optimized parameters
my_cell = PN(0, somaarea, AISarea, dendarea, optimal_erev, ena, ek, optimal_leak, gna, gh, optimal_gklt, gkht)

# # Set optimized parameters
# soma.g_leak = nstomho(opt_leak)
# #soma.gkhtbar_HT = nstomho(optimal_gkht)
# soma.gkltbar_LT = nstomho(opt_gklt)
# #soma.gnabar_NaCh = nstomho(optimal_gna)
# soma.ghbar_IH = nstomho(opt_gh)
# soma.erev_leak = opt_erev


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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
param_file_path = os.path.join(project_root, "best_fit_params_mc.txt")
with open(param_file_path, "w") as f:
    f.write(f"{optimal_leak},{optimal_gklt},{optimal_gh},{optimal_erev}")

# Print the optimal parameters
print("\n✅ Optimal Parameters Found:")
print(f"Leak conductance: {optimal_leak:.2f} nS")
print(f"KLT conductance:  {optimal_gklt:.2f} nS")
print(f"IH conductance:   {optimal_gh:.2f} nS")
print(f"Leak reversal:    {optimal_erev:.2f} mV")

# === Optional: save human-readable version with timestamped folder ===
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(os.getcwd(), "figures", f"BestFit_P9_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "best_fit_params_readable_mc.txt"), "w") as f:
    f.write("📐 Best-Fit Parameters\n")
    f.write(f"Leak:  {optimal_leak:.2f} nS\n")
    f.write(f"KLT:   {optimal_gklt:.2f} nS\n")
    f.write(f"IH:    {optimal_gh:.2f} nS\n")
    f.write(f"ELeak: {optimal_erev:.2f} mV\n")

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

# Call the MNTB_PN_simulation script
print("\n🚀 Launching simulation with best-fit parameters...\n")

# Ensure the correct working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
simulation_script = os.path.join(script_dir, "MNTB_PN_simulation_from_BestFit_v2_mc.py")

result = subprocess.run(
    [sys.executable, simulation_script],
    cwd=script_dir,
    capture_output=True,
    text=True
)

# Show output
print(result.stdout)
if result.stderr:
    print("⚠️ Simulation error:")
    print(result.stderr)




