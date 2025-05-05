import os
from neuron import h
h.load_file("stdrun.hoc")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from MNTB_PN import MNTB
import MNTB_PN_myFunctions as mFun

# ========== Load Experimental Data ==========
script_dir = os.path.dirname(os.path.abspath(__file__))
exp_data_path = os.path.join(script_dir, "..", "MNTB_Model_Dann", "experimental_data_P9.csv")
experimental_data = pd.read_csv(exp_data_path)

exp_currents = experimental_data["Current"].values * 1e-3  # Convert pA to nA
exp_steady_state_voltages = experimental_data["SteadyStateVoltage"].values

# ========== Define soma parameters ==========
totalcap = 20  # pF
somaarea = (totalcap * 1e-6) / 1  # cm^2

def nstomho(x, somaarea):
    return (1e-9 * x / somaarea)

# ========== Create soma section for optimization ==========
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

# IClamp stimulus
st = h.IClamp(0.5)
st.dur = 300
st.delay = 10
h.tstop = 510

# Recording vectors
v_vec = h.Vector()
t_vec = h.Vector()
v_vec.record(soma(0.5)._ref_v)
t_vec.record(h._ref_t)

# ========== Run Optimization ==========
initial_guess = [10, 100, 25, -70]
bounds = [(0,20),(0,200),(0,50), (-90,-50)]

result = minimize(
    mFun.compute_ess, initial_guess,
    args=(soma, nstomho, somaarea, exp_currents, exp_steady_state_voltages, st, t_vec, v_vec),
    bounds=bounds
)

optimal_leak, optimal_gklt, optimal_gh, optimal_erev = result.x

print("\n✅ Optimal Parameters Found:")
print(f"Leak conductance: {optimal_leak:.2f} nS")
print(f"KLT conductance:  {optimal_gklt:.2f} nS")
print(f"IH conductance:   {optimal_gh:.2f} nS")
print(f"Leak reversal:    {optimal_erev:.2f} mV")

# Save parameters
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(script_dir, "figures", f"BestFit_P9_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "best_fit_params_readable.txt"), "w") as f:
    f.write("📐 Best-Fit Parameters\n")
    f.write(f"Leak:  {optimal_leak:.2f} nS\n")
    f.write(f"KLT:   {optimal_gklt:.2f} nS\n")
    f.write(f"IH:    {optimal_gh:.2f} nS\n")
    f.write(f"ELeak: {optimal_erev:.2f} mV\n")

# ========== Create MNTB Cell with Best-Fit Parameters ==========
my_cell = MNTB(0, somaarea, optimal_erev, optimal_leak, 62.77, 300, optimal_gh,
               optimal_gklt, 300, -106.8)

stim = h.IClamp(my_cell.soma(0.5))
stim_traces = h.Vector().record(stim._ref_i)
soma_v = h.Vector().record(my_cell.soma(0.5)._ref_v)
t = h.Vector().record(h._ref_t)

amps = np.round(np.arange(-0.100, 0.6, 0.020), 3)
stimdelay, stimdur, totalrun, v_init = 100, 300, 1000, -70
t_min, t_max = stimdelay + stimdur - 60, stimdelay + stimdur - 10

rmp = None
average_soma_values = np.array([])
ap_counts, ap_times, trace_data_apc = [], [], []

netcon = h.NetCon(my_cell.soma(0.5)._ref_v, None, sec=my_cell.soma)
netcon.threshold = 0
spike_times = h.Vector()
netcon.record(spike_times)

first_trace_detected = False
first_trace_data = None
fig1, ax1 = plt.subplots()
axin = ax1.inset_axes([0.6, 0.1, 0.2, 0.2])

for amp in amps:
    v_init = mFun.custom_init(v_init)
    soma_vals, stim_vals, t_vals = mFun.run_simulation(amp, stim, soma_v, t, totalrun,
                                                       stimdelay, stimdur, stim_traces)
    _, _, average_soma_values = mFun.avg_ss_values(soma_vals, t_vals, t_min, t_max, average_soma_values)
    _, _, ap_counts, ap_times = mFun.count_spikes(ap_counts, stimdelay, stimdur,
                                                  spike_times, ap_counts, ap_times)
    ax1.plot(t_vals, soma_vals, color='red', linewidth=0.5)
    axin.plot(t_vals, stim_vals, color='black', linewidth=0.5)
    spike_times.clear()

for amp, avg in zip(amps, average_soma_values):
    if amp == 0:
        rmp = avg

slopes = np.array([])
for i in range(1, len(amps)):
    slope = (average_soma_values[i] - average_soma_values[i - 1]) / (amps[i] - amps[i - 1]) / 1000
    slopes = np.append(slopes, np.round(slope, 3))

slope_range_index = np.where((amps[:-1] == -0.02) & (amps[1:] == 0))[0]
input_resistance = slopes[slope_range_index[0]] if len(slope_range_index) > 0 else None

annotation_text = f"""RMP: {rmp}mV
Rin: {input_resistance} GOhms
gLeak: {optimal_leak}nS
gNa: 300nS
gIH: {optimal_gh}nS
gKLT: {optimal_gklt}nS
gKHT: 300nS
ELeak: {optimal_erev}mV
Ek: -106.8mV
ENa: 62.77mV"""
ax1.annotate(annotation_text, xy=(600, -80), xytext=(600, -50),
             fontsize=10, bbox=dict(boxstyle="round", facecolor="lightyellow"))

fig1.savefig(os.path.join(output_dir, "trace_plot.png"))

# Rheobase Plot
fig_rheo, ax_rheo = plt.subplots()
netcon.threshold = -10
spike_times = h.Vector()
netcon.record(spike_times)

for amp in amps:
    v_init = mFun.custom_init(v_init)
    soma_vals, t_vals = mFun.run_simulation(amp, stim, soma_v, t, totalrun, stimdelay, stimdur)
    n_spikes = sum(stimdelay <= time <= stimdelay + stimdur for time in spike_times)
    trace_data_apc.append((t_vals.copy(), soma_vals.copy(), amp, n_spikes))
    if not first_trace_detected and n_spikes > 0:
        first_trace_data = (t_vals.copy(), soma_vals.copy(), amp)
        first_trace_detected = True
        ap_data = mFun.analyze_AP(t_vals, soma_vals)
    spike_times.clear()

for t_vals, soma_vals, amp, n_spikes in trace_data_apc:
    ax_rheo.plot(t_vals, soma_vals, color='red' if n_spikes == 0 else 'gray', linewidth=0.5)
if first_trace_data:
    t_vals, soma_vals, amp = first_trace_data
    ax_rheo.plot(t_vals, soma_vals, color='black', label=f'Rheobase {amp * 1000} pA')
    ax_rheo.legend()

fig_rheo.savefig(os.path.join(output_dir, "rheobase_trace.png"))

if ap_data:
    print("AP Analysis:")
    for key, val in ap_data.items():
        print(f"{key}: {val:.2f}")

    fig_ap, ax_ap = plt.subplots()
    ax_ap.plot(t_vals, soma_vals, color='black', label="Voltage Trace")
    ax_ap.scatter(ap_data["spike time"], ap_data["peak"], color='red', label="Peak")
    ax_ap.axhline(ap_data["AHP"], linestyle='--', color='purple', label="AHP")
    ax_ap.scatter(t_vals[np.where(soma_vals == ap_data["threshold"])[0][0]], ap_data["threshold"],
                  color='blue', label="Threshold")
    ax_ap.legend()
    fig_ap.savefig(os.path.join(output_dir, "AP_features.png"))

print(f"All plots saved to {output_dir}")
