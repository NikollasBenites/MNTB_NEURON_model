from neuron import h
h.load_file("stdrun.hoc")

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mntb_neuron import MNTB
from simulation_setup import setup_simulation
from plotting import plot_traces, plot_inset
from calculate_input_resistance import calculate_input_resistance as Rin

# Set working directory
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
print("Current working directory:", os.getcwd())

# Define parameters
annotation = False
totalcap = 30
somaarea = (totalcap * 1e-6) / 1
v_rest = 0
revleak = -70
revk = -80
revna = 50
reveh = -45
leakg = 3
nag = 210
kltg = 5
khtg = 150
ihg = 35
kag = 3
stim_amps = np.round(np.arange(-0.1, -0.320, 0.020), 3)  # Amps from -100 pA to 300 pA in 20 pA steps
stimdelay = 100
stimdur = 300

filename = f'Ek_{revk}_L{leakg}_Na{nag}_IH{ihg}_KLT{kltg}_KHT{khtg}_KA_{kag}.png' # change the name according

# Create neuron
my_cell = MNTB(0, somaarea, revleak, leakg, revna, nag, ihg, kltg, khtg, kag, revk)
# Create stimulus and record vectors
stim = h.IClamp(my_cell.soma(0.5))
t = h.Vector().record(h._ref_t)
stim_traces = []
soma_values = []
t_values = []
# Setup plotting
fig1, ax1 = plt.subplots()
axin = ax1.inset_axes([0.6, 0.1, 0.2, 0.2])

if annotation == True:
    annotation_text = f"""gLeak: {leakg}nS
gNa: {nag}nS
gIH: {ihg}nS
gKLT: {kltg}nS
gKHT: {khtg}nS
gKA: {kag}nS


ELeak: {revleak}mV
Ek: {revk}mV
ENa: {revna}mV"""
    ax1.annotate(
        annotation_text,
        xy=(600, -80),  # Point to annotate (x, y)
        xytext=(800, -40),  # Position of the text (x, y)
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow")  # Add a box around the text
    )

# Run simulation and collect data
for amp in stim_amps:
    setup_simulation(stim, amp, stimdelay, stimdur, v_rest)
    stim_trace = h.Vector().record(stim._ref_i)
    soma_v = h.Vector().record(my_cell.soma(0.5)._ref_v)
    stim_values = np.array(stim_trace.to_python())
    soma_v_values = np.array(soma_v.to_python())
    t_values = np.array(t.to_python())
    soma_values.append(soma_v_values)
    plot_traces(ax1, t_values, soma_v_values, amp)
    plot_inset(axin, t_values, stim_values)

# Calculate input resistance
# input_resistance = Rin(soma_values, t_values, stimdelay, stimdur, stim_amps)
#
# # Prepare to display results
# results = pd.DataFrame({
#     'Stimulus Amplitude (pA)': stim_amps,
#     'Input Resistance (MΩ)': [input_resistance] * len(stim_amps)  # Display the same input resistance value
# })

# Print results
# print("Input Resistance Calculation:")
# print(results)

# Save the plot
# plt.savefig(filename)
plt.show()