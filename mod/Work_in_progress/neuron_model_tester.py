# neuron_model.py

from neuron import h, rxd
from neuron.units import ms, mV, µm
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
h.load_file("stdrun.hoc")
from calculate_input_resistance import calculate_input_resistance

# Initialize the NEURON environment and your model here
# Example: Create a simple neuron model
soma = h.Section(name='soma')
soma.L = soma.diam = 12.6157
soma.insert('hh')

# Define stimulus
stim = h.IClamp(soma(0.5))
stim.delay = 100  # in ms
stim.dur = 500   # in ms

# List of stimulus amplitudes in nA (convert pA to nA by dividing by 1000)
stim_amps = np.arange(-0.1, 0.301, 0.01)  # from -100 pA to 300 pA, step 10 pA

# Desired range for analysis in nA
analysis_range = (-0.01, 0.01)  # -10 pA to 10 pA

# Record voltage and time
t_vec = h.Vector().record(h._ref_t)  # time vector
v_vec = h.Vector().record(soma(0.5)._ref_v)  # voltage vector

# Store results for plotting after the loop
results = []

# Loop through each stimulus amplitude
for stim_amp in stim_amps:
    stim.amp = stim_amp  # Set stimulus amplitude
    # Run the simulation
    h.finitialize(-65)
    h.continuerun(600)  # in ms

    # Convert vectors to numpy arrays
    time = np.array(t_vec)
    voltage = np.array(v_vec)

    # Debugging information
    print(f"Stimulus Amplitude: {stim_amp * 1000} pA")
    print(f"Time Vector: {time}")
    print(f"Voltage Trace: {voltage}")

    # Check if the stimulus amplitude is within the analysis range
    if analysis_range[0] <= stim_amp <= analysis_range[1]:
        # Calculate the input resistance in the last 10 ms of the stimulation
        stim_end_time = stim.delay + stim.dur
        try:
            input_resistance = calculate_input_resistance(voltage, time, stim_end_time, stim_amp, )
            print(f"Membrane Input Resistance: {input_resistance} MΩ")
            results.append((time, voltage, stim_amp * 1000))

        except ValueError as e:
            print(e)

plt.figure()
# Plot the results for visualization after the loop
for time, voltage, stim_amp_pA in results:

    plt.plot(time, voltage)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title(f'Membrane Potential for Stimulus Amplitude {stim_amp_pA} pA')

    plt.show()