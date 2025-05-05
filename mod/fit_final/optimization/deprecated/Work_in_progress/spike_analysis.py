import numpy as np
from neuron import h
h.load_file("stdrun.hoc")

def count_spikes(v_trace, threshold=-10):
    spikes = np.where((v_trace[:-1] < threshold) & (v_trace[1:] >= threshold))[0]
    return len(spikes)

def detect_spikes_with_netcon(my_cell, threshold=-10):
    netcon = h.NetCon(my_cell.soma(0.5)._ref_v, None, sec=my_cell.soma)
    netcon.threshold = threshold
    spike_times = h.Vector()
    netcon.record(spike_times)
    return netcon, spike_times

def clear_spike_times(spike_times):
    spike_times.clear()
