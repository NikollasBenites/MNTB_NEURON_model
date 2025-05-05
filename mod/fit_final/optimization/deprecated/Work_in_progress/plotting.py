import matplotlib.pyplot as plt
import numpy as np


def plot_traces(ax, t_values, soma_values, amp, color='red', linewidth=0.5):
    ax.plot(t_values, soma_values, color=color, linewidth=linewidth)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')


def plot_inset(axin, t_values, stim_values):
    axin.plot(t_values, stim_values, color='black', linewidth=0.5)
    axin.set_ylabel('I (nA)')
    axin.grid(False)

def plot_ap_count(ax, amps, ap_counts):
    ax.plot(amps * 1000, ap_counts, marker='o', linestyle='-', color='b')
    ax.set_xlabel('Stimulus Amplitude (pA)')
    ax.set_ylabel('Number of APs')
    ax.set_title('Number of APs vs. Stimulus Amplitude')
    ax.grid(True)
