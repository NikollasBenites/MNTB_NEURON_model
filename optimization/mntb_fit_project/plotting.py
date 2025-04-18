# plotting.py

import matplotlib.pyplot as plt
import os
import config
import numpy as np

def plot_voltage_fit(t_exp, v_exp, t_sim, v_sim):
    plt.figure(figsize=(10,5))
    plt.plot(t_exp, v_exp, label='Experimental', linewidth=2)
    plt.plot(t_sim, v_sim, label='Simulated', linestyle='--')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Action Potential Fit')
    plt.legend()
    plt.grid()
    if config.save_figures:
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        plt.savefig(f"{config.output_dir}/voltage_fit.png", dpi=300)
    if config.show_plots:
        plt.show()

def plot_phase_plane(trace, time, label='Trace'):
    dt = time[1] - time[0]
    dVdt = np.gradient(trace, dt)
    plt.plot(trace, dVdt, label=label)

    plt.xlabel('Voltage (mV)')
    plt.ylabel('dV/dt (mV/ms)')
    plt.grid()
    plt.legend()
