# calculate_input_resistance.py

from neuron import h
import numpy as np


def calculate_input_resistance(soma_values, t_values, stimdelay, stimdur, stim_amps, duration=10):
    """
    Calculate the membrane input resistance from voltage traces for different stimulus amplitudes.

    Parameters:
    soma_values (list of numpy arrays): List of voltage traces for different stimulus amplitudes.
    t_values (numpy array): The time vector corresponding to the voltage traces.
    stimdelay (float): The delay before the stimulus starts (in ms).
    stimdur (float): The duration of the stimulus (in ms).
    stim_amps (list of floats): List of stimulus amplitudes (in pA).
    duration (float): The duration in ms to consider before the end of the stimulation (default: 10 ms).

    Returns:
    float: The membrane input resistance.
    """
    # Find the indices for the end of the stimulation period and the duration window
    end_indices = np.where(t_values >= stimdur + stimdelay)[0]
    if end_indices.size == 0:
        raise ValueError("Stim end time is outside the range of the time vector.")

    end_index = end_indices[0]
    start_indices = np.where(t_values >= (stimdur + stimdelay - duration))[0]
    if start_indices.size == 0:
        raise ValueError("Stim end time - duration is outside the range of the time vector.")

    start_index = start_indices[0]

    # Calculate the average voltage for each stimulus amplitude
    avg_voltages = [np.mean(voltage_trace[start_index:end_index]) for voltage_trace in soma_values]

    # Calculate the change in voltage and stimulus amplitude
    voltage_change = np.diff(avg_voltages)
    stim_amp_change = np.diff(stim_amps)

    if len(voltage_change) == 0 or len(stim_amp_change) == 0:
        raise ValueError("Not enough data to calculate input resistance.")

    # Calculate input resistance (Ohm's law: R = ΔV / ΔI)
    input_resistance = np.mean(voltage_change / stim_amp_change)  # in MΩ (since voltage in mV and current in pA)

    return input_resistance
