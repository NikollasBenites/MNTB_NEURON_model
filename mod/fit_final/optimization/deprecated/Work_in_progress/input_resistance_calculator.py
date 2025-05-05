# input_resistance_calculator.py

from neuron import h
import numpy as np
h.load_file("stdrun.hoc")

def avg_soma_values(t_values,stimdelay,stimdur,soma_values,amp):
    """
    Calculate the the average of the voltage traces for different stimulus amplitudes on the setady state as default.

    Parameters:
    t_values (numpy array): The time vector corresponding to the voltage traces.
    t_min (float): The time where you want to begin to analyze the voltage traces (.
    t_max (float): The time where you want to finish the analysis of the voltage traces.

    soma_values (list of numpy arrays): List of voltage traces for different stimulus amplitudes.
    stimdelay (float): The delay before the stimulus starts (in ms).
    stimdur (float): The duration of the stimulus (in ms).
    amps (list of floats): List of stimulus amplitudes (in pA).


    Returns:
    float: The membrane input resistance.
    """

    average_soma_values = np.array([])
    t_min = stimdelay + stimdur - 60
    t_max = stimdelay + stimdur - 10

    # Extract the range of soma_values based on t_values to calculate input resistance
    range_indices = np.where((t_values >= t_min) & (t_values <= t_max))
    soma_values_range = np.round(soma_values[range_indices], 3)
    t_values_range = np.round(t_values[range_indices], 3)

    # Calculate the average of soma_values within the specified range
    average_soma_value = np.round(np.mean(soma_values_range), 3)
    average_soma_values = np.append(average_soma_values, average_soma_value)

    # for amp in amps:
    #     delta_amp = amps[i] - amps[i - 1]
    #     delta_soma = average_soma_values[i] - average_soma_values[i - 1]
    #     slope = np.round((delta_soma / delta_amp) / 1000, 3)
    #     slopes = np.append(slopes,slope)

    # # Calculate the slope for each step in amps relative to the average_soma_values
    return average_soma_values