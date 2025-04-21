# bluepyopt_evaluator.py
import os
import bluepyopt as bpop
import config_bpop
from simulation_bpop import cost_function, cost_function_all
from neuron_model import create_neuron
from data_loader import load_heka_data
import matplotlib.pyplot as plt

class MNTBEvaluator(bpop.evaluators.Evaluator):
    def __init__(self):
        super().__init__()

        if int(os.environ.get('SCOOP_WORKER', '0')) == 0:
            # Load experimental data once
            voltage, time, stim, labels = load_heka_data(
                config_bpop.full_path_to_file,
                config_bpop.group_idx,
                config_bpop.series_idx,
                config_bpop.channel_idx
            )

            n_sweeps = len(voltage)
            plt.figure(figsize=(12, 6))

            for i in range(n_sweeps):
                plt.plot(time[i] * 1000, voltage[i] * 1000, label=f"Sweep {i}")

            plt.xlabel("Time (ms)")
            plt.ylabel("Voltage (mV)")
            plt.title("Loaded Experimental Sweeps")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # 👉 Ask user which sweep to fit:
            print("\nAvailable sweeps:")
            for i in range(n_sweeps):
                print(f"  Sweep {i}")

        # ✅ Now set the sweep according to user choice
        # Keep all sweeps
        #self.v_exp_list = [v * 1000 for v in voltage]
        #self.t_exp_list = [t * 1000 for t in time]

        voltage, time, stim, labels = load_heka_data(
            config_bpop.full_path_to_file,
            config_bpop.group_idx,
            config_bpop.series_idx,
            config_bpop.channel_idx
        )

        # Save only simple things
        self.v_exp_list = [v.copy() * 1000 for v in voltage]  # deep copy
        self.t_exp_list = [t.copy() * 1000 for t in time]

        # Create the model neuron
        #self.soma, self.axon, self.dend = create_neuron()

        self.objectives = [bpop.objectives.Objective('cost')]

        # Define parameters
        self.params = []
        param_names = [
            "gna", "gkht", "gklt","gh",
            "cam", "kam", "cbm", "kbm",
            "cah", "kah", "cbh", "kbh",
            "can", "kan", "cbn", "kbn",
            "cap", "kap", "cbp", "kbp",
            "na_scale", "kht_scale", "klt_scale", "ih_soma", "ih_dend"
        ]

        for name, (minval, maxval) in zip(param_names, config_bpop.bounds):
            self.params.append(bpop.parameters.Parameter(name=name, bounds=(minval, maxval)))

    def evaluate_with_lists(self, param_values):
        """BluePyOpt calls this function with a list of parameter values."""

        soma, axon, dend = create_neuron()
        v_exp_list = self.v_exp_list
        t_exp_list = self.t_exp_list

        cost = cost_function_all(param_values, soma, axon, dend, t_exp_list, v_exp_list)
        return [cost]


    def init_simulator_and_evaluate_with_lists(self, param_values):
        """Needed for DEAPOptimisation compatibility"""
        return self.evaluate_with_lists(param_values)
