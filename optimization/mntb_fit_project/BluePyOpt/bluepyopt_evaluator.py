# bluepyopt_evaluator.py
import os
import bluepyopt as bpop
import config_bpop
from simulation_bpop import cost_function
from neuron_model import create_neuron
from data_loader import load_heka_data, select_sweep
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

            # Select sweep interactively
            v_exp, t_exp, sweep_idx = select_sweep(voltage, time, labels)

            self.v_exp = v_exp
            self.t_exp = t_exp

        # Create the model neuron
        self.objectives = [bpop.objectives.Objective('cost')]

        # Define parameters
        self.params = []
        param_names = [
            "gna", "gkht", "gklt","gh","gleak",
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
        cost = cost_function(param_values, soma, axon, dend, self.t_exp, self.v_exp)
        return [cost]

    def init_simulator_and_evaluate_with_lists(self, param_values):
        """Needed for DEAPOptimisation compatibility"""
        return self.evaluate_with_lists(param_values)
