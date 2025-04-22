# main_bpop_debug.py
import os
import numpy as np
import bluepyopt as bpop
from bluepyopt.deapext.optimisations import DEAPOptimisation
import bluepyopt.deapext.algorithms as algorithms
import config_bpop
import datetime

from bluepyopt_evaluator import MNTBPNEvaluator
from neuron_model import create_neuron

h = None
try:
    from neuron import h
except ImportError:
    raise ImportError("This script must be run inside NEURON environment.")

# Create output directory if it doesn't exist
if not os.path.exists(config_bpop.output_dir):
    os.makedirs(config_bpop.output_dir)

# Create neuron
soma, axon, dend = create_neuron()

# Set up evaluator
evaluator = MNTBPNEvaluator(soma, axon, dend)

# Configure optimizer for small debug mode
small_mu = 5       # small population
small_lambda = 5   # number of offspring
small_ngen = 2     # number of generations
cxpb = 0.5         # crossover prob
mutpb = 0.3        # mutation prob
seed = np.random.randint(0, 1e6)

print(f"[Debug Mode] Using seed {seed}")

optimizer = bpop.optimisations.DEAPOptimisation(
    evaluator=evaluator,
    offspring_size=small_lambda,
    seed=seed,
    verbose=True,
    checkpoint_file=os.path.join(config_bpop.output_dir, 'checkpoint.pkl'),
)

# Run the optimization
print("[Debug Mode] Starting small optimization run...")
final_pop, hall_of_fame, logs, hist = optimizer.run(
    max_ngen=small_ngen,
    cp_frequency=1,
    cp_filename=os.path.join(config_bpop.output_dir, 'checkpoint.pkl'),
)

# Save best individual
best_params = hall_of_fame[0].values

param_names = [p.name for p in evaluator.params]
best_params_named = dict(zip(param_names, best_params))

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
best_param_file = os.path.join(config_bpop.output_dir, f"best_params_debug_{timestamp}.csv")

import pandas as pd
pd.DataFrame.from_dict(best_params_named, orient='index').to_csv(best_param_file)

print(f"[Debug Mode] Best parameters saved to {best_param_file}")
