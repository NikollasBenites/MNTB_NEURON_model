# main_bluepyopt.py

from bluepyopt_evaluator import MNTBEvaluator
from bluepyopt.deapext.optimisations import DEAPOptimisation
import bluepyopt as bpop
import config_bpop
import pandas as pd
import os

# Create the evaluator
evaluator = MNTBEvaluator()

# Create the optimizer
optimizer = DEAPOptimisation(evaluator=evaluator,
                             offspring_size=50)

# Run optimization
final_pop, hall_of_fame, logs, hist = optimizer.run(max_ngen=40)

# Get the best individual
best_individual = hall_of_fame[0]
print("\nBest individual parameter values:")
for name, value in zip([p.name for p in evaluator.params], best_individual):
    print(f"{name:10s}: {value:.6f}")

# Save best parameters to CSV if you want (like you did before)
import pandas as pd
import os

param_names = [p.name for p in evaluator.params]
output_dir = config_bpop.output_dir
os.makedirs(output_dir, exist_ok=True)

df = pd.DataFrame({
    'Parameter': param_names,
    'Value': best_individual
})

df.to_csv(os.path.join(output_dir, 'optimized_parameters_bluepyopt.csv'), index=False)
print(f"\nParameters saved to {output_dir}/optimized_parameters_bluepyopt.csv")
