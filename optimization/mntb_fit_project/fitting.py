# fitting.py

from scipy.optimize import minimize, differential_evolution
import config

def fit_parameters(cost_function):
    result_global = differential_evolution(cost_function, config.bounds,
                                            strategy='best1bin',
                                            maxiter=config.maxiter_global,
                                            popsize=config.popsize_global,
                                            polish=False)
    result_local = minimize(cost_function, result_global.x,
                             bounds=config.bounds,
                             method='L-BFGS-B',
                             options={'maxiter': config.maxiter_local})
    return result_local
