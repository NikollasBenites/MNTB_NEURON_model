# fitting.py
from scipy.optimize import minimize, differential_evolution
import config_noKinetics
def fit_parameters(cost_function):
    result_global = differential_evolution(
        cost_function,
        config_noKinetics.bounds,
        strategy='best1bin',
        maxiter=config_noKinetics.maxiter_global,
        popsize=config_noKinetics.popsize_global,
        polish=False,
        tol=config_noKinetics.tol_global  # <-- Added here for DE
    )

    result_local = minimize(
        cost_function,
        result_global.x,
        bounds=config_noKinetics.bounds,
        method='L-BFGS-B',
        options={
            'maxiter': config_noKinetics.maxiter_local,
            'ftol': config_noKinetics.ftol_local,  # <-- Added ftol (function value tolerance)
            'gtol': config_noKinetics.gtol_local   # <-- Optional: gradient norm tolerance
        }
    )

    return result_local