# functions used by multiprocessing are in a separate file
# for faster process initialization, so that the whole owpeakfit
# does not have to be imported on child processes

from lmfit import Model
import numpy as np
import scipy.integrate


def n_best_fit_parameters(model, params):
    """Number of output parameters for best fit results"""
    number_of_peaks = len(model.components)
    var_params = [name for name, par in params.items() if par.vary]
    number_of_params = len(var_params)
    return number_of_peaks + number_of_params + 1


def best_fit_results(model_result, x, shape):
    """Return array of best-fit results"""
    out = model_result
    sorted_x = np.sort(x)
    comps = out.eval_components(x=sorted_x)
    best_values = out.best_values

    output = np.zeros(shape)

    # add peak values to output storage
    col = 0
    for comp in out.model.components:
        # Peak area
        output[col] = scipy.integrate.trapezoid(
            np.broadcast_to(comps[comp.prefix], x.shape), sorted_x
        )
        col += 1
        for param in [n for n in out.var_names if n.startswith(comp.prefix)]:
            output[col] = best_values[param]
            col += 1
    output[-1] = out.redchi
    return output


lmfit_model = None
lmfit_x = None


def pool_initializer(model, parameters, x):
    # Pool initializer is used because lmfit's CompositeModel is not picklable.
    # Therefore we need to use loads() and dumps() to transfer it between processes.
    global lmfit_model
    global lmfit_x
    lmfit_model = Model(None).loads(model), parameters
    lmfit_x = x


def pool_fit(v):
    x = lmfit_x
    model, parameters = lmfit_model
    model_result = model.fit(v, params=parameters, x=x)
    shape = n_best_fit_parameters(model, parameters)
    bpar = best_fit_results(model_result, x, shape)
    fitted = np.broadcast_to(model_result.eval(x=x), x.shape)

    return model_result.dumps(), bpar, fitted, model_result.residual


def pool_fit2(v, model, parameters, x):
    model = Model(None).loads(model)
    model_result = model.fit(v, params=parameters, x=x)
    return model_result.dumps()
