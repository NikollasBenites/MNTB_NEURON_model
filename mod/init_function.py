from neuron import h
import numpy as np
import neuron
import numpy.ma as ma # masked array
h.load_file("stdrun.hoc")
def custom_init(v_init=-70):
    """
    Perform a custom initialization of the current model/section.

    This initialization follows the scheme outlined in the
    NEURON book, 8.4.2, p 197 for initializing to steady state.

    N.B.: For a complex model with dendrites/axons and different channels,
    this initialization will not find the steady-state the whole cell,
    leading to initial transient currents. In that case, this initialization
    should be followed with a 0.1-5 second run (depends on the rates in the
    various channel mechanisms) with no current injection or active point
    processes to allow the system to settle to a steady- state. Use either
    h.svstate or h.batch_save to save states and restore them. Batch is
    preferred

    Parameters
    ----------
    v_init : float (default: -60 mV)
        Voltage to start the initialization process. This should
        be close to the expected resting state.
    """
    inittime = -1e10
    tdt = neuron.h.dt  # save current step size
    dtstep = 1e9
    neuron.h.finitialize(v_init)
    neuron.h.t = inittime  # set time to large negative value (avoid activating
    # point processes, we hope)
    tmp = neuron.h.cvode.active()  # check state of variable step integrator
    if tmp != 0:  # turn off CVode variable step integrator if it was active
        neuron.h.cvode.active(0)  # now just use backward Euler with large step
    neuron.h.dt = dtstep
    n = 0
    while neuron.h.t < -1e9:  # Step forward
        neuron.h.fadvance()
        n += 1
    # print('advances: ', n)
    if tmp != 0:
        neuron.h.cvode.active(1)  # restore integrator
    neuron.h.t = 0
    if neuron.h.cvode.active():
        neuron.h.cvode.re_init()  # update d(state)/dt and currents
    else:
        neuron.h.fcurrent()  # recalculate currents
    neuron.h.frecord_init()  # save new state variables
    neuron.h.dt = tdt  # restore original time step

    return v_init