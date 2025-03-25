from neuron import h
h.load_file("stdrun.hoc")

def setup_simulation(stim, amp, delay, duration, v_rest):
    if v_rest != 0:
        h.finitialize(v_rest)
    else:
        h.finitialize()
    stim.amp = amp
    stim.delay = delay
    stim.dur = duration
    h.continuerun(1000)

def create_stimulus(my_cell, stimdelay, stimdur):
    stim = h.IClamp(my_cell.soma(0.5))
    stim_traces = h.Vector().record(stim._ref_i)
    stim.delay = stimdelay
    stim.dur = stimdur
    return stim

def record_vectors(my_cell):
    soma_v = h.Vector().record(my_cell.soma(0.5)._ref_v)
    t = h.Vector().record(h._ref_t)
    return soma_v, t
