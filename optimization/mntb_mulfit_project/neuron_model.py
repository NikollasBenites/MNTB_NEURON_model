# neuron_model.py

import numpy as np
from neuron import h
import MNTB_PN_myFunctions as mFun
import config_bpop


h.load_file('stdrun.hoc')

def create_neuron():
    soma = h.Section(name='soma')
    axon = h.Section(name='axon')
    dend = h.Section(name='dend')

    soma.diam = 15.5
    soma.Ra = 150
    soma.cm = 1
    soma.insert('leak')
    soma.insert('IH_dth')
    soma.insert('HT_dth_nmb')
    soma.ek = config_bpop.ek
    soma.v = config_bpop.v_init
    soma.erev_leak = config_bpop.erev

    axon.L = 25
    axon.diam = 3
    axon.Ra = 100
    axon.cm = 1
    axon.insert('leak')
    axon.insert('NaCh_nmb')
    axon.insert('LT_dth')
    axon.ek = config_bpop.ek
    axon.ena = config_bpop.ena
    axon.erev_leak = config_bpop.erev
    axon.v = config_bpop.v_init

    dend.L = 80
    dend.diam = 3
    dend.Ra = 100
    dend.cm = 1
    dend.insert('leak')
    dend.insert('IH_dth')
    dend.v = config_bpop.v_init
    dend.erev_leak = config_bpop.erev

    axon.connect(soma(1))
    dend.connect(soma(0))

    return soma, axon, dend

def nstomho(x, area):
    return (1e-9 * x / area)

def set_conductances(soma, axon, dend, neuron_params, na_scale, kht_scale, klt_scale, ih_soma,ih_dend, erev=config_bpop.erev, gleak=config_bpop.gleak):
    # Unpack parameters
    (gna, gkht, gklt,gh,
     cam, kam, cbm, kbm,
     cah, kah, cbh, kbh,
     can, kan, cbn, kbn,
     cap, kap, cbp, kbp) = neuron_params
     #na_scale, kht_scale) = params

    # Calculate areas
    totalcap = config_bpop.total_capacitance_pF
    soma_area = (totalcap * 1e-6) / 1  # cm², assuming 1 uF/cm²
    axon_area = np.pi * axon.diam * axon.L * 1e-8  # cm²

    # Helper functions
    def nstomho_soma(x):
        return (1e-9 * x / soma_area)

    def nstomho_axon(x):
        return (1e-9 * x / axon_area)

    # Set soma conductances
    soma.gkhtbar_HT_dth_nmb = nstomho_soma(gkht)*kht_scale
    soma.can_HT_dth_nmb = can
    soma.kan_HT_dth_nmb = kan
    soma.cbn_HT_dth_nmb = cbn
    soma.kbn_HT_dth_nmb = kbn
    soma.cap_HT_dth_nmb = cap
    soma.kap_HT_dth_nmb = kap
    soma.cbp_HT_dth_nmb = cbp
    soma.kbp_HT_dth_nmb = kbp
    soma.ghbar_IH_dth = nstomho_soma(gh)*ih_soma
    soma.erev_leak = erev
    soma.g_leak = nstomho_soma(gleak)

    # Set axon conductances
    axon.gnabar_NaCh_nmb = nstomho_axon(gna)*na_scale
    axon.cam_NaCh_nmb = cam
    axon.kam_NaCh_nmb = kam
    axon.cbm_NaCh_nmb = cbm
    axon.kbm_NaCh_nmb = kbm
    axon.cah_NaCh_nmb = cah
    axon.kah_NaCh_nmb = kah
    axon.cbh_NaCh_nmb = cbh
    axon.kbh_NaCh_nmb = kbh
    axon.gkltbar_LT_dth = nstomho_axon(gklt)*klt_scale
    axon.erev_leak = erev
    axon.g_leak = nstomho_axon(gleak)

    # Set dendrite conductances
    for seg in dend:
        seg.g_leak = nstomho_soma(gleak)
        seg.erev_leak = erev
        seg.ghbar_IH_dth = nstomho_soma(gh)*ih_dend
