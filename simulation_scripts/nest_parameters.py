import time, operator, pickle, hashlib

## Define parameters
NEST_PSET = dict(
    label ='brunel',
    save_spikes=False,
    dt=0.1,         # Simulation time resolution in ms
    simtime=400.,  # Simulation time in ms
    nest_seed=int(time.time()), # base for seeds, will be updated for each
                                # individual parameterset
    numpy_seed=int(time.time())//2,
    g=5.0,          # ratio inhibitory weight/excitatory weight
    eta=1.0,        # external rate relative to threshold rate
    epsilon=0.1,    # connection probability (before: 0.1)
    CMem=250.0,       # capacitance of membrane in in pF
    theta=20.0,     # membrane threshold potential in mV
    V_reset=0.0,    # reset potential of membrane in mV
    E_L=0.0,        # resting membrane potential in mV
    V_m=0.0,        # membrane potential in mV

    tauMem=20.0,    # time constant of membrane potential in ms
    delay=1.5,      # synaptic delay
    t_ref=2.0,      # refractory period
    tauSyn=5.0,     # synaptic time constant
    order=2000,     # network scaling factor
)

NEST_PSET.update(dict(
    NE = 4 * NEST_PSET['order'], # number of excitatory neurons
    NI = 1 * NEST_PSET['order']  # number of inhibitory neurons
))

NEST_PSET.update(dict(
    N_neurons = NEST_PSET['NE'] + NEST_PSET['NI'], # total number of neurons
    CE = int(NEST_PSET['NE'] * NEST_PSET['epsilon']), # number of excitatory synapses per neuron
    CI = int(NEST_PSET['NI'] * NEST_PSET['epsilon'])  # number of inhibitory synapses per neuron
))

NEST_PSET.update(dict(
    C_tot = NEST_PSET['CE'] + NEST_PSET['CI']  # total number of synapses per neuron
))
