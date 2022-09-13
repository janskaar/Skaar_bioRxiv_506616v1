"""
Runs NEST simulations and predicts LFPs with parameters defined by
scripts set_up_parameters....py
"""
import os, sys, io, time, h5py, pickle
import nest
import numpy as np
from mpi4py import MPI

top_dir = sys.argv[1]
param_dir = os.path.join(top_dir, 'parameters')
log_dir = os.path.join(top_dir, 'logs')

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def simulate():
    """
    Runs a single simulation, parameters taken from upper scope
    """

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": PSET['dt'], "print_time": False,
                          "overwrite_files": True})

    nest.SetKernelStatus({'grng_seed': PSET['nest_seed'], 'rng_seeds': [PSET['nest_seed']+2]})

    np.random.seed(PSET['numpy_seed'])

    print("Building network")

    ## Set parameters for neurons and poisson generator
    nest.SetDefaults("iaf_psc_alpha", neuron_params)
    nest.SetDefaults("poisson_generator", {"rate": PSET['p_rate']})

    ## Create all neurons and recorders
    # local populations
    nodes_ex = nest.Create("iaf_psc_alpha", PSET['NE'])
    nodes_in = nest.Create("iaf_psc_alpha", PSET['NI'])

    # external population
    noise = nest.Create("poisson_generator")

    # spike recorders
    espikes = nest.Create("spike_detector")
    ispikes = nest.Create("spike_detector")
    print("first exc node: {}".format(nodes_ex[0]))
    print("first inh node: {}".format(nodes_in[0]))

    ## Set initial membrane voltages to random values between 0 and threshold
    nest.SetStatus(nodes_ex, "V_m",
                   np.random.rand(len(nodes_ex)) * neuron_params["V_th"])
    nest.SetStatus(nodes_in, "V_m",
                   np.random.rand(len(nodes_in)) * neuron_params["V_th"])

    ## Spike recording parameters
    nest.SetStatus(espikes, [{
        "label": os.path.join(PSET['savefolder'], 'nest_output', PSET['ps_id'], label + "-EX"),
        "withtime": True,
        "withgid": True,
        "to_file": PSET['save_spikes'],
    }])
    nest.SetStatus(ispikes, [{
        "label": os.path.join(PSET['savefolder'], 'nest_output', PSET['ps_id'], label + "-IN"),
        "withtime": True,
        "withgid": True,
        "to_file": PSET['save_spikes'],
     }])

    ## Set synaptic weights
    nest.CopyModel("static_synapse", "excitatory",
                    {"weight": PSET['J'], 'delay': PSET['delay']})
    nest.CopyModel("static_synapse", "inhibitory",
                    {"weight": -PSET['J']*PSET['g'], 'delay': PSET['delay']})

    ## Connect 'external population' poisson generator to local neurons
    nest.Connect(noise, nodes_ex, 'all_to_all', "excitatory")
    nest.Connect(noise, nodes_in, 'all_to_all', "excitatory")

    ## Record spikes to be saved from a subset of each population
    nest.Connect(nodes_ex, espikes, 'all_to_all', 'excitatory')
    nest.Connect(nodes_in, ispikes, 'all_to_all', 'excitatory')


    ## Connect local populations
    conn_params_ex = {'rule': 'fixed_indegree', 'indegree': PSET['CE']}
    nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, 'excitatory')
    conn_params_in = {'rule': 'fixed_indegree', 'indegree': PSET['CI']}
    nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, 'inhibitory')


    endbuild = time.time()

    nest.Simulate(PSET['simtime'])

    endsimulate = time.time()

    ## Calculate firing rate
    events_ex = nest.GetStatus(espikes, "n_events")[0]
    events_in = nest.GetStatus(ispikes, "n_events")[0]

    ## Get sample spikes for saving
    ex_events = nest.GetStatus(espikes, 'events')[0]
    in_events = nest.GetStatus(ispikes, 'events')[0]
    # ex_spikes = np.stack((ex_events['senders'], ex_events['times'])).T
    # in_spikes = np.stack((in_events['senders'], in_events['times'])).T

    ## Get population firing histograms for calculating LFP
    ex_hist_times = ex_events['times']
    in_hist_times = in_events['times']
    hist_bins_1 = np.arange(PSET['simtime'] + 2, dtype=np.float32) - 0.5
    ex_hist, _ = np.histogram(ex_hist_times, bins=hist_bins_1)
    in_hist, _ = np.histogram(in_hist_times, bins=hist_bins_1)

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('ex_hist', data=ex_hist)
        f.create_dataset('in_hist', data=in_hist)
    print('%s complete'%i)

## divvy up simulations
sim_list = os.listdir(os.path.join(top_dir, 'nest_output'))
sim_list.sort()
if rank == 0:
    print('%d simulations in total'%len(sim_list))
sims_per_rank = len(sim_list) // size
remainder = len(sim_list) % size
local_sim_indices = [sims_per_rank*rank + i for i in range(sims_per_rank)]
local_ids = [sim_list[i] for i in local_sim_indices]
if rank < remainder:
    local_sim_indices.append(sims_per_rank*size+rank)
    local_ids.append(sim_list[-(rank+1)])

## Start iterating through simulations
for j, i in enumerate(local_ids):
    print('Starting sim ', local_sim_indices[j])

    ## Load parameters from file
    with open(os.path.join(param_dir, i + '.pkl'), 'rb') as f:
        PSET = pickle.load(f)

    output_file = os.path.join(PSET['savefolder'], 'nest_output', PSET['ps_id'], 'LFP_firing_rate.h5')

    startbuild = time.time()

    neuron_params = {"C_m": PSET['CMem'],
                     "tau_m": PSET['tauMem'],
                     "t_ref": PSET['t_ref'],
                     "E_L": PSET['E_L'],
                     "V_reset": PSET['V_reset'],
                     "V_m": PSET['V_m'],
                     "V_th": PSET['theta'],
                     "I_e": 0.,
                     "tau_syn_ex": PSET['tauSyn'],
                     "tau_syn_in": PSET['tauSyn']}

    label = 'brunel-py'

    ## Run simulation
    simulate()
