"""Sets up simulation directories and parameters for NEST simulations
 including LFP approximations"""
import os, pickle, time
import numpy as np
from pyDOE import lhs

n_sims = 10000

# Define parameters
base_pset = dict(
    label ="brunel",
    save_spikes=False,
    dt=0.1,         # Simulation time resolution in ms
    epsilon=0.1,    # connection probability (before: 0.1)
    E_L=0.0,        # resting membrane potential in mV
    V_m=0.0,        # membrane potential in mV
    order=2000,     # network scaling factor
)

base_pset.update(dict(
    NE = 4 * base_pset["order"], # number of excitatory neurons
    NI = 1 * base_pset["order"]  # number of inhibitory neurons
))

base_pset.update(dict(
    N_neurons = base_pset["NE"] + base_pset["NI"], # total number of neurons
    CE = int(base_pset["NE"] * base_pset["epsilon"]), # number of excitatory synapses per neuron
    CI = int(base_pset["NI"] * base_pset["epsilon"])  # number of inhibitory synapses per neuron
))

base_pset.update(dict(
    C_tot = base_pset["CE"] + base_pset["CI"]  # total number of synapses per neuron
))

#############################
##       LHS               ##
#############################

varying_parameters = lhs(10, n_sims)

# max/min differences for [eta, g, tausyn, taumem, delay, t_ref, Csyn, theta, Cmem, Vreset]
deltas = np.array([[2.5, 3.5, 7., 15., 3.9, 5.0, 75., 10., 200., 10.]])

# minimum values for [eta, g, tausyn, taumem, delay, t_ref, Csyn, theta, Cmem, Vreset]
mins = np.array([[1.0, 4.5, 1., 15, 0.1, 0.0, 25., 15., 100., 0.]])

varying_parameters*= deltas
varying_parameters+= mins

# delay and refractory period must be multiple of time resolution
varying_parameters[:,4] = np.around(varying_parameters[:,4], 1)
varying_parameters[:,5] = np.around(varying_parameters[:,5], 1)

base_pset["simtime"] = 10500.

# set up directory structure
savefolder = os.path.join("./brunel_simulations_10p/")
parameterset_dest = os.path.join(savefolder, "parameters")
nest_output = os.path.join(savefolder, "nest_output")

if not os.path.isdir(savefolder):
    os.mkdir(savefolder)
if not os.path.isdir(parameterset_dest):
    os.mkdir(parameterset_dest)
if not os.path.isdir(nest_output):
    os.mkdir(nest_output)


base_seed = int(time.time())
rng = np.random.default_rng()
seeds = rng.integers(low=1, high=2 ** (31), size=n_sims)

print("Start parameter iteration")
for i, _ in enumerate(varying_parameters):
    paramset = base_pset.copy()
    paramset.update({"nest_seed": seeds[i]})

    # Update parameter set with the random values
    paramset.update({"eta": varying_parameters[i,0],
                     "g": varying_parameters[i,1],
                     "tauSyn": varying_parameters[i,2],
                     "tauMem": varying_parameters[i,3],
                     "delay": varying_parameters[i,4],
                     "t_ref": varying_parameters[i,5],
                     "cSyn": varying_parameters[i,6],
                     "theta": varying_parameters[i,7],
                     "CMem": varying_parameters[i,8],
                     "V_reset": varying_parameters[i,9]
                     })

    paramset.update({"J": paramset["cSyn"] / (paramset["tauSyn"] * np.exp(1))}),

    # Calculate rate for external population
    nu_th  = paramset["theta"] * paramset["CMem"] /(paramset["J"]*paramset["tauMem"] * np.exp(1) * paramset["tauSyn"])
    nu_ex  = paramset["eta"]*nu_th
    p_rate = 1000.0*nu_ex

    paramset.update({"p_rate": p_rate})

    ps_id = f"{i:05}"
    spike_output_path = os.path.join(nest_output, ps_id)
    if not os.path.isdir(spike_output_path):
        os.mkdir(spike_output_path)

    paramset.update({
        "ps_id": ps_id,
        "spike_output_path": spike_output_path,
        "savefolder": savefolder
                    })
    parameterset_file = os.path.join(os.path.join(parameterset_dest, "{}.pkl".format(ps_id)))
    with open(parameterset_file, "wb") as f:
        pickle.dump(paramset, f)
