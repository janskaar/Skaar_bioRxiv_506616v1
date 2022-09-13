'''Sets up simulation directories and parameters for NEST simulations
 including LFP approximations'''
import os, pickle
import numpy as np
from nest_parameters import NEST_PSET
from pyDOE import lhs
from sklearn.utils.extmath import cartesian

posterior_dir = os.path.join('../posterior_mcmc_scripts/')
## Add the random varying parameters
PSET = NEST_PSET.copy()

varying_parameters = []
for i in range(100):
    proposals = np.load(os.path.join(posterior_dir, f'{i:04d}', 'proposals.npy'))
    acceptance = np.load(os.path.join(posterior_dir, f'{i:04d}', 'acceptance.npy'))
    samples = np.concatenate([proposals[:,j][acceptance[:,j]] for j in range(5)])
    indices = np.random.choice(np.arange(len(samples)), replace=False, size=50)
    varying_parameters.append(samples[indices])
varying_parameters = np.concatenate(varying_parameters, axis=0)
## max/min differences for [eta, g, tausyn, taumem, delay, t_ref, Csyn, theta, Cmem, Vreset]
deltas = np.array([[2.5, 3.5, 7., 15., 3.9, 5.0, 75., 10., 200., 10.]])

## minimum values for [eta, g, tausyn, taumem, delay, t_ref, Csyn, theta, Cmem, Vreset]
mins = np.array([[1.0, 4.5, 1., 15, 0.1, 0.0, 25., 15., 100., 0.]])

varying_parameters *= deltas
varying_parameters += mins

varying_parameters[:,4] = np.around(varying_parameters[:,4], 1)
varying_parameters[:,5] = np.around(varying_parameters[:,5], 1)

PSET['simtime'] = 10500.

# set up directory structure
savefolder = os.path.join('./brunel_simulations_10p_posteriors_rate/')
parameterset_dest = os.path.join(savefolder, 'parameters')
nest_output = os.path.join(savefolder, 'nest_output')

if not os.path.isdir(savefolder):
    os.mkdir(savefolder)
if not os.path.isdir(parameterset_dest):
    os.mkdir(parameterset_dest)
if not os.path.isdir(nest_output):
    os.mkdir(nest_output)

print('Start parameter iteration')
for i, _ in enumerate(varying_parameters):
    paramset = PSET.copy()
    paramset.update({'nest_seed': paramset['nest_seed'] + i * 10})
    paramset.update({'numpy_seed': paramset['numpy_seed'] + i * 10})

    ## Update parameter set with the random values
    paramset.update({'eta': varying_parameters[i,0],
                     'g': varying_parameters[i,1],
                     'tauSyn': varying_parameters[i,2],
                     'tauMem': varying_parameters[i,3],
                     'delay': varying_parameters[i,4],
                     't_ref': varying_parameters[i,5],
                     'cSyn': varying_parameters[i,6],
                     'theta': varying_parameters[i,7],
                     'CMem': varying_parameters[i,8],
                     'V_reset': varying_parameters[i,9]
                     })

    paramset.update({'J': paramset['cSyn'] / (paramset['tauSyn'] * np.exp(1))}),

    ## Calculate rate for external population
    nu_th  = paramset['theta'] * paramset['CMem'] /(paramset['J']*paramset['tauMem'] * np.exp(1) * paramset['tauSyn'])
    nu_ex  = paramset['eta']*nu_th
    p_rate = 1000.0*nu_ex

    paramset.update({'p_rate': p_rate})

    ps_id = f'{i:04}'
    spike_output_path = os.path.join(nest_output, ps_id)
    if not os.path.isdir(spike_output_path):
        os.mkdir(spike_output_path)

    paramset.update({
        'ps_id': ps_id,
        'spike_output_path': spike_output_path,
        'savefolder': savefolder
                    })

    parameterset_file = os.path.join(parameterset_dest, '{}.pkl'.format(ps_id))
    with open(parameterset_file, 'wb') as f:
        pickle.dump(paramset, f)
