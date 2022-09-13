'''Sets up simulation directories and parameters for NEST simulations
 including LFP approximations'''
import os, pickle
import numpy as np
from nest_parameters import NEST_PSET
from pyDOE import lhs

if __name__ == '__main__':
    ## Add the random varying parameters
    PSET = NEST_PSET.copy()

    ## max/min differences for [eta, g, tausyn, taumem, delay, t_ref, Csyn, theta, Cmem, Vreset]
    deltas = np.array([[2.5, 3.5, 7., 15., 3.9, 5.0, 75., 10., 200., 10.]])

    ## minimum values for [eta, g, tausyn, taumem, delay, t_ref, Csyn, theta, Cmem, Vreset]
    mins = np.array([[1.0, 4.5, 1., 15, 0.1, 0.0, 25., 15., 100., 0.]])

    parameters = np.array([[2., 3.5, 1., 20., 2., 2., 40., 20., 250., 0.],
                           [3., 6., 6., 18., 3.6, 5., 75., 25., 220., 6.],
                           [2., 5.5, 5., 20., 2., 2., 40., 20., 250., 0.]])

    PSET['simtime'] = 10000.

    # set up directory structure
    savefolder = os.path.join('./brunel_simulations_example_states')
    paramset_dest = os.path.join(savefolder, 'parameters')
    nest_output = os.path.join(os.path.join(savefolder, 'nest_output'))

    if not os.path.isdir(savefolder):
        os.mkdir(savefolder)
    if not os.path.isdir(paramset_dest):
        os.mkdir(paramset_dest)
    if not os.path.isdir(nest_output):
        os.mkdir(nest_output)

    print('Start parameter iteration')
    for i, _ in enumerate(parameters):
        paramset = PSET.copy()
        paramset.update({'nest_seed': paramset['nest_seed'] + i * 10})
        paramset.update({'numpy_seed': paramset['numpy_seed'] + i * 10})

        ## Update parameter set with the random values
        paramset.update({'eta': parameters[i,0],
                         'g': parameters[i,1],
                         'tauSyn': parameters[i,2],
                         'tauMem': parameters[i,3],
                         'delay': parameters[i,4],
                         't_ref': parameters[i,5],
                         'cSyn': parameters[i,6],
                         'theta': parameters[i,7],
                         'CMem': parameters[i,8],
                         'V_reset': parameters[i,9]
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
            'savefolder': savefolder,
            'save_spikes': True
                        })

        parameterset_file = os.path.join(paramset_dest, '{}.pkl'.format(ps_id))
        with open(parameterset_file, 'wb') as f:
            pickle.dump(paramset, f)
