import os, time
import numpy as np
from hybridLFPy import PostProcess, Population, CachedNetwork, setup_file_dest
import lfpykit
import h5py
import neuron
from parameters import ParameterSet
from mpi4py import MPI
import sys, pickle


index = 3
simpath = os.path.join("brunel_simulations_lfp_validation")
with open(os.path.join(simpath, "parameters", f"0000{index}.pkl"), "rb") as f:
    PSET = pickle.load(f)

################# Initialization of MPI stuff ##################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


PSET.update(dict(
    #no cell type specificity within each E-I population
    #hence X == x and Y == X
    X = ["EX", "IN"],

    #population-specific LFPy.Cell parameters
    cellParams = dict(
        #excitory cells
        EX = dict(
            morphology = './morphologies/L4E_53rpy1_cut.hoc',
            v_init = PSET['E_L'],
            passive_parameters = dict(g_pas=(PSET["CMem"] / 250.) / (PSET["tauMem"] * 1E3),  # 250. is "baseline C_m" in NEST.
                                      e_pas=PSET['E_L']),
            cm = PSET["CMem"] / 250.,
            Ra = 150,
            nsegs_method = 'lambda_f',
            lambda_f = 100,
            tstart = 0,
            tstop = PSET['simtime'],
            verbose = False,
            dt = 0.1,
            passive=True
        ),
        #inhibitory cells
        IN = dict(
            morphology = './morphologies/L4I_oi26rbc1.hoc',
            v_init = PSET['E_L'],
            passive_parameters = dict(g_pas=(PSET["CMem"] / 250.) / (PSET["tauMem"] * 1E3),  # 250. is "baseline C_m" in NEST.
                                      e_pas=PSET['E_L']),
            cm = PSET["CMem"] / 250.,
            Ra = 150,
            nsegs_method = 'lambda_f',
            lambda_f = 100,
            tstart = 0,
            tstop = PSET['simtime'],
            verbose = False,
            dt = 0.1,
            passive=True
    )),

    #assuming excitatory cells are pyramidal
    rand_rot_axis = dict(
        EX = ['z'],
        IN = ['x', 'y', 'z'],
    ),

    #kwargs passed to LFPy.Cell.simulate()
    simulationParams = dict(),

    #set up parameters corresponding to cylindrical model populations
    populationParams = dict(
        EX = dict(
            number = PSET['NE'],
            radius = np.sqrt(1000**2 / np.pi),
            z_min = -450,
            z_max = -350,
            min_cell_interdist = 1.,
            min_r=[[-1E199, -600, -550, 1E99], [0, 0, 10, 10]],
            ),
        IN = dict(
            number = PSET['NI'],
            radius = np.sqrt(1000**2 / np.pi),
            z_min = -450,
            z_max = -350,
            min_cell_interdist = 1.,
            min_r=[[-1E199, -600, -550, 1E99], [0, 0, 10, 10]],
            ),
    ),

    #set the boundaries between the "upper" and "lower" layer
    layerBoundaries = [[0., -300],
                       [-300, -500]],

    #set the geometry of the virtual recording device
    electrodeParams = dict(
            #contact locations:
            x = [0]*6,
            y = [0]*6,
            z = [x*-100. for x in range(6)],
            #extracellular conductivity:
            sigma = 0.3,
            #contact surface normals, radius, n-point averaging
            N = [[1, 0, 0]]*6,
            r = 5,
            n = 20,
            seedvalue = None,
            #dendrite line sources, soma as sphere source (Linden2014)
            method = 'root_as_point',
            #no somas within the constraints of the "electrode shank":
    ),

    #runtime, cell-specific attributes and output that will be stored
    savelist = [],
    pp_savelist = [],
    plots=False,

    #time resolution of saved signals
    dt_output = 0.1
))

#for each population, define layer- and population-specific connectivity
#parameters
PSET.update(dict(
    #number of connections from each presynaptic population onto each
    #layer per postsynaptic population, preserving overall indegree
    k_yXL = dict(
        EX = [[int(PSET['CE']*0.5), 0],
              [int(PSET['CE']*0.5), PSET['CI']]],
        IN = [[0, 0],
              [PSET['CE'], PSET['CI']]],
    ),

    #set up synapse parameters as derived from the network
    synParams = dict(
        EX = dict(
            section = ['apic', 'dend'],
            syntype = 'AlphaISyn'
        ),
        IN = dict(
            section = ['dend', 'soma'],
            syntype = 'AlphaISyn'
        ),
    ),

    #set up table of synapse time constants from each presynaptic populations
    tau_yX = dict(
        EX = [PSET["tauSyn"], PSET["tauSyn"]],
        IN = [PSET["tauSyn"], PSET["tauSyn"]]
    ),
    #set up delays, here using fixed delays of network
    synDelayLoc = dict(
        EX = [PSET['delay'], PSET['delay']],
        IN = [PSET['delay'], PSET['delay']],
    ),
    #no distribution of delays
    synDelayScale = dict(
        EX = [None, None],
        IN = [None, None],
    ),
))

#putative mappting between population type and cell type specificity,
#but here all presynaptic senders are also postsynaptic targets
PSET.update(dict(
    mapping_Yy = list(zip(PSET['X'], PSET['X']))
))

J_ex = PSET["J"] / 1E3
J_in = PSET["g"] * J_ex

PSET.update(dict(J_yX = dict(
               EX = [J_ex, J_in],
               IN = [J_ex, J_in],
               ))),

PSET['spike_output_path'] = os.path.join(simpath, "nest_output", f"0000{index}")
output_path = os.path.join(simpath, "hybrid_output", f"0000{index}")

cell_path = os.path.join(output_path, 'cells')
population_path = os.path.join(output_path, 'populations')
figure_path = os.path.join(output_path, 'figures')

#check if mod file for synapse model specified in alphaisyn.mod is loaded
if not hasattr(neuron.h, 'AlphaISyn'):
    if RANK == 0:
        os.system('nrnivmodl')
    COMM.Barrier()
    neuron.load_mechanisms('.')

if RANK == 0:
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(cell_path):
        os.mkdir(cell_path)
    if not os.path.isdir(population_path):
        os.mkdir(population_path)
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)

COMM.Barrier()

SIMULATIONSEED = 123456

################################################################################
# MAIN simulation procedure                                                    #
################################################################################


if __name__ == '__main__':
    #tic toc
    tic = time.time()

    #Create an object representation containing the spiking activity of the network
    #simulation output that uses sqlite3. Again, kwargs are derived from the brunel
    #network instance.
    networkSim = CachedNetwork(
        simtime = PSET['simtime'],
        dt = PSET['dt'],
        spike_output_path = PSET['spike_output_path'],
        label = '',
        ext = 'dat',
        GIDs = {'EX' : [1, PSET['NE']], 'IN' : [PSET['NE']+1, PSET['NI']]},
        X = ['EX', 'IN'],
        cmap='rainbow_r',
    )

    ####### Set up populations #####################################################
    electrode = lfpykit.RecExtElectrode(cell=None, **PSET["electrodeParams"])

    #iterate over each cell type, and create populationulation object
    for i, Y in enumerate(PSET['X'][:1]):
        #create population:
        pop = Population(
                cellParams = PSET['cellParams'][Y],
                rand_rot_axis = PSET['rand_rot_axis'][Y],
                simulationParams = PSET['simulationParams'],
                populationParams = PSET['populationParams'][Y],
                y = Y,
                layerBoundaries = PSET['layerBoundaries'],
                probes=[electrode],
                savelist = PSET['savelist'],
                savefolder = output_path,
                dt_output = PSET['dt_output'],
                POPULATIONSEED = SIMULATIONSEED + i,
                X = PSET['X'],
                networkSim = networkSim,
                k_yXL = PSET['k_yXL'][Y],
                synParams = PSET['synParams'][Y],
                synDelayLoc = PSET['synDelayLoc'][Y],
                synDelayScale = PSET['synDelayScale'][Y],
                J_yX = PSET['J_yX'][Y],
                tau_yX = PSET['tau_yX'][Y],
            )

        #run population simulation and collect the data
        pop.run()
        pop.collect_data()

        #object no longer needed
        del pop


    ####### Postprocess the simulation output ######################################

    #reset seed, but output should be deterministic from now on
    np.random.seed(SIMULATIONSEED)

    #do some postprocessing on the collected data, i.e., superposition
    #of population LFPs, CSDs etc
    postproc = PostProcess(y = PSET['X'][:1],
                           dt_output = PSET['dt_output'],
                           probes=[electrode],
                           savefolder = output_path,
                           mapping_Yy = PSET['mapping_Yy'],
                           savelist = PSET['pp_savelist'],
                           cells_subfolder = os.path.split(cell_path)[-1],
                           populations_subfolder = os.path.split(population_path)[-1],
                           figures_subfolder = os.path.split(figure_path)[-1],
                           compound_file = f'lfp_sum.h5'
                           )

    #run through the procedure
    postproc.run()

    COMM.Barrier()


