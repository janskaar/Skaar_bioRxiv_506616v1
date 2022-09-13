import h5py, os, re
import numpy as np

np.random.seed(12345)

def create_raster(ex_path, in_path):
    """
    Get spiketrains of 80 excitatory and 20 inhibitory spike trains
    for raster plot.
    """
    ex_spikes = np.loadtxt(ex_path)
    in_spikes = np.loadtxt(in_path)
    ex_args = ex_spikes[:,0].argsort()
    in_args = in_spikes[:,0].argsort()
    ex_spikes = ex_spikes[ex_args]
    in_spikes = in_spikes[in_args]
    ex_ids, ex_inds = np.unique(ex_spikes[:,0], return_index=True)
    in_ids, in_inds = np.unique(in_spikes[:,0], return_index=True)
    ex_sample = np.random.choice(np.arange(len(ex_inds)), size=80, replace=False)
    in_sample = np.random.choice(np.arange(len(in_inds)), size=20, replace=False)
    ex_trains = []
    in_trains = []
    n = 0
    for j in ex_sample:
        if j == (len(ex_inds) - 1):
            i = ex_inds[j]
            ii = len(ex_spikes)
        else:
            i = ex_inds[j]
            ii = ex_inds[j+1]
        times = np.sort(ex_spikes[i:ii,1])
        if len(times) > n:
            n = len(times)
        ex_trains.append(times)
    for j in in_sample:
        if j == (len(in_inds) - 1):
            i = in_inds[j]
            ii = len(in_spikes)
        else:
            i = in_inds[j]
            ii = in_inds[j+1]
        times = np.sort(in_spikes[i:ii,1])
        if len(times) > n:
            n = len(times)
        in_trains.append(times)
    raster_arr = np.zeros([100, n])
    for i, train in enumerate(ex_trains + in_trains):
        raster_arr[i,0:len(train)] = train
    return raster_arr

example_data_dir = os.path.join('.')

example_lfps = []
example_hlfps = []
example_frates = []
example_rasters = []
for i in range(3):
    with h5py.File(os.path.join(example_data_dir, 'nest_output', f'{i:04d}', 'LFP_firing_rate.h5')) as f:
        lfp = f['data'][()]
        ex_hist = f['ex_hist'][()]
        in_hist = f['in_hist'][()]
        gfrate = (ex_hist + in_hist)/10000*1000

    raster = create_raster(os.path.join(example_data_dir, 'nest_output', f'{i:04d}', 'brunel-py-EX-10002-0.gdf'),
                           os.path.join(example_data_dir, 'nest_output', f'{i:04d}', 'brunel-py-IN-10003-0.gdf'))
    #example_hlfps.append(hlfp)
    example_lfps.append(lfp)
    example_rasters.append(raster)
    example_frates.append(gfrate)

largest_raster = max([r.shape[-1] for r in example_rasters])
raster_arr = np.zeros([len(example_rasters), 100, largest_raster])
for i, r in enumerate(example_rasters):
    raster_arr[i,:,:r.shape[-1]] = r

#example_hlfps = np.array(example_hlfps)
example_frates = np.array(example_frates)
example_lfps = np.array(example_lfps)
np.save(os.path.join('example_rasters.npy'), raster_arr)
np.save(os.path.join('example_lfps.npy'), example_lfps)
# #np.save(os.path.join('example_hybrid_lfps.npy'), example_hlfps)
np.save(os.path.join('example_frates.npy'), example_frates)
