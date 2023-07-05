import os, h5py
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import welch
import LFPy

TICK_FONT_SIZE = 7
TITLE_FONT_SIZE = 7
LABEL_FONT_SIZE = 7
FULL_WIDTH_FIG_SIZE = 7
HALF_WIDTH_FIG_SIZE = FULL_WIDTH_FIG_SIZE / 2

data_dir = os.path.join('../simulation_scripts')
fig_dir = os.path.join('.')

lfps = np.load(os.path.join(data_dir, 'example_lfps.npy'))
rasters = np.load(os.path.join(data_dir, 'example_rasters.npy'))
hists = np.load(os.path.join(data_dir, 'hists_lfp_prediction.npy'))
ex_hist = hists[0]
in_hist = hists[1]
kernels = np.load(os.path.join(data_dir, 'lfp_kernels.npy'))
ex_kernel = kernels[0]
in_kernel = kernels[1]

erasters = rasters[-1][:80,:]
irasters = rasters[-1][80:,:]
lfp = lfps[-1]

def clear_axis(ax, middle=False):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if middle:
        ax.axis('off')

fig = plt.gcf()
fig.set_size_inches([7, 2.5])

ax1 = fig.add_axes([0.08, 0.55, 0.18, 0.28])    ## raster
ax2 = fig.add_axes([0.0, 0.02, 0.37, 0.4])      ## column
ax3 = fig.add_axes([0.35, 0.1, 0.25, 0.3])      ## lfp
ax4 = fig.add_axes([0.35, 0.72, 0.25, 0.1])     ## inhibitory histogram
ax5 = fig.add_axes([0.35, 0.57, 0.25, 0.1])     ## excitatory histogram

ax6 = fig.add_axes([0.7, 0.72, 0.25, 0.1])     ## inhibitory PSD
ax7 = fig.add_axes([0.7, 0.57, 0.25, 0.1])     ## excitatory PSD
ax8 = fig.add_axes([0.7, 0.1, 0.25, 0.3])     ## lfp PSD



tstart = 300
tstop = 900

## Raster plots
for i in range(len(erasters)):
    ax1.scatter(erasters[i][(erasters[i] >= tstart) & (erasters[i] <= tstop)],
                np.ones_like(erasters[i][(erasters[i] >= tstart) & (erasters[i] <= tstop)])*i-1,
                marker='|', color='orange', s=5.0, linewidth=0.8)

for i in range(len(irasters)):
    ax1.scatter(irasters[i][(irasters[i] >= tstart) & (irasters[i] <= tstop)],
                np.ones_like(irasters[i][(irasters[i] >= tstart) & (irasters[i] <= tstop)])*(i+len(erasters)+1),
                marker='|', color='royalblue', s=5.0, linewidth=0.8)

ax1.set_xlim(tstart, tstop)
ax1.set_ylim(0,len(erasters)+len(irasters))
ax1.set_xticks([tstart, tstart+(tstop-tstart)/2, tstop])
ax1.set_yticks([])
ax1.set_xlabel('t (ms)', fontdict={'fontsize': LABEL_FONT_SIZE})
ax1.plot([tstart, tstop], [len(erasters)-0.5]*2, linewidth=0.6, color='grey')

## Column and morphology plot
stretched_ex = os.path.join('./morphologies/L4E_53rpy1_cut.hoc')
stretched_in = os.path.join('./morphologies/L4I_oi26rbc1.hoc')

scellParams = dict(
    #excitory cells
    EX = dict(
        morphology = stretched_ex
    ),
    #inhibitory cells
    IN = dict(
        morphology = stretched_in
        ))

populationParams = dict(
    EX = dict(
        z_min = -450,
        z_max2 = -350
        ),

    IN = dict(
        z_min = -450,
        z_max2 = -350
        ))

r = np.sqrt(1000**2/np.pi) * 0.6

icell = LFPy.Cell(morphology=stretched_in, nsegs_method='fixed_length', max_nsegs_length=1.)
icell.set_pos(-200, 0, -400)

ecell = LFPy.Cell(morphology=stretched_ex, nsegs_method='fixed_length', max_nsegs_length=1.)
ecell.set_pos(200, 0, -400)
ecell.set_rotation(z=-0.05*np.pi)

iverts = []
for x, z in icell.get_idx_polygons():
    iverts.append(list(zip(x, z )))

everts = []
for x, z in ecell.get_idx_polygons():
    everts.append(list(zip(x, z)))

ipoly = PolyCollection(iverts, facecolors='royalblue', edgecolors='royalblue', linewidths=.5)
epoly = PolyCollection(everts, facecolors='orange',  edgecolors='orange', linewidths=0.5)

t1 = np.linspace(-np.pi, 0, 50)
t2 = np.linspace(0, np.pi, 50)

ax2.plot([-r, -r], [-450, -350], color='black', linewidth=0.5)
ax2.plot([r, r], [-450, -350], color='black', linewidth=0.5)

ax2.plot([-r, 0], [-500, -500], '--', color='black', linewidth=0.5)
ax2.plot([-450, -280], [-600, -500], color='black', linewidth=0.5)

ax2.text(-500, -650, 'r = %.0f $\mathrm{\mu m}$'%r, fontdict={'fontsize': LABEL_FONT_SIZE})

ax2.plot(r*np.cos(t1), 25*np.sin(t1) - 500, color='black', linewidth=0.5)
ax2.plot(r*np.cos(t2), 25*np.sin(t2) - 500, ':', color='black', linewidth=0.5)
ax2.plot(r*np.cos(t1), 25*np.sin(t1) - 300, color='black', linewidth=0.5)
ax2.plot(r*np.cos(t2), 25*np.sin(t2) - 300, ':', color='black', linewidth=0.5)
ax2.plot([-r, -r], [-500, -300], color='black', linewidth=0.5)
ax2.plot([r, r], [-500, -300], color='black', linewidth=0.5)

ax2.plot(r*np.cos(t1), 25*np.sin(t1) - 0, color='black', linewidth=0.5)
ax2.plot(r*np.cos(t2), 25*np.sin(t2) - 0, color='black', linewidth=0.5)
ax2.plot([-r, -r], [0, -300], color='black', linewidth=0.5)
ax2.plot([r, r], [0, -300], color='black', linewidth=0.5)

# ax2.text(-r-320, -310, 'z = -300 $\mathrm{\mu m}$', fontdict={'fontsize': LABEL_FONT_SIZE})
# ax2.text(-r-320, -510, 'z = -500 $\mathrm{\mu m}$', fontdict={'fontsize': LABEL_FONT_SIZE})
# ax2.text(-r-270, -5, 'z = 0 $\mathrm{\mu m}$', fontdict={'fontsize': LABEL_FONT_SIZE})

elec_z = [0, -100, -200, -300, -400, -500]
elec_x = [0, 0, 0, 0, 0, 0]

ax2.scatter(elec_x, elec_z, marker='o', color='black', s=10)

ax2.plot([-20, -20], [0, -100], '--', color='black', linewidth=0.5)
ax2.plot([-250, -20], [-100, -50], color='black', linewidth=0.5)
ax2.plot([-20, -3], [0, 0], '--', color='black', linewidth=0.5)
ax2.plot([-20, -3], [-100, -100], '--', color='black', linewidth=0.5)

ax2.text(-450, -150, 'd = 100 $\mathrm{\mu m}$', fontdict={'fontsize': LABEL_FONT_SIZE})

ax2.add_collection(ipoly)
ax2.add_collection(epoly)
ax2.set_xlim(-r-200,r+200)
ax2.set_ylim(-580, 70)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.axis('off')

ax2.axis(ax2.axis('equal'))

## LFP plot
for i in range(6):
    ax3.plot(np.arange(tstart, tstop), (lfp[i, tstart:tstop]-lfp[i, tstart:tstop].mean())*0.4 + (500 - i*100)*0.01, color='black', linewidth=0.8)

#ax3.plot([800, 800], [1, 2], color='red')
#ax3.text(810, 1.25, '1 mV', fontdict={'fontsize': 8})
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.set_ylim(-0.7, 5.7)
ax3.set_xlim(tstart, tstop)
ax3.set_xticks([tstart, tstart+(tstop-tstart)/2, tstop])
ax3.set_yticks([])
#ax3.set_yticklabels(['ch. 1', 'ch. 2', 'ch. 3', 'ch. 4', 'ch. 5', 'ch. 6'][::-1], fontdict={'fontsize': LABEL_FONT_SIZE})
ax3.tick_params(axis='y', length=0.0)
ax3.set_xlabel('t (ms)', fontdict={'fontsize': LABEL_FONT_SIZE})
ax3.set_ylabel('$\phi (\mathbf{r}, t)$', fontdict={'fontsize': LABEL_FONT_SIZE})

for ax in [ax1, ax2, ax3]:
    ax.tick_params(labelsize=TICK_FONT_SIZE)

## Firing histogram plots
ax4.plot(np.arange(300, 900), in_hist[300:900], color='royalblue', linewidth=0.8)
ax5.plot(np.arange(300, 900), ex_hist[300:900], color='orange', linewidth=0.8)
ax4.set_yticks([])
ax5.set_yticks([])
ax5.set_xticks([300, 600, 900])
ax4.set_xticks([300, 600, 900])
ax4.tick_params(axis='both',  labelsize=TICK_FONT_SIZE)
ax5.tick_params(axis='both',  labelsize=TICK_FONT_SIZE)
ax5.set_xlabel('t (ms)', fontdict={'fontsize': LABEL_FONT_SIZE})
ax5.set_ylabel('$\\nu_I(t)$', fontdict={'fontsize': LABEL_FONT_SIZE})
ax4.set_ylabel('$\\nu_E(t)$', fontdict={'fontsize': LABEL_FONT_SIZE})
[clear_axis(ax) for ax in [ax4, ax5]]
ax4.set_xticklabels([])
ax5.set_xlabel('t (ms)')
ax4.set_xlim(300, 900)
ax5.set_xlim(300, 900)

## Firing rate PSD
fs, epsd = welch(ex_hist, fs=1000)
fs, ipsd = welch(in_hist, fs=1000)
ax6.plot(fs, np.log(ipsd), color='royalblue', linewidth=1.5)
ax7.plot(fs, np.log(epsd), color='orange', linewidth=1.5)
ax6.set_yticks([])
ax7.set_yticks([])
#ax7.set_xticks([300, 600, 900])
#ax6.set_xticks([300, 600, 900])
ax6.tick_params(axis='both',  labelsize=TICK_FONT_SIZE)
ax7.tick_params(axis='both',  labelsize=TICK_FONT_SIZE)
ax7.set_xlabel('t (ms)', fontdict={'fontsize': LABEL_FONT_SIZE})
ax7.set_ylabel('log $P_{\\nu_I}(f)$', fontdict={'fontsize': LABEL_FONT_SIZE})
ax6.set_ylabel('log $P_{\\nu_E}(f)$', fontdict={'fontsize': LABEL_FONT_SIZE})
[clear_axis(ax) for ax in [ax6, ax7]]
ax6.set_xticklabels([])
ax7.set_xlabel('f ($s^{-1}$)', fontdict={'fontsize': LABEL_FONT_SIZE})

## LFP PSD
fs, psd = welch(lfp, fs=1000)
logpsd = np.log(psd)
for i in range(6):
    ax8.plot(fs, (logpsd[i] - logpsd[i].mean()) * 0.05 + (500 - i*100)*0.01, color='black', linewidth=0.8)

ax8.spines['right'].set_visible(False)
ax8.spines['top'].set_visible(False)
ax8.set_ylim(-0.7, 5.7)
ax8.set_yticks([])
#ax3.set_yticklabels(['ch. 1', 'ch. 2', 'ch. 3', 'ch. 4', 'ch. 5', 'ch. 6'][::-1], fontdict={'fontsize': LABEL_FONT_SIZE})
ax8.tick_params(axis='y', length=0.0)
ax8.set_xlabel('f ($s^{-1}$)', fontdict={'fontsize': LABEL_FONT_SIZE})
ax8.set_ylabel('log $P_\phi (\mathbf{r}, f)$', fontdict={'fontsize': LABEL_FONT_SIZE})
ax8.tick_params(axis='both',  labelsize=TICK_FONT_SIZE)

#fig.savefig('results_fig1.pdf', bbox_inches='tight')
fig.savefig('results_fig1.eps', bbox_inches='tight')

plt.show()
