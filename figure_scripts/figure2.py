import h5py, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import welch

TITLE_FONT_SIZE = 7
LABEL_FONT_SIZE = 7
FULL_WIDTH_FIG_SIZE = 7
HALF_WIDTH_FIG_SIZE = FULL_WIDTH_FIG_SIZE / 2

data_dir = os.path.join('brunel_simulations_example_states')
fig_dir = os.path.join('.')

def plot_column(axes, lfp, raster, gfrate, lfp_fs_psd, v_fs_psd, t_start, t_stop, i, yticks=False, lfp_bar=0.1):
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]
    ax5 = axes[4]

    i_store = i

    ## Raster plot
    ax1.set_xlim(t_start, t_stop)
    ax1.set_ylim(0, 100)
    ax1.set_xticks([])
    for i, r in enumerate(raster):
        train = r[np.logical_and(r < t_stop, r > t_start)]
        ax1.scatter(train, np.ones_like(train)*i, s=1.0, c='black', marker='|', linewidth=1.0)
        ax1.set_yticks([0, 50, 100])
#    ax1.set_title('$\eta$ = %.1f, g = %.1f'%(params[0], params[1]), fontsize=TITLE_FONT_SIZE)

    ## Global firing rate
    ax2.set_xlim(t_start, t_stop)
    ax2.set_xlabel('t (ms)', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1)
    ax2.plot(np.arange(t_start, t_stop), gfrate[t_start:t_stop], color='black', lw=0.5)
    # ax2.set_yticks([0, int(gfrate[t_start:t_stop].mean())])
    if i_store!=0:
        ax2.set_yticks([0, np.round(gfrate[t_start:t_stop].max()*0.5, decimals=-1)])
    if i_store==0:
        ax2.set_yticks([0, np.round(gfrate[t_start:t_stop].max()*0.5, decimals=-2)])
    ax2.tick_params(axis='both', pad=-0.5)

    ## PSD of global firing rate
    ax3.semilogy(v_fs_psd[0], v_fs_psd[1]*1e12, lw=0.5, color='black')
    ax3.set_xlabel('f (Hz)', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1)
    ax3.tick_params(axis='x', pad=-1)
    ax3.set_xlim(0, 500)
#    ax3.set_ylim(1e3, 1e12)
    #ax3.set_yticks([1e3, 1.e6, 1e9])

    ## LFP traces
    ax4.set_xlim(t_start, t_stop)
    ax4.set_ylim(-0.2, 6.15)
    lfp_scaling = lfp[0,t_start:t_stop].std()*4.0  ## scale LFP
    lfp_unit = 1 / lfp_scaling     ## length of 1mV

    for i in range(6):
        ax4.plot(np.arange(t_start, t_stop),
                 5.5 - i + (lfp[i][t_start:t_stop]-lfp[i][t_start:t_stop].mean())/lfp_scaling,
                 color='black', lw=0.5)
        phi = 1
        # ax4.plot(np.arange(t_start, t_stop),
        #          5.5 - i + (hlfp[i][t_start-phi:t_stop-phi]-hlfp[i][t_start-phi:t_stop-phi].mean())/lfp_scaling,
        #          color='red', lw=0.5, ls='--')
        ax4.set_xlabel('t (ms)', fontdict={'fontsize': LABEL_FONT_SIZE, 'rotation': 0}, labelpad=-1)
    xpos = t_start + (t_stop-t_start)*0.7
    #ax3.plot([xpos, xpos], [3.0, 3.5], color='red')
    ax4.plot([xpos, xpos], [2.0, 2.0+lfp_unit*lfp_bar], color='red', lw=1.5)
    ax4.text(xpos + (t_stop-t_start)*0.07, 2.0, str(lfp_bar) + ' mV',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6), fontdict={'fontsize': LABEL_FONT_SIZE})

    ## PSD of channel 1 LFP
    ax5.semilogy(lfp_fs_psd[0], lfp_fs_psd[1]*1e6, lw=0.5, color='black')
    #ax5.loglog(hlfp_fs_psd[0], hlfp_fs_psd[1]*1e6, lw=0.5, color='red', ls='--')
    ax5.set_xlabel('f (Hz)', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1)
    ax5.tick_params(axis='x', pad=-1)
    ax5.set_xlim(0, 500)
#    ax5.set_ylim(1e-7, 1e5)
    #ax5.set_yticks([1e-6, 1e3])

    ## Ticks and labels
    if yticks:
        ax1.set_ylabel('neuron #', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=11)
        ax2.set_ylabel(r'$\overline{\nu}  \mathrm{(t)}$ (s$^{-1}$)', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=12)
        ax3.set_ylabel('$\mathrm{P_{\\nu} (f)  \ (Hz)}$', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=5)
        ax4.set_yticks([5.5 - i for i in range(6)])
        ax4.set_yticklabels(['ch. %d'%i for i in range(1,7)], fontdict={'fontsize': LABEL_FONT_SIZE, 'rotation': 0})
        ax4.set_ylabel('$\mathrm{\phi (\mathbf{r}, t)}$', fontdict={'fontsize': LABEL_FONT_SIZE})
        ax5.set_ylabel('$\mathrm{P_{\phi} ( \mathbf{r}, f)  \ (\mu V^2}/\mathrm{Hz})}$',
                       fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-1)
    else:
        ax1.set_yticklabels([])
        ax4.set_yticks([])
        #ax5.set_yticklabels([])
        #ax3.set_yticklabels([])

## Start and stop time for each plot
a= 0
tstart = [300+a, 300+a, 300+a, 300+a, 300+a]
tstop = [500+a, 600+a, 600+a, 600+a, 600+a]

## Load data, created in 'create_activity_figure_data.py'
rasters = np.load(os.path.join(data_dir, 'example_rasters.npy'))
lfps = np.load(os.path.join(data_dir, 'example_lfps.npy'))
#hybrid_lfps = np.load(os.path.join(data_dir, 'example_hybrid_lfps.npy'))
frates = np.load(os.path.join(data_dir, 'example_frates.npy'))

## Plotting
fig = plt.gcf()
fig.set_size_inches([FULL_WIDTH_FIG_SIZE,5.0])
gs = GridSpec(12,3, hspace=0.1, left=0.1, right=0.98, bottom=0.1, top=0.9)

titles = ['SR', 'SI (fast)', 'AI']
labels = ['A', 'B', 'C']
title_x = [0.2 + 0.32*i for i in range(3)]
label_x = [0.1 + 0.31*i for i in range(3)]

for i in range(3):
    yticks = True if i == 0 else False

    ## Size of scale bar in LFP plots
    lfp_bar = 1.0 if i == 1 else 0.1

    #hlfp_fs, hlfp_psd = welch(hybrid_lfps[i,0,150:], nperseg=300, fs=1000)
    lfp_fs, lfp_psd = welch(lfps[i,0,150:], nperseg=300, fs=1000)
    v_fs, v_psd = welch(frates[i][150:]/1000, nperseg=300, fs=1000)

    ax1 = plt.subplot(gs[:3,i])
    ax2 = plt.subplot(gs[3,i])
    ax3 = plt.subplot(gs[5,i])
    ax4 = plt.subplot(gs[7:10,i])
    ax5 = plt.subplot(gs[11,i])
    plot_column([ax1, ax2, ax3, ax4, ax5],
                lfps[i],
                rasters[i],
                frates[i],
                (lfp_fs, lfp_psd),
                (v_fs, v_psd),
                tstart[i],
                tstop[i],
                i,
                yticks=yticks,
                lfp_bar=lfp_bar)
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.tick_params(axis='both', labelsize=LABEL_FONT_SIZE)

    ## Add title
    tax = fig.add_axes([title_x[i], 0.95, 0.05, 0.05])
    tax.axis('off')
    tax.text(0, 0, titles[i], fontdict={'fontsize': LABEL_FONT_SIZE, 'rotation': 0})

    ## Add labels
    lax = fig.add_axes([label_x[i], 0.95, 0.02, 0.02])
    lax.axis('off')
    lax.text(0,0, labels[i], fontdict={'fontsize': 12, 'rotation': 0})

fig.savefig(os.path.join('results_fig2.pdf'))
#fig.savefig(os.path.join(fig_dir, 'Fig5.eps'))
plt.show()
