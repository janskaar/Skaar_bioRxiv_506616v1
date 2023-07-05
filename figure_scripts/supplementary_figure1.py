import numpy as np
import matplotlib.pyplot as plt
import os, re

logdir = os.path.join("../train_conditional_distributions/deep_lmc_hyperparam_scan_final/output")
logfs = os.listdir(logdir)
logfs = [f for f in logfs if ".log" in f]
params = np.array([list(map(int, re.findall("\d+", f))) for f in logfs])[:,:2]
losses = [np.loadtxt(os.path.join(logdir, f), delimiter=",") for f in logfs]
minlosses = np.array([l.min(0) for l in losses])

num1 = [10, 20, 30, 40]
num2 = [10, 20, 30, 40, 50, 60]

test_lossgrid = np.full((len(num1), len(num2)), np.nan)
train_lossgrid = np.full((len(num1), len(num2)), np.nan)
for i, h1 in enumerate(num1):
    for j, h2 in enumerate(num2):
        indices = (params == np.array([[h1, h2]])).all(1).nonzero()[0]
        if len(indices) == 0:
            continue
        test_lossgrid[i, j] = min(minlosses[indices][:,1])
        train_lossgrid[i, j] = min(minlosses[indices][:,0])


fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True)
fig.set_size_inches([9,4])
fig.subplots_adjust(wspace=0.3)
vmin, vmax = np.nanmin(train_lossgrid), -300. #np.nanmax(test_lossgrid)

ax[0].pcolormesh(train_lossgrid, vmin=vmin, vmax=vmax)
ax[0].set_yticks(np.arange(len(num1)) + 0.5)
ax[0].set_yticklabels(num1)
ax[0].set_xticks(np.arange(len(num2)) + 0.5)
ax[0].set_xticklabels(num2)
ax[0].set_xlabel("num layer2")
ax[0].set_ylabel("num layer1")

ax[1].pcolormesh(test_lossgrid, vmin=vmin, vmax=vmax)
ax[1].set_yticks(np.arange(len(num1)) + 0.5)
ax[1].set_yticklabels([])
ax[1].set_xticks(np.arange(len(num2)) + 0.5)
ax[1].set_xticklabels(num2)
ax[1].set_xlabel("num layer2")
#ax[1].set_ylabel("num neurons")

midpoint = (vmax - vmin) / 2 + vmin
color = "white"
for y, _ in enumerate(num1):
    for x, _ in enumerate(num2):
        val = train_lossgrid[y,x]
        ax[0].text(x + 0.5, y + 0.5, f"{val:.1f}",
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=6,
                   color=color)

        val = test_lossgrid[y,x]
        ax[1].text(x + 0.5, y + 0.5, f"{val:.1f}",
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=6,
                   color=color)

ax[0].set_title("Training loss")
ax[1].set_title("Test loss")

fig.savefig("dgp_hyperparam_scan.pdf")
plt.show()

