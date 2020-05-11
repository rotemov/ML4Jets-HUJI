import numpy as np
import matplotlib.pyplot as plt

import energyflow as ef

plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'serif'

# load quark and gluon jets
X, y = ef.qg_jets.load(2000, pad=False)
num = 750

# the jet radius for these jets
R = 0.4

# process jets
Gs, Qs = [], []
for arr,events in [(Gs, X[y==0]), (Qs, X[y==1])]:
    for i,x in enumerate(events):
        if i >= num:
            break

        # ignore padded particles and removed particle id information
        x = x[x[:,0] > 0,:3]

        # center jet according to pt-centroid
        yphi_avg = np.average(x[:,1:3], weights=x[:,0], axis=0)
        x[:,1:3] -= yphi_avg

        # mask out any particles farther than R=0.4 away from center (rare)
        x = x[np.linalg.norm(x[:,1:3], axis=1) <= R]

        # add to list
        arr.append(x)

# choose interesting events
ev0, ev1 = Gs[0], Gs[15]

# calculate the EMD and the optimal transport flow
R = 0.4
emdval, G = ef.emd.emd(ev0, ev1, R=R, return_flow=True)

# plot the two events
colors = ['red', 'blue']
labels = ['Gluon Jet 1', 'Gluon Jet 2']
for i, ev in enumerate([ev0, ev1]):
    pts, ys, phis = ev[:, 0], ev[:, 1], ev[:, 2]
    plt.scatter(ys, phis, marker='o', s=2 * pts, color=colors[i], lw=0, zorder=10, label=labels[i])

# plot the flow
mx = G.max()
xs, xt = ev0[:, 1:3], ev1[:, 1:3]
for i in range(xs.shape[0]):
    for j in range(xt.shape[0]):
        if G[i, j] > 0:
            plt.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                     alpha=G[i, j] / mx, lw=1.25, color='black')

# plot settings
plt.xlim(-R, R);
plt.ylim(-R, R)
plt.xlabel('Rapidity');
plt.ylabel('Azimuthal Angle')
plt.xticks(np.linspace(-R, R, 5));
plt.yticks(np.linspace(-R, R, 5))

plt.text(0.6, 0.03, 'EMD: {:.1f} GeV'.format(emdval), fontsize=10, transform=plt.gca().transAxes)
plt.legend(loc=(0.1, 1.0), frameon=False, ncol=2, handletextpad=0)

plt.show()

# compute pairwise EMDs between all jets (takes about 3 minutes, can change n_jobs if you have more cores)
g_emds = ef.emd.emds(Gs, R=R, norm=True, verbose=1, n_jobs=1, print_every=25000)
q_emds = ef.emd.emds(Qs, R=R, norm=True, verbose=1, n_jobs=1, print_every=25000)

# prepare for histograms
bins = 10**np.linspace(-2, 0, 60)
reg = 10**-30
midbins = (bins[:-1] + bins[1:])/2
dmidbins = np.log(midbins[1:]) - np.log(midbins[:-1]) + reg
midbins2 = (midbins[:-1] + midbins[1:])/2

# compute the correlation dimensions
dims = []
for emd_vals in [q_emds, g_emds]:
    uemds = np.triu(emd_vals)
    counts = np.cumsum(np.histogram(uemds[uemds > 0], bins=bins)[0])
    dims.append((np.log(counts[1:] + reg) - np.log(counts[:-1] + reg))/dmidbins)

# plot the correlation dimensions
plt.plot(midbins2, dims[0], '-', color='blue', label='Quarks')
plt.plot(midbins2, dims[1], '-', color='red', label='Gluons')

# labels
plt.legend(loc='center right', frameon=False)

# plot style
plt.xscale('log')
plt.xlabel('Energy Scale Q/pT'); plt.ylabel('Correlation Dimension')
plt.xlim(0.02, 1); plt.ylim(0, 5)

plt.show()