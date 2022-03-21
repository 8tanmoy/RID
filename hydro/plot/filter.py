import csv

with open('fe_042.out') as f:
	reader = csv.reader(f)
	data = list(reader)

data_split = []
for ii in range(len(data)):
	data_split.append(data[ii][0].split())

import numpy as np

np_data1 = np.array(data_split).astype(np.float)
np_data1[:,3] /= -4.184
np_data1[:,3] -= np.min(np_data1[:,3])

print('mean,std: ', np.mean(np_data1[:,3]), np.std(np_data1[:,3]))

fe_threshold = 46.0
np_data2	= np_data1[np_data1[:,3] < fe_threshold]
np_data3	= np_data2[np_data2[:,2] > -1.50]

print(len(np_data1))
print(len(np_data2))

np.savetxt("fe_042_filtered.out", np_data3, delimiter=" ")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
TICS_FS = 25
plt.rcParams.update({'font.size': TICS_FS, 'xtick.labelsize' : TICS_FS, 'ytick.labelsize' : TICS_FS, 'font.family': "sans-serif"})

fig = plt.figure(figsize=[13,9])
#ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig, proj_type='ortho')

poatt	= (10*np_data3[:,0]).tolist()
polg	= (10*np_data3[:,1]).tolist()
asym	= (10*np_data3[:,2]).tolist()
fe 		= (np_data3[:,3]).tolist()

ax.set_xlim3d(np.min(poatt), np.max(poatt))
ax.set_ylim3d(np.min(polg), np.max(polg))
ax.set_zlim3d(np.min(asym), np.max(asym))
ax.set_xticks(np.arange(1.5, 4.00, 0.50))
ax.set_yticks(np.arange(1.5, 4.00, 0.50))
ax.set_zticks(np.arange(-2.0, 3.0, 1.0))

LABEL_FS = 30
img = ax.scatter(poatt, polg, asym, c=fe, cmap=plt.jet(), marker='o', alpha=1.0, depthshade=False)
ax.set_xlabel(r"$P-O_{attack}\,\,(\AA)$", fontsize=LABEL_FS+1, labelpad=28)
ax.set_ylabel(r"$P-O_{lg}\,\,(\AA)$", fontsize=LABEL_FS+1, labelpad=28)
ax.set_zlabel(r"Asymmetric Stretch " + r"$(\AA)$", fontsize=LABEL_FS-1, labelpad=17)
cbar = fig.colorbar(img, shrink=0.65, pad=0.11)
cbar.set_ticks(np.arange(0, 50, 6))
cbar.set_label(r"Free Energy " + r"$(kcal\,\,mol^{-1})$", fontsize=LABEL_FS-1, labelpad=12, rotation=90)
#plt.tight_layout()

ax.view_init(elev=20, azim=-54)

#plt.show()
plt.savefig("fe40.png", transparent=True, quality=100, bbox_inches='tight')
plt.savefig("fe40.eps", transparent=False, quality=100, bbox_inches='tight')

#for ii in range(-85,92,2):
#	ax.view_init(elev=20, azim=ii)
#	plt.savefig(f"fe40_az{ii}.png", transparent=False, quality=100, bbox_inches='tight')#