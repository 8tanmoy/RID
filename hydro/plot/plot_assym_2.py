import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from wolframclient.language import wl

TICS_FS =  30
plt.rcParams.update({'font.size': TICS_FS, 'xtick.labelsize' : TICS_FS, 'ytick.labelsize' : TICS_FS, 'axes.linewidth' : 1.5})

data = pd.read_csv('fe.out.sort3.min.80', header=None, delim_whitespace=True)
x 	= data[0].tolist()
y 	= data[1].tolist()
z 	= data[2].tolist()
assert len(x) == len(y) == len(z)

def screen(th, m, n, o):
	assert len(m) == len(n) == len(o)
	qm = []
	qn = []
	qo = []
	for ii in range(len(m)):
		if o[ii] < th:
			qm.append(m[ii])
			qn.append(n[ii])
			qo.append(o[ii])
	return(qm, qn, qo)

#--3D plot--
"""
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
plt.show()
"""

#--scatterplot with colors--
"""
x, y, z = screen(46.0, x, y, z)
plt.scatter(x, y, c=z, cmap=plt.jet())
plt.colorbar()
plt.show()
"""
LABEL_FS = 34

xx = np.linspace(1.53, 4.00, 80)
yy = np.linspace(1.53, 4.00, 80)
zz = np.array(z).reshape(80,80)

fig = plt.figure(figsize=([12,9]))
LEVELS = np.linspace(0, 46.0, 24)
cs = plt.contourf(xx, yy, zz.transpose(), LEVELS, cmap=plt.jet())
#cs2 = plt.contour(cs, levels=LEVELS, colors='black', linewidths=0.5)
plt.clim(0,46.0)
plt.ylabel(r"$P-O_{lg}\,\,(\AA)$", fontsize=LABEL_FS, labelpad=12)
plt.xlabel(r"$P-O_{attack}\,\,(\AA)$", fontsize=LABEL_FS, labelpad=12)
plt.colorbar(cs) #figsize=12,9
#plt.show()
plt.savefig("fe40_asym2.png", transparent=True, quality=100, bbox_inches='tight')
plt.savefig("fe40_asym2.eps", transparent=False, quality=100, bbox_inches='tight')