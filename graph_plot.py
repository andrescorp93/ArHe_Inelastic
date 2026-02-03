import numpy as np
import matplotlib.pyplot as plt
import cycler
from funcs import *
from scipy.interpolate import make_interp_spline

dirs = [n for n in os.listdir() if (os.path.isdir(n) and (n[-2:]=='_s'))]
dirp = [n for n in os.listdir() if (os.path.isdir(n) and (n[-2:]=='_p'))]

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 15
plt.rcParams['image.cmap'] = 'Paired'
plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
                                                          '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'])
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['lines.linewidth'] = 3

energy_plot(dirs, colors)
energy_plot(dirp, colorp)
# so_plot(dirs)
# so_plot(dirp)
# ddr_plot(dirp)
# elastic_plot(dirs, colors)
# elastic_plot(dirp, colorp)
# diffuse_plot(dirs, colors)
# diffuse_plot(dirp, colorp)
# inelastic_plot(dirs)
# inelastic_plot(dirp)