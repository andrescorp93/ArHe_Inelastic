import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simps
from scipy.interpolate import CubicSpline
from funcs import *

h = open('z3.txt', 'r').readline()
a3 = np.loadtxt('z3.txt', skiprows=1)
a4 = np.loadtxt('z4.txt', skiprows=1)
a5 = np.loadtxt('z5.txt', skiprows=1)
rtab = a3[:,0] * 1.88973
u3ztab = a3[:,1:]
u4ztab = a4[:,1:]
u5ztab = a5[:,1:]
ucbstab = np.zeros((len(rtab),len(a3[0])))
ucbstab[:,0] = rtab
t = np.matrix([[1,np.exp(-2),np.exp(-4)],[1,np.exp(-3),np.exp(-9)],[1,np.exp(-4),np.exp(-16)]]).getI()
for i in range(len(u3ztab[0])):
    u = np.matrix([u3ztab[:,i],u4ztab[:,i],u5ztab[:,i]])
    ucbstab[:,i+1] = t[0].dot(u)
# print(np.min(ucbstab[-1,:]))
np.savetxt('ecbs.txt',ucbstab[::-1,:], header=h[:-1])
