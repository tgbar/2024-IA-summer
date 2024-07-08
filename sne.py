import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# some constants
omegb = 0.02226 # omega_b * h**2

######################3
#
# Lambda CDM

# E(z) squared
def Esq(z, Om):
    return Om*(1+z)**3 + 1 - Om

# inverse E(z)
def Einv(z, Efunc, param):
    return 1/np.sqrt(Efunc(z, param))

# integral of 1/E(z) with solve_ivp
def intdif(z, Efunc, param):
    Ei = lambda x, y: Einv(x, Efunc, param)
    sol = solve_ivp(Ei, [0, z[-1]],[0], t_eval=z)
    return sol.y.flatten()

#######################################
#
# MAIN


# read supernovae data
z_sn, mb, dmb = np.loadtxt('binned.txt', usecols=(2,4,5), unpack=True, skiprows=1)
z_sn, idsort = np.unique(z_sn, return_index=True) 
mb = mb[idsort]
dmb = dmb[idsort]

plt.errorbar(z_sn, mb, yerr=dmb)