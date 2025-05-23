import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# some constants
omegb = 0.02226 # omega_b * h**2

######################
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
    sol = solve_ivp(Ei, [0, z[-1]],[0], dense_output=True)
    return sol.sol(z).flatten()

#######################################
#
# MAIN


# read supernovae data
z_sn, mb, dmb = np.loadtxt('data/Pantheon+SH0ES.dat', usecols=(2,8,9), unpack=True, skiprows=1)
z_sn, idsort = np.unique(z_sn, return_index=True) 
mb = mb[idsort]
dmb = dmb[idsort]


d_sn = (1+z_sn)*intdif(z_sn, Esq, 0.3)
m_sn = 5*np.log10(d_sn)

# M = -19.36
# c/H0 = 3000/0.7
# M_curl = M + 25  + 5 log_10 (3000/0.7)
# M_curl = 23.80
plt.plot(z_sn,m_sn + 23.80)
plt.errorbar(z_sn, mb, fmt='.', yerr=dmb, color='r')