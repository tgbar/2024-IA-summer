import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


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
    sol = solve_ivp(Ei, [0, z[-1]],[0], t_eval=z)
    return sol.y.flatten()

#######################################
#
# MAIN

################################################
#
# COSMIC CHRONOMETERS

# read cosmic chronometers data
z_cc, H_cc, dH_cc = np.loadtxt('data/cosmic_chrono.txt', unpack=True, skiprows=1)

h = 0.7
Om = 0.3

H_th = 100*h*np.sqrt(Esq(z_cc, Om))

plt.plot(z_cc, H_th)
plt.errorbar(z_cc, H_cc, fmt='.',yerr=dH_cc, color='r')
