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

# integral of 1/E(z) with solve_ivp and interpolation
def intdif_interp(z, Efunc, param):
    lx = 1000
    zx = np.linspace(0, z[-1], lx)
    Ei = lambda x, y: Einv(x, Efunc, param)
    sol = solve_ivp(Ei, [0, z[-1]],[0], t_eval=zx)
    # interpolate the solution
    sol = np.interp(z, zx, sol.y.flatten())
    return sol


#######################################
#
# MAIN

# random redshift for mock SNe data
# (uniform distribution)
# (z=0.01 to z=3)
z = np.random.uniform(low=1e-2, high=3, size=2000)
z = np.sort(z)

# cosmological parameters
Om = 0.3


# distance
dl = (1+z)*intdif(z, Esq, 0.3)

# magnitude (not calibrated)
m = 5*np.log10(dl)

# plot magnitudes
#plt.plot(z, m)