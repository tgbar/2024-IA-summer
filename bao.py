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
    sol = solve_ivp(Ei, [0, z[-1]],[0], t_eval=z)
    return sol.y.flatten()

#######################################
#
# MAIN


#######################################
#
# BAO data

# # # SDSS and/or DESI Dv data
z_v_bao = np.array([0.15, 1.49])
#Dv / rd
dv_bao = np.array([4.47, 26.07])
ddv_bao = np.array([0.17, 0.67 ])

### SDSS and/or DESI Dm and Dh data
z_bao = np.array([0.38, 0.51, 0.71, 0.93, 1.32, 2.33])
# D_M / rd
dm_bao = np.array([10.23, 13.36, 16.85, 21.71, 27.79, 38.8])
ddm_bao = np.array([0.17, 0.21, 0.32, 0.28, 0.69, 0.75])
# D_H / rd
dh_bao = np.array([25.00, 22.33, 20.08, 17.88, 13.82, 8.72])
ddh_bao = np.array([0.76, 0.58, 0.60, 0.35, 0.42, 0.14])

r_bao = np.array([0.228, 0.117, -0.42, -0.389, -0.444, -0.48])

def rd(Om, h):
    # DESI: 
    # Neff=3.04
    # omegb = 0.02236
    return 147.05*(Om*h**2/0.1432)**(-0.32)

def dv(z, param, Efunc):
    Om, h = param
    return 2998/h*( (z*(intdif(z, Efunc, Om))**2/np.sqrt(Efunc(z, Om))) )**(1/3)

#rs_fid = 147.78

##################
# example plot

Om = 0.295
hr = 101.8


z = np.linspace(0.05,2.5,50)

dm = 2998*intdif(z, Esq, Om)/hr
dh = 2998/hr/np.sqrt(Esq(z, Om))

dv = 2998*( (z*(intdif(z, Esq, Om))**2/np.sqrt(Esq(z,Om))) )**(1/3)/hr

plt.figure()
plt.plot(z, dm)
plt.errorbar(z_bao, dm_bao, yerr=ddm_bao, fmt='.')
plt.xlabel(r"$z$")
plt.ylabel(r"$D_M(z)/r_d$")

plt.figure()
plt.plot(z, dh)
plt.errorbar(z_bao, dh_bao, yerr=ddh_bao, fmt='.')
plt.xlabel(r"$z$")
plt.ylabel(r"$D_H(z)/r_d$")

plt.figure()
plt.plot(z, dv)
plt.errorbar(z_v_bao, dv_bao, yerr=ddv_bao, fmt='.')
plt.xlabel(r"$z$")
plt.ylabel(r"$D_V(z)/r_d$")