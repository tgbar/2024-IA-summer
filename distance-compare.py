import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import timeit


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
    Ei = lambda x, y: Einv(x, Efunc, param)
    sol = solve_ivp(Ei, [0, z[-1]],[0], dense_output=True)
    # interpolate the solution
    return sol.sol(z).flatten()


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
dl1 = (1+z)*intdif(z, Esq, 0.3)
dl2 = (1+z)*intdif_interp(z, Esq, 0.3)

plt.plot(z, dl1, label="intdif")
plt.plot(z, dl2, label="intdif_interp")

print("Outputs close:", np.allclose(dl1, dl2, rtol=1e-5, atol=1e-8))
print("Max absolute error:", np.max(np.abs(dl1 - dl2)))

repeat = 300
t1 = timeit.timeit(lambda: intdif(z, Esq, 0.3), number=repeat)
t2 = timeit.timeit(lambda: intdif_interp(z, Esq, 0.3), number=repeat)

print(f"Version 1 time: {t1:.6f} s")
print(f"Version 2 time: {t2:.6f} s")
print(t1/t2*100)


# magnitude (not calibrated)
# m = 5*np.log10(dl)

# plot magnitudes
#plt.plot(z, m)

# t1 = timeit.timeit(lambda: f1(input_data), number=repeat)
# t2 = timeit.timeit(lambda: f2(input_data), number=repeat)

# print(f"Version 1 time: {t1:.6f} s")
# print(f"Version 2 time: {t2:.6f} s")
