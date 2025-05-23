import numpy as np
from scipy.integrate import solve_ivp
import emcee

import corner
import matplotlib.pyplot as plt


# --- Load Pantheon+ data ---
def load_pantheon_data(data_file):
    names = open(data_file).readline().strip().split()
    data = np.genfromtxt(data_file, skip_header=1, names=names)

    mask = data['zHD'] > 0.01
    zCMB = data['zCMB'][mask]
    zHEL = data['zHEL'][mask]
    m_obs = data['m_b_corr'][mask]
    m_obs_sig = data['m_b_corr_err_DIAG'][mask]

    return zCMB, zHEL, m_obs, m_obs_sig, mask, len(data['zHD'])

# --- Load and reconstruct covariance matrix ---
def load_covariance(cov_file, mask, N_full):
    with open(cov_file) as f:
        f.readline()  # skip the first line (length info)
        raw = np.fromiter((float(x) for x in f), dtype=float)

    C_full = np.zeros((N_full, N_full))
    idx = 0
    for i in range(N_full):
        for j in range(N_full):
            C_full[i, j] = raw[idx]
            idx += 1

    # Apply the mask
    mask_idx = np.where(mask)[0]
    C = C_full[np.ix_(mask_idx, mask_idx)]
    return C

# --- Cosmological distance calculations ---
def E(z, Omega_m, w):
    return np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m) * (1 + z)**(3 * (1 + w)))

def comoving_distance(z, Omega_m, w):
    Ei = lambda x, y: 1/E(x, Omega_m, w)
    sol = solve_ivp(Ei, [0, z[-1]], [0], dense_output=True)
    return 2997.92458 * sol.sol(z).flatten()  # Mpc

def angular_diameter_distance(z, Omega_m, w):
    return comoving_distance(z, Omega_m, w) / (1 + z)

def mu_theory(zCMB, zHEL, Omega_m, w, H0, M):
    # Compute D_A(z), interpolate, and apply Pantheon+ conversion
    D_A = angular_diameter_distance(zCMB, Omega_m, w)
    dL = (1 + zCMB) * (1 + zHEL) * D_A  # Luminosity distance in Mpc
    mu = 5 * np.log10(dL * 1e6) - 5  # Distance modulus in mag
    return mu + M

# --- Likelihood function for emcee ---
def make_log_likelihood(data_file, cov_file):
    zCMB, zHEL, m_obs, m_obs_sig, mask, N_full = load_pantheon_data(data_file)
    C = load_covariance(cov_file, mask, N_full)
    Cinv = np.linalg.inv(C)

    def log_likelihood(theta):
        Omega_m, M = theta
        mu_model = mu_theory(zCMB, zHEL, Omega_m, -1, 70, M)
        delta = m_obs - mu_model
        return -0.5 * np.dot(delta, Cinv @ delta)
    
    # alternative version using no covariance
    def log_likelihood_diag(theta):
        Omega_m, M = theta
        mu_model = mu_theory(zCMB, zHEL, Omega_m, -1, 70, M)
        delta = m_obs - mu_model
        return -0.5 * np.sum((delta / m_obs_sig)**2)

    return log_likelihood, log_likelihood_diag

# --- Likelihood function for binned Pantheon+ data ---
def make_log_likelihood_binned(data_file):
    z_sn, mb, dmb = np.loadtxt(data_file, usecols=(2,4,5), unpack=True, skiprows=1)
    z_sn, idsort = np.unique(z_sn, return_index=True) 
    mb = mb[idsort]
    dmb = dmb[idsort]
    
    def log_likelihood_binned(theta):
        Omega_m, M_curl = theta
        M = M_curl-25-5*np.log10(3000/.7)
        mu_model = mu_theory(z_sn, z_sn, Omega_m, -1, 70, M)
        delta = mb - mu_model
        return -0.5 * np.sum((delta / dmb) ** 2)

    return log_likelihood_binned

# Setup likelihood
#log_likelihood, log_likelihood_diag = make_log_likelihood("pantheon plus/Pantheon+SH0ES.dat", "pantheon plus/Pantheon+SH0ES_STAT+SYS.cov")
log_likelihood = make_log_likelihood_binned("data/binned.txt")


# Priors
def log_prior(theta):
    Omega_m, M = theta
    if 0.0 < Omega_m < 1.0 and 15 < M < 30:
        return 0.0
    return -np.inf

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# Run emcee
ndim = 2
nwalkers = 32
# initial = [0.3, -19] # for Pantheon+ data
initial = [0.3, 20] # for binned data
pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)

total_steps = 10000

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
sampler.run_mcmc(pos, total_steps, progress=True)

# Discard burn-in and flatten the chains
samples = sampler.get_chain(discard=500, flat=True)

# Plot corner plot
fig = corner.corner(samples, labels=[r"$\Omega_m$", r"$M$"], show_titles=True)

plt.show()

# For each parameter, calculate the median and 1-sigma confidence intervals
for i in range(samples.shape[1]):
    mcmc = np.percentile(samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print(f"Parameter {i}: {mcmc[1]:.3f} (+{q[1]:.3f}, -{q[0]:.3f})")