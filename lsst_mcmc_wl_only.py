import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
import sacc
from cobaya import run
from getdist import plots
import scipy.optimize as opt

#%%
# Version with all baryonic parameters free with mock cosmic shear data only

#%%
s = sacc.Sacc.load_fits('mock_lsst_data.fits')

#%%

# See what measurements are contained within the dataset
for n, t in s.tracers.items():
    print(t.name, t.quantity, type(t))

# Type of power spectra
data_types = np.unique([d.data_type for d in s.data])
print("Data types: ", data_types)

# Tracer combinations
print("Tracer combinations: ", s.get_tracer_combinations())

# Data size
print("Size: ", s.mean.size)

#%%

# Retrieve the full data vector and its covariance
d = s.mean
cov = s.covariance.dense
print(d.shape, cov.shape)

# Get the ell values at which the observations were taken
ells, cells = s.get_ell_cl('cl_ee', 'wl_0', 'wl_0')
print(ells)

#%%

# Compute the inverse of the covariance matrix of the data
icov = np.linalg.inv(cov)

def A_IA(amp, eta, z, z_star=0.62):
    return amp * ((1+z)/(1+z_star))**eta

# Define the k and a arrays used to compute the non-linear power spectrum
a_array = np.linspace(1/(1+1.4), 1, 32)
l10k_array = np.linspace(-3, 1, 256)
k_array = 10**l10k_array

def lnprob(Om_m, s8, A_IA0, eta_IA, log_Mc, log_eta, log_beta, log_M1, sign=1):
    
    # Set top-hat priors
    if ((Om_m < 0.23) or
        (Om_m > 0.40) or
        (s8 < 0.73) or
        (s8 > 0.90) or
        (A_IA0 < -5.0) or 
        (A_IA0 > 5.0) or
        (eta_IA < -5.0) or
        (eta_IA > 5.0) or
        (log_Mc < 9.0) or
        (log_Mc > 15.0) or
        (log_eta < -0.70) or
        (log_eta > 0.70) or
        (log_beta < -1.0) or
        (log_beta > 0.70) or
        (log_M1 < 9.0) or
        (log_M1 > 13.0)):
        return -np.inf * sign
    
    cosmo = ccl.Cosmology(Omega_c=Om_m-0.045,
                      Omega_b=0.045,
                      h=0.685,
                      n_s=0.973,
                      sigma8=s8)
    
    # Calculate the non-linear power spectrum without baryons
    pk_array = np.array([ccl.nonlin_matter_power(cosmo, k_array, a) for a in a_array])
    pk_no_baryons = ccl.Pk2D(a_arr=a_array, lk_arr=np.log(k_array), pk_arr=np.log(pk_array))

    # Now add the impact of baryonic effects using the bacco emulator
    bar = ccl.BaccoemuBaryons(log10_M_c=log_Mc, log10_eta=log_eta, log10_beta=log_beta, log10_M1_z0_cen=log_M1)
    try:
        pk_with_baryons = bar.include_baryonic_effects(cosmo, pk_no_baryons)
    except:  # This might fail if params are out of BACCO priors
        return -np.inf*sign

    # Loop over all combinations of redshift bins and stores the theoretical value in a vector m
    m = []
    for t1, t2 in s.get_tracer_combinations():
        lens_1 = ccl.WeakLensingTracer(cosmo, dndz=(s.tracers[t1].z, s.tracers[t1].nz), 
                                       ia_bias=(s.tracers[t1].z, A_IA(amp=A_IA0, eta=eta_IA, z=s.tracers[t1].z)), n_samples=255)
        lens_2 = ccl.WeakLensingTracer(cosmo, dndz=(s.tracers[t2].z, s.tracers[t2].nz),
                                       ia_bias=(s.tracers[t2].z, A_IA(amp=A_IA0, eta=eta_IA, z=s.tracers[t2].z)), n_samples=255)
        C_ell = ccl.angular_cl(cosmo, lens_1, lens_2, ells, p_of_k_a=pk_with_baryons)
        m.append(C_ell)

    m = np.concatenate(m)
    diff = d - m
    logprob = -0.5*np.dot(diff, np.dot(icov, diff)) * sign
    return logprob

#%%

# Minimise the logprob by creating a wrapper function
def lprob_min(p):
    return lnprob(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], -1)

# Initial guess
p = np.array([0.27, 0.83, 1.0, 1.0, 11.0, 0.1, -0.2, 11.0])

theta_star = opt.fmin(lambda p: lnprob(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], -1), p)
print(theta_star)

#%%

info = {"likelihood": {"logprob": lnprob}}

#%%

info["params"] = {
    "Om_m": {"prior": {"min": 0.23, "max": 0.40}, "ref": theta_star[0], "proposal": 0.01, "latex": r"\Omega_{\mathrm{m}}"},
    "s8": {"prior": {"min": 0.73, "max": 0.90}, "ref": theta_star[1], "proposal": 0.01, "latex": r"\sigma_8"},
    "A_IA0" : {"prior": {"min": -5.0, "max": 5.0},"ref": theta_star[2], "proposal": 0.05, "latex": r"A_{\mathrm{IA}}"},
    "eta_IA" : {"prior": {"min": -5.0, "max": 5.0},"ref": theta_star[3], "proposal": 0.05, "latex": r"\eta_{\mathrm{IA}}"},
    "log_Mc": {"prior": {"min": 9.0, "max": 15.0}, "ref": theta_star[4], "proposal": 0.5, "latex": r"\log M_{\mathrm{c}}"},
    "log_eta": {"prior": {"min": -0.70, "max": 0.70}, "ref": theta_star[5], "proposal": 0.5, "latex": r"\log \eta"},
    "log_beta": {"prior": {"min": -1.0, "max": 0.70}, "ref": theta_star[6], "proposal": 0.5, "latex": r"\log \beta"},
    "log_M1": {"prior": {"min": 9.0, "max": 13.0}, "ref": theta_star[7], "proposal": 0.5, "latex": r"\log M_1"}}

info["sampler"] = {"mcmc": {"Rminus1_stop": 0.03, "max_tries": 1000}}

info["output"] = "lsst_only"

#%%

updated_info, sampler = run(info)

#%%

gd_sample = sampler.products(to_getdist=True, skip_samples=0.3)["sample"]

mean = gd_sample.getMeans()[:8]
covmat = gd_sample.getCovMat().matrix[:8, :8]
print("Mean:")
print(mean)
print("Covariance matrix:")
print(covmat)

#%%

# Plot the marginalised posteriors
g = plots.get_subplot_plotter(subplot_size=3)
g.settings.legend_fontsize = 32
g.settings.axes_labelsize = 30
g.settings.axes_fontsize = 20
g.triangle_plot(gd_sample, ["Om_m", "s8", "log_Mc", "log_eta", "log_beta", "log_M1"], filled=True)
plt.savefig('posteriors_WL_only.png', dpi=100)
plt.show()