import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
import sacc
from cobaya import run
from getdist import plots
import scipy.optimize as opt

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

# Define functions to compute the theoretical prediction for the stellar mass fraction

M_piv = np.array([4.8e14, 2e14, 1e14])
M_norm = 1e13

def f_star(M_200c, z, M10): 
    '''
    Calculates the stellar mass fraction using the parametrisation and values in Behroozi 2013

    '''
    a = 1/(1+z)
    nu = np.exp(-4*a**2)
    
    def g_func(x, a, nu):
        alpha = -1.412 + (0.731 * (a-1)) * nu
        delta = 3.508 + (2.608 * (a-1) - 0.043*(1/a-1)) * nu
        gamma = 0.316 + (1.319 * (a-1) + 0.279*(1/a-1)) * nu
        return -np.log(10**(alpha*x)+1) + (delta * (np.log(1+np.exp(x)))**gamma) / (1 + np.exp(10**(-x)))
    
    logM1 = M10 + (-1.793 * (a-1) - 0.251*(1/a-1)) * nu 
    M1 = 10**logM1
    logep = -1.777 + (-0.006 * (a-1)) * nu - 0.119 * (a-1)
    ep = 10**logep
    g0 = g_func(0, a, nu)
    g = g_func(np.log(M_200c/M1), a, nu)
    
    return ep * (M1/M_200c) * 10**(g-g0)

#%%

# Define functions to compute the theoretical prediction for the bound gas mass fraction

def f_ICM(M_200c, f_star, Om_m, logMc, logbeta, Om_b=0.045):
    '''
    Finds the theoretical prediction of the bound gas mass fraction for haloes
    of mass M_200c for particular cosmological and baryonic parameters

    Parameters
    ----------
    M_200c : the halo mass enclosed within the radius r_200c, where the average density
                of the halo is 200 times the critical density of the Universe at a given z
                (note that we need to set this to the value at which M_500c = M_piv)
    f_star : the stellar mass fraction

    '''
    M_c = 10**logMc
    beta = 10**logbeta
    return (Om_b/Om_m - f_star) / (1 + (M_c/M_200c)**beta)

def A_ICM(M_piv, M_norm, f):
    '''
    Finds the theoretical prediction for the parameter A_ICM 
    which gives the normalisation at the pivot mass

    Parameters
    ----------
    M_piv : the median mass of the cluster sample
    M_norm : the mass normalisation = 1e13 solar masses
    f : f_ICM evaluated at the value of M_200c at which M_500c = M_piv

    '''
    return (M_piv/M_norm) * f

def B_ICM(M_200c, f_star, Om_m, logMc, logbeta, Om_b=0.045):
    '''
    Finds the theoretical prediction for the parameter B_ICM
    which gives the power law index of the gas mass – halo mass relation

    '''
    M_c = 10**logMc
    beta = 10**logbeta
    return 1 + (beta * (M_c/M_200c)**beta) / (1 + (M_c/M_200c)**beta)


#%%

# Define a function to convert between halo mass definitions

def M_conv(cosmo, M_500c, a):
    '''
    Converts from M_500c to M_200c using the concentration-mass relation in Ishiyama21
    
    ''' 
    hmd_200c = ccl.halos.MassDef200c
    hmd_500c = ccl.halos.MassDef500c
    conc = ccl.halos.ConcentrationIshiyama21(mass_def=hmd_500c) # Use the concentration-mass relation for Delta = 500
    mass_trans = ccl.halos.mass_translator(mass_in=hmd_500c, mass_out=hmd_200c, concentration=conc) # Translate from Delta = 500 to Delta = 200
    M_200c = mass_trans(cosmo, M_500c, a)
    return M_200c

cosmo = ccl.Cosmology(Omega_c=0.260-0.045,
                      Omega_b=0.045,
                      h=0.685,
                      n_s=0.973,
                      sigma8=0.821)

z_med = np.array([0.60, 0.35, 0.30]) # Median redshifts of the three cluster samples ordered from highest to lowest pivot mass
M200s = np.array([M_conv(cosmo, M_piv[i], 1/(1+z)) for i, z in enumerate(z_med)])
print(M200s) 

#%%

# Input the X-ray data
#A_ICMs = np.array([5.69, 1.08, 1.92])
#B_ICMs = np.array([1.33, 1.19, 1.23])
#f_stars = np.array([8.3e-3, , 21.2e-3])
#D = np.concatenate((d, A_ICMs, B_ICMs))

#%%

# Alternatively generate some mock data
z_med = np.array([0.60, 0.35, 0.30])
f_stars = f_star(M_piv, z_med, 11.5)
print(f_stars)

f_ICMs = f_ICM(M_200c=M200s, f_star=f_stars, Om_m=0.260, logMc=14.0, logbeta=-0.22)
A_ICMs = A_ICM(M_piv=M200s, M_norm=M_norm, f=f_ICMs)
B_ICMs = B_ICM(M_200c=M200s, f_star=f_stars, Om_m=0.260, logMc=14.0, logbeta=-0.22)

D = np.concatenate((d, A_ICMs, B_ICMs)) # New data vector that also contains the A_ICMs and B_ICMs

#%%

# Construct the covariance matrix associated with the X-ray data
A_errs = np.array([0.62, 0.13, 0.10])
B_errs = np.array([0.09, 0.01, 0.12])
cov_x = np.zeros([6, 6])
cov_x[:3, :3] = np.diag(A_errs**2)  # Assign squared A_errs to the top-left 3x3 block
cov_x[3:, 3:] = np.diag(B_errs**2)  # Assign squared B_errs to the bottom-right 3x3 block

# Construct the total covariance matrix for the weak lensing and X-ray data
Cov = np.zeros((1506, 1506))
Cov[:1500, :1500] = cov
Cov[1500:, 1500:] = cov_x

#%%

# Compute the inverse of the covariance matrix of the data
icov = np.linalg.inv(Cov)

def A_IA(amp, eta, z, z_star=0.62):
    return amp * ((1+z)/(1+z_star))**eta

# Define the k and a arrays used to compute the non-linear power spectrum
a_array = np.linspace(1/(1+1.4), 1, 32)
l10k_array = np.linspace(-3, 1, 256)
k_array = 10**l10k_array

def lnprob(Om_m, s8, A_IA0, eta_IA, log_Mc, log_eta, log_beta, log_M1, sign=1):
    '''
    Calculates the log-likelihood for the difference between the data values
    and the theoretical prediction for the angular power spectra

    Parameters
    ----------
    Om_m : Omega_m = Omega_c + Omega_b 
    s8 : sigma_8
    log_Mc : describes how much gas has been expelled by AGN
    log_eta : describes how far from the halo virial radius gas has been expelled by AGN
    log_beta : mass fraction of hot gas in haloes
    log_M1 : characteristic halo mass scale for central galaxy
    sign : change to -2 to return the chi-squared instead of the log-likelihood

    Returns
    -------
    logprob: the log-likelihood

    '''
    
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
    
    # We need to set the value M_200c to the value at which M_500c = M_piv
    z_med = np.array([0.60, 0.35, 0.30])
    M200s = np.array([M_conv(cosmo, M_piv[i], 1/(1+z)) for i, z in enumerate(z_med)])
    f_stars = f_star(M_piv, z_med, log_M1)
    
    f_tp = f_ICM(M_200c=M200s, f_star=f_stars, Om_m=Om_m, logMc=log_Mc, logbeta=log_beta)
    A_tp = A_ICM(M_piv=M200s, M_norm=M_norm, f=f_tp)
    B_tp = B_ICM(M_200c=M200s, f_star=f_stars, Om_m=Om_m, logMc=log_Mc, logbeta=log_beta)
    
    M = np.concatenate((m, A_tp, B_tp)) 
    
    diff = D - M
    logprob = -0.5*np.dot(diff, np.dot(icov, diff)) * sign
    return logprob

#%%

likelihood_Mc11 = lnprob(Om_m=0.260, s8=0.821, A_IA0=0, eta_IA=0, log_Mc=11.0, log_eta=0.20, log_beta=-0.22, log_M1=11.5)
likelihood_Mc14 = lnprob(Om_m=0.260, s8=0.821, A_IA0=0, eta_IA=0, log_Mc=14.0, log_eta=0.20, log_beta=-0.22, log_M1=11.5)
print(likelihood_Mc11, likelihood_Mc14)

#%%

M200s = np.array([M_conv(cosmo, M_piv[i], 1/(1+z)) for i, z in enumerate(z_med)])
f_stars = f_star(M_piv, z_med, 11.5)
f_tp_1 = f_ICM(M_200c=M200s, f_star=f_stars, Om_m=0.260, logMc=13.0, logbeta=-0.10)
f_tp_2 = f_ICM(M_200c=M200s, f_star=f_stars, Om_m=0.260, logMc=13.0, logbeta=-0.30)

plt.plot(M200s, f_tp_1, marker='x', label=r'$\beta=-0.1$')
plt.plot(M200s, f_tp_2, marker='x', label=r'$\beta=-0.3$')
plt.xlabel(r'$M_{200\mathrm{c}}$')
plt.ylabel(r'$f_{\mathrm{ICM}}$')
plt.legend()
plt.show()

#%%

# Minimise the logprob by creating a wrapper function
def lprob_min(p):
    return lnprob(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], -1)

# Initial guess
p = np.array([0.27, 0.83, 1.0, 1.0, 11.0, 0.1, -0.2, 11.0])

theta_star = opt.fmin(lambda p: lnprob(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], -1), p)
print(theta_star)

#%%

info= {"likelihood": {"logprob": lnprob}}

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

info["output"] = "lsst_mcmc_with_xrays"

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
plt.savefig('posteriors_WL+X-rays.png', dpi=100)
plt.show()