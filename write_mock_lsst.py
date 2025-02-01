import numpy as np
import pyccl as ccl
import sacc

#%%

# Load the mock LSST data
d = np.load('data/redshift_distributions_lsst.npz')

#%%

z = d['z']
dndz0 = d['dndz']
ndens = d['ndens_arcmin']
n_bins = len(ndens)

#%%

# Remove the data points at z=0 because they will result in negative redshift in finite differences
z = z[1:]

dndz = np.zeros([n_bins, len(z)])
for i in range(n_bins):
    dndz[i, :] = dndz0[i, 1:]

#%%

# Compute the mean redshift for each bin
mean_z = np.zeros(n_bins)
for i in range(n_bins):
    mean_z[i] = np.sum(z * dndz[i, :]) / np.sum(dndz[i, :])
    

# Photo-z errors for LSST are sig_z[i] = 0.001 (1 + mean_z[i])
sig_zs = np.zeros(n_bins)
for i in range(n_bins):
    sig_zs[i] = 0.001 * (1 + mean_z[i])
    
#%%

# Define the cosmology
cosmo = ccl.Cosmology(Omega_c=0.260-0.045,
                      Omega_b=0.045,
                      h=0.685,
                      n_s=0.973,
                      sigma8=0.821)

# Define the ell values
d_ell = 10 # d_ell is the spacing between consecutive ell values
ells = (np.arange(100) + 0.5) * d_ell
n_ell = len(ells)

# Define the k and a arrays used to compute the non-linear power spectrum
a_array = np.linspace(1/(1+1.4), 1, 32)
l10k_array = np.linspace(-3, 1, 255)
k_array = 10**l10k_array

# Calculate the non-linear power spectrum without baryons
pk_array = np.array([ccl.nonlin_matter_power(cosmo, k_array, a) for a in a_array])
pk_no_baryons = ccl.Pk2D(a_arr=a_array, lk_arr=np.log(k_array), pk_arr=np.log(pk_array))

# Now add the impact of baryonic effects using the bacco emulator
bar = ccl.BaryonsBaccoemu(log10_M_c=14.0)
pk_with_baryons = bar.include_baryonic_effects(cosmo, pk_no_baryons)

#%%

# Compute the angular power spectra without the nuisance parameters
C_ells = np.zeros([n_bins, n_bins, n_ell])

for i in range(n_bins):
    noise_i = np.ones(len(ells)) * 0.28**2 / (ndens[i] * (60 * 180 / np.pi)**2) # Convert from arcmin^2 to steradians
    lens_i = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[i,:]), n_samples=255)
    for j in range(n_bins):
        lens_j = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[j,:]), n_samples=255)
        C_ell_ij = ccl.angular_cl(cosmo, lens_i, lens_j, ells, p_of_k_a=pk_with_baryons)
        if i==j: 
            C_ells[i,j,:] = C_ell_ij + noise_i
        else:
            C_ells[i,j,:] = C_ell_ij

#%%

def P_matrix(sig_delta_z, sig_m, n_bins):
    '''
    Computes and returns the matrix P used to update the covariance matrix via the Laplace approximation

    Parameters
    ----------
    sig_delta_z : an array of size equal to n_bins with one sig_delta_z per redshift bin
    sig_m : an array of size equal to n_bins with one sig_m per redshift bin
    n_bins : number of redshift bins

    '''
    P = np.zeros([n_bins*2, n_bins*2])
    P[:5, :5] = np.diag(sig_delta_z**2)
    P[5:, 5:] = np.diag(sig_m**2)            
    return P

#%%

# We use finite differences to compute the derivatives with respect to Delta z

def t_star(err_delta_z, m_vals, n_bins):
    '''
    Calculates the theoretical prediction of the angular power spectrum at the 
    Delta z values plus/minus a small increment

    Parameters
    ----------
    err_delta_z : an array of delta(Delta z) = [Delta z_1, ..., Delta z_5]
    m_vals : an array of the multiplicative bias for each redshift bin
    
    Returns
    -------
    t : array of the theoretical prediction for the angular power spectrum values

    '''
    
    t = []
    for i in range(n_bins):
        noise_i = np.ones(len(ells)) * 0.28**2 / (ndens[i] * (60 * 180 / np.pi)**2)
        lens_i = ccl.WeakLensingTracer(cosmo, dndz=(z+err_delta_z[i], dndz[i,:]), n_samples=255)
        for j in range(n_bins):
            lens_j = ccl.WeakLensingTracer(cosmo, dndz=(z+err_delta_z[j], dndz[j,:]), n_samples=255)
            C_ell = (1 + m_vals[i]) * (1 + m_vals[j]) * ccl.angular_cl(cosmo, lens_i, lens_j, ells, p_of_k_a=pk_with_baryons)
            if i==j:
                C_ell += noise_i
                t.append(C_ell)
            if i<j:
                t.append(C_ell)
            
    t = np.concatenate(t)
    return t


def finite_dif(err_delta_z, m_vals, n_bins):
    '''
    Evaluates the derivatives at the default Delta z values
    using the finite difference method
    
    Parameters
    ----------
    dDz : an array of delta(Delta z)

    '''
    derivs = []
    for i in range(n_bins):
        err_delta_array = np.zeros(n_bins)
        err_delta_array[i] = err_delta_z[i]
        deriv = (t_star(err_delta_array, m_vals, n_bins)-t_star(-err_delta_array, m_vals, n_bins))/(2*err_delta_z[i])
        derivs.append(deriv)
    derivs = np.concatenate(derivs)
    return derivs

#%%

# We can evaluate the derivatives with respect to m analytically

def ana_deriv(m_vals, n_bins, kbin):
    '''
    Returns the analytical derivative of the theoretical prediction 
    with respect to the multiplicative bias parameter m

    '''
    derivs = []
    for i in range(n_bins):
        noise_i = np.ones(len(ells)) * 0.28**2 / (ndens[i] * (60 * 180 / np.pi)**2)
        lens_i = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[i,:]), n_samples=255)
        dik = i == kbin
        for j in range(n_bins):
            lens_j = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[j,:]), n_samples=255)
            cl_nom = ccl.angular_cl(cosmo, lens_i, lens_j, ells, p_of_k_a=pk_with_baryons)
            djk = j == kbin
            C_ell = (dik*(1+m_vals[j])+djk*(1+m_vals[i]))*cl_nom
            if i==j:
                C_ell += noise_i
                derivs.append(C_ell)
            if i<j:
                derivs.append(C_ell)

    derivs = np.concatenate(derivs)
    return derivs

#%%

# We can also compute the derivatives with respect to m via finite differences
# Comparing the analytical and numerical derivatives will provide a sanity check

def num_deriv(m_vals, err_ms, n_bins, kbin):
        
    derivs = []
    for i in range(n_bins):
        noise_i = np.ones(len(ells)) * 0.28**2 / (ndens[i] * (60 * 180 / np.pi)**2)
        lens_i = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[i,:]), n_samples=255)
        dik = i == kbin
        for j in range(n_bins):
            lens_j = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz[j,:]), n_samples=255)
            cl_ij = ccl.angular_cl(cosmo, lens_i, lens_j, ells, p_of_k_a=pk_with_baryons)
            djk = j == kbin
            A = (dik*((1+m_vals[i]+err_ms[i])-(1+m_vals[i]+err_ms[i]))*(1+m_vals[j]))/(2*err_ms[i])
            B = (djk*((1+m_vals[j]+err_ms[j])-(1+m_vals[j]+err_ms[j]))*(1+m_vals[i]))/(2*err_ms[j])
            C_ell = (A + B) * cl_ij
            if i==j:
                C_ell += noise_i
                derivs.append(C_ell)
            if i<j:
                derivs.append(C_ell)
    derivs = np.concatenate(derivs)
    return derivs

#%%

# Check that the analytical and numerical derivatives match
sig_ms = [0.01, 0.01, 0.01, 0.01, 0.01]
err_ms = [0.005, 0.005, 0.005, 0.005, 0.005]
ms = [0.01, 0.01, 0.01, 0.01, 0.01]

ana = ana_deriv(ms, n_bins , 1)
num = num_deriv(ms, err_ms, n_bins ,1)
print(ana-num)

#%%

def T_matrix(err_delta_z, m_vals, n_bins):
    '''
    Computes and returns the matrix P used to update the covariance matrix via the Laplace approximation.

    Parameters
    ----------
    err_delta_z : an array of delta(Delta z) = [Delta z_1, ..., Delta z_5]
    m_vals : an array of the multiplicative bias for each redshift bin
    n_bins : number of redshift bins

    '''
    n_cross = (n_bins * (n_bins + 1)) // 2
    T = np.zeros([n_cross * n_ell, n_bins*2])
    T_dz = finite_dif(err_delta_z, m_vals, n_bins)
    
    for i in range(n_bins):
        T[:,i] = T_dz[i]
        
    for i in range(n_bins, n_bins*2):
        T[:,i] = ana_deriv(m_vals, n_bins, i-n_bins)
    
    return T

#%%

# Now use the functions above to calculate the correction to the covariance matrix M = T P T^T
def M_matrix(T, P):
    return np.matmul(T, np.matmul(P, np.transpose(T)))

#%%

sig_ms = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
P = P_matrix(sig_zs, sig_ms, n_bins)

err_zs = np.array([0.005, 0.005, 0.005, 0.005, 0.005])
ms = np.array([0.01, 0.01, 0.01, 0.01, 0.01])

#%%

# Trial and error to find appropriate values of delta(Delta z) to use
err_zs2 = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
T1 = T_matrix(err_zs, ms, n_bins)
T2 = T_matrix(err_zs2, ms, n_bins)
print(T2-T1) # We find the differences are of the order of e-11

#%%

T = T_matrix(err_zs, ms, n_bins)
M = M_matrix(T, P)

#%%

# Create the covariance matrix with the correction M = T P T^T from the Laplace approximation

def Cov_matrix(M, n_bins, n_ell, fsky=0.1):

    n_cross = (n_bins * (n_bins + 1)) // 2
    covar = np.zeros([n_cross, n_ell, n_cross, n_ell])

    id_ij = 0
    for i in range(n_bins):
        for j in range(i, n_bins):
            id_km = 0
            for k in range(n_bins):
                for m in range(k, n_bins):
                    cl_ik = C_ells[i, k, :]
                    cl_jm = C_ells[j, m, :]
                    cl_im = C_ells[i, m, :]
                    cl_jk = C_ells[j, k, :]
                    # Knox formula
                    cov = (cl_ik * cl_jm + cl_im * cl_jk) / (d_ell * fsky * (2 * ells + 1))
                    covar[id_ij, :, id_km, :] = np.diag(cov)
                    id_km += 1
            id_ij += 1
                    
    return covar.reshape([n_cross * n_ell, n_cross * n_ell]) + M

covar = Cov_matrix(M, n_bins, n_ell)

#%%

# Now create the SACC file
s = sacc.Sacc()

# Add the weak lensing tracers
for i in range(n_bins):
    s.add_tracer('NZ', 'wl_{x}'.format(x=i),
                 quantity='galaxy_shear',
                 spin=2,
                 z=z,
                 nz=dndz[i,:],
                 extra_columns={'error': 0.1*dndz[i,:]},
                 sigma_g=0.28)


# Add the angular power spectra
for i in range(n_bins):
    for j in range(n_bins):
        if i==j:
            noise_i = np.ones(len(ells)) * 0.28**2 / (ndens[i] * (60 * 180 / np.pi)**2)
            s.add_ell_cl('cl_ee', 'wl_{x}'.format(x=i), 'wl_{y}'.format(y=j), ells, C_ells[i,j,:] - noise_i)
        if i<j:
            s.add_ell_cl('cl_ee', 'wl_{x}'.format(x=i), 'wl_{y}'.format(y=j), ells, C_ells[i,j,:])

# Add the covariance
s.add_covariance(covar)

#%%

# Write it to a file
s.save_fits("mock_lsst_data.fits", overwrite=True)