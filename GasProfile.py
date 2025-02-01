import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d

#%%

def M_conv(cosmo, M_500c, z):
    '''
    Convert halo mass definitions from M_500c to M_200c using the concentration-mass relation in Ishiyama21

    '''
    a = 1/(1+z)
    hmd_200c = ccl.halos.MassDef200c
    hmd_500c = ccl.halos.MassDef500c
    conc = ccl.halos.ConcentrationIshiyama21(mass_def=hmd_500c)
    mass_trans = ccl.halos.mass_translator(mass_in=hmd_500c, mass_out=hmd_200c, concentration=conc)
    M_200c = mass_trans(cosmo, M_500c, a)
    return M_200c
    

class GasProfile:
    
    def __init__(self, M_200c, z, cosmo, logMc, logeta, logbeta, logM10):
        
        self._M_200c = M_200c
        self._z = z
        self._cosmo = cosmo
        self._logMc = logMc
        self._logeta = logeta
        self._logbeta = logbeta
        self._logM10 = logM10
        self._a = 1/(1+self._z)
        self._Om_b = self._cosmo['Omega_b']
        self._Om_m = self._cosmo['Omega_c'] - self._cosmo['Omega_b']
        
        m_def = ccl.halos.MassDef200c
        r_200c = m_def.get_radius(self._cosmo, self._M_200c, self._a) / self._a # Comoving distance
        
        theta_out = 10**0.25 
        theta_inn = 10**(-0.86)
        M_inn = 10**13.574
        r_out = r_200c / theta_out
        r_inn = theta_inn * r_200c
        beta_inn = 3 - (M_inn/self._M_200c)**0.31
        
        self._r_200c = r_200c
        self._theta_out = theta_out
        self._theta_inn = theta_inn
        self._M_inn = M_inn
        self._r_out = r_out
        self._r_inn = r_inn
        self._beta_inn = beta_inn
    
        cM = ccl.halos.ConcentrationIshiyama21(mass_def=m_def)
        c = cM(cosmo=cosmo, M=self._M_200c, a=self._a)
        r_s = self._r_200c / c
        self._r_s = r_s
        
    
    def f_star(self):
        '''
        Calculates the stellar mass fraction using the parametrisation and values in Behroozi 2013

        '''
        nu = np.exp(-4*self._a**2)
    
        def g_func(x, a, nu):
            alpha = -1.412 + (0.731 * (a-1)) * nu
            delta = 3.508 + (2.608 * (a-1) - 0.043 * (1/a-1)) * nu
            gamma = 0.316 + (1.319 * (a-1) + 0.279 * (1/a-1)) * nu
            return -np.log(10**(alpha*x)+1) + (delta * (np.log(1+np.exp(x)))**gamma) / (1 + np.exp(10**(-x)))
    
        logM1 = self._logM10 + (-1.793 * (self._a-1) - 0.251 * (1/self._a-1)) * nu 
        M1 = 10**logM1
        logep = -1.777 + (-0.006 * (self._a-1)) * nu - 0.119 * (self._a-1)
        ep = 10**logep
        g0 = g_func(0, self._a, nu)
        g = g_func(np.log(self._M_200c/M1), self._a, nu)
    
        return ep * (M1/self._M_200c) * 10**(g-g0)
    
    
    def f_ICM(self, M):
        '''
        Calculates the theoretical prediction for the bound gas mass fraction for haloes
        of mass M_200c for particular cosmological and baryonic parameters

        '''
        beta = 10**self._logbeta
        return (self._Om_b/self._Om_m - self.f_star()) / (1 + (M/self._M_200c)**beta)
    
    
    def f_ej(self):
        '''
        Calculates the theoretical prediction for the ejected gas mass fraction for haloes
        of mass M_200c
        '''
        return (self._Om_b/self._Om_m) - self.f_star() - self.f_ICM(M=self._M_200c)
    
    
    def rho_nfw(self, r):
        '''
        Returns the unnormalised NFW profile

        '''
        r_tr = 8 * self._r_200c # Truncation radius
        x = r/self._r_s
        tau = r_tr / self._r_s
        return (1/(x*(1+x)**2)) * (1/(1+(x/tau)**2)**2)

        
    def g(self, r):
        '''
        Calculates the unnormalised bound gas density profile by first calculating the 
        normalisation constant for the NFW profile

        '''
        def f(r, r_inn, r_out, beta_inn):
            return (1+r/r_inn)**(-beta_inn) * (1+(r/r_out)**2)**(-2)
        
        # BACCO bound gas profile for r < r_out
        rho_bound = f(r, self._r_inn, self._r_out, self._beta_inn)

        # NFW profile for r > r_out
        B = f(self._r_out, self._r_inn, self._r_out, self._beta_inn) / self.rho_nfw(self._r_out)
        rho_nfw = B * self.rho_nfw(r)
    
        # Combine the two conditions using np.where to select between the two models
        rho = np.where(r < self._r_out, rho_bound, rho_nfw)
        
        return rho
    
    
    def get_A(self):
        '''
        Calculates the overal normalisation costant A of the bound gas density profile

        '''
        def integrand(r):
            return self.g(r) * r**2
        
        # Discretise the range [0, 10*self._r_200c]
        r_values = np.linspace(0, 10 * self._r_200c, 1000)
        integrand_values = integrand(r_values)

        # Use the trapezoidal rule to integrate
        integral = np.trapz(integrand_values, r_values)
        A = self._M_200c * self.f_ICM(M=self._M_200c) * 1 / (4 * np.pi) * integral**(-1)
        
        return A
    
    
    def rho_bgas(self, r, A):
        '''
        Returns the overall bound gas density profile A x g(r)

        '''
        return A * self.g(r) * (1+self._z)**3
    
    
    def rho_egas(self, r):
        '''
        Returns the ejected gas density profile as parametrised in BACCO

        '''
        eta = 10**self._logeta
        r_esc = 0.5 * np.sqrt(200) * self._r_200c
        r_ej = 0.75 * eta * r_esc
        return self._M_200c * self.f_ej() * (2 * np.pi * r_ej**2)**(-3/2) * np.exp(-0.5 * (r/r_ej)**2) * (1+self._z)**3
    
    
    def A_ICM(self, M_piv, M_norm):
        '''
        Finds the theoretical prediction for the parameter A_ICM 
        which gives the normalisation at the pivot mass

        Parameters
        ----------
        M_piv : the median mass of the cluster sample
        M_norm : the mass normalisation = 1e13 solar masses
        f : f_ICM evaluated at the value of M_200c at which M_500c = M_piv

        '''
        M_val = M_conv(cosmo=self._cosmo, M_500c=M_piv, z=self._z)
        f = self.f_ICM(M=M_val)
        return (M_piv/M_norm) * f


    def B_ICM(self):
        '''
        Finds the theoretical prediction for the parameter B_ICM
        which gives the power law index of the gas mass – halo mass relation

        '''
        M_c = 10**self._logMc
        beta = 10**self._logbeta
        return 1 + (beta * (M_c/self._M_200c)**beta) / (1 + (M_c/self._M_200c)**beta)
    

    def Delta_T(self, aperture_radii, A):
        '''
        Calculates the output of the compensated aperture photometry filter on the kSZ temperature map

        '''
        X_H = 0.76
        m_amu = 1.66e-27 / 1.989e30 # In units of solar masses
        T_CMB = 2.726e6 # In micro Kelvin
        v_r = 1.06e-3 # v_r / c
        sigma_T = 6.65e-29 * 1.05e-45 # Convert to Mpc^2
        d_A = ccl.comoving_angular_distance(self._cosmo, self._a) # In Mpc
        
        def integrand(l, theta):
            x = np.sqrt(l**2 + (d_A*theta)**2)
            rho = self.rho_bgas(x, A) + self.rho_egas(x)
            return rho * (X_H + 1) / (2 * m_amu)
        
        def I1(theta):
            l_values = np.linspace(0, 10 * self._r_200c, 1000)
            integrand_values = integrand(l_values, theta)
            result = np.trapz(integrand_values, l_values)
            return result

        theta = np.linspace(0, np.radians(6.0/60), num=128)
        I1s = np.array([I1(th) for th in theta])
        f = interp1d(theta, I1s, bounds_error=False, fill_value=0.0)

        theta = np.linspace(0, np.radians(6.0/60), num=128)
        I1s = np.array([I1(th) for th in theta])
        f = interp1d(theta, I1s, bounds_error=False, fill_value=0.0)
        
        
        # Convolve with beam profile
        
        L = np.radians(12.0/60)
        ngrid = 512
        theta_x = np.linspace(-L/2, L/2, ngrid)
        theta_y = theta_x.copy()
        theta_r = np.sqrt(theta_x[None, :]**2 + theta_y[:, None]**2) # Create a 2D array of the distance from each pixel to the centre
        f2d = f(theta_r)
        fwhm = np.radians(2.1/60)
        sigma_beam = fwhm/2.355
        ell_array = np.fft.fftfreq(ngrid, d=L/(ngrid*2*np.pi))
        
        def b_ell(ell):
            return np.exp(-0.5*(ell*sigma_beam)**2) # Fourier transform of the Gaussian beam
        
        ell_r = np.sqrt(ell_array[None, :]**2 + ell_array[:, None]**2)
        beam_map = b_ell(ell_r)

        f2d_fourier = np.fft.fft2(f2d) # Fourier transform of the original function
        f2d_fourier_beamed = f2d_fourier * beam_map # The convlution becomes a multiplication in Fourier space
        f2d_beamed = np.real(np.fft.ifft2(f2d_fourier_beamed)) # Take the inverse Fourier transform to obtain the map convolved with the beam

        # Form a histogram of the number of cells for a given theta_r
        counts_theta_r, edges_theta_r = np.histogram(theta_r.flatten(), range=(0, np.radians(6.0/60)), bins=128)
        
        # Form a histogram for the same quantity weighted by the value at each pixel of the corresponding function
        counts_f2d_beamed, _ = np.histogram(theta_r.flatten(), range=(0, np.radians(6.0/60)), bins=128, weights=f2d_beamed.flatten())
        
        # Take their ratio to translate back to a one-dimensional array of the average value of the function for all pixel values with similar theta_r
        f1d_beamed = counts_f2d_beamed / counts_theta_r
        
        # Use the result of the beaming to build an interpolator and pass this to the second integral 
        f1d_ip = interp1d(theta, theta * f1d_beamed, bounds_error=False, fill_value=0.0)
        
        
        # Now perform the integral over theta
                
        def I2(start, stop):
            theta_values = np.linspace(start, stop, 1000)
            f1d_values = f1d_ip(theta_values)
            result = np.trapz(f1d_values, theta_values)
            return result

        delta_Ts = []
        for aperture_radius in aperture_radii:
            int2a = I2(0, aperture_radius)
            int2b = I2(aperture_radius, np.sqrt(2) * aperture_radius)
            delta_Ts.append(4 * np.pi * sigma_T * T_CMB * v_r * 1/(1+self._z) * (int2a-int2b))
            
        return np.array(delta_Ts)
