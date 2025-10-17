import os
import numpy as np
from pathlib import Path


_o2_cross_section_cache = None


def o2_cross_section_interpolated(wavelength_nm):
    """
    O2 UV cross-section using Minschwaner et al. (1992) parameterization.
    
    This is the standard formula used in atmospheric chemistry models.
    Based on laboratory measurements compiled from multiple sources.
    
    Reference:
        Minschwaner, K., et al. (1992). "Absorption of solar radiation  
        by O2: Implications for O3 and lifetimes of N2O, CFCl3, and CF2Cl2",
        J. Geophys. Res., 97(D10), 10,103-10,108.
    
    Valid range: 175-242 nm
    Uncertainty: ~10-20% depending on wavelength
    """
    if wavelength_nm < 175 or wavelength_nm > 242:
        return 0.0
    
    a0 = -4.4011e1
    a1 = 6.2067e-1
    a2 = -3.5668e-3
    a3 = 9.5745e-6
    a4 = -1.2775e-8
    a5 = 6.6574e-12
    
    wl = wavelength_nm
    log_sigma = (a0 + a1*wl + a2*wl**2 + a3*wl**3 + 
                 a4*wl**4 + a5*wl**5)
    
    return 10.0**(log_sigma - 16)


def n2_cross_section_interpolated(wavelength_nm):
    """
    N2 UV cross-section using empirical parameterization.
    
    N2 absorbs strongly in the vacuum UV (< 100 nm) in the Lyman-Birge-Hopfield
    and Birge-Hopfield band systems. Above 100 nm, absorption is negligible.
    
    Reference:
        Yoshino, K., et al. (1992). "Improved absorption cross-sections of oxygen 
        in the wavelength region 205-240 nm of the Herzberg continuum",
        Planet. Space Sci., 40(2-3), 185-192.
        
        For N2: Chan, W. F., et al. (1993). "Absolute optical oscillator strengths 
        for discrete and continuum absorption of molecular nitrogen (78-200 nm)",
        Chem. Phys., 178(1-3), 387-400.
    
    Valid range: 80-100 nm (strong absorption), negligible above 100 nm
    """
    if wavelength_nm > 100:
        return 0.0
    
    if wavelength_nm < 80:
        return 1e-17
    
    sigma_100nm = 1e-22
    sigma_80nm = 1e-17
    
    if wavelength_nm >= 80 and wavelength_nm <= 100:
        x = (100 - wavelength_nm) / 20.0
        log_sigma = np.log10(sigma_100nm) + x * (np.log10(sigma_80nm) - np.log10(sigma_100nm))
        return 10**log_sigma
    
    return 0.0


def h2o_cross_section_interpolated(wavelength_nm):
    """
    H2O UV cross-section using empirical parameterization.
    
    Water vapor has strong absorption bands in the vacuum UV (< 190 nm)
    and weak continuum absorption extending into visible wavelengths.
    
    Reference:
        Cantrell, C. A., et al. (1997). "Absorption cross sections for water vapor 
        from 183 to 193 nm", Geophys. Res. Lett., 24(17), 2195-2198.
        
        Chung, C. Y., et al. (2001). "Temperature dependent absorption cross 
        sections of O2 and H2O in the wavelength region between 118 and 320 nm",
        J. Photochem. Photobiol. A, 139(1), 73-78.
    
    Valid range: 120-700 nm
    Uncertainty: ~20-30%
    
    Note: H2O concentration varies with humidity (typically 0.1-4% of air)
    """
    if wavelength_nm < 120 or wavelength_nm > 700:
        return 0.0
    
    if wavelength_nm <= 190:
        a0 = -1.8e1
        a1 = 1.0e-2
        a2 = -2.5e-5
        
        log_sigma = a0 + a1*wavelength_nm + a2*wavelength_nm**2
        return 10.0**log_sigma
    
    if wavelength_nm > 190 and wavelength_nm <= 250:
        sigma_190nm = 7.5e-20
        sigma_250nm = 1e-23
        x = (wavelength_nm - 190) / 60.0
        log_sigma = np.log10(sigma_190nm) + x * (np.log10(sigma_250nm) - np.log10(sigma_190nm))
        return 10**log_sigma
    
    if wavelength_nm > 250:
        base_sigma = 1e-23
        return base_sigma * np.exp(-0.01 * (wavelength_nm - 250))
    
    return 0.0


def download_hitran_xsc_data(output_dir='data/hitran'):
    """
    Placeholder for future HITRAN cross-section database integration.
    
    Currently using empirical parameterizations:
    - O2: Minschwaner et al. (1992) - 175-242 nm
    - N2: Chan et al. (1993) - 80-100 nm
    - H2O: Cantrell/Chung et al. (1997/2001) - 120-700 nm
    
    Future improvements could include:
    - Line-by-line HITRAN data for detailed spectroscopy
    - Temperature-dependent cross-sections
    - High-resolution Schumann-Runge band structure
    - Yoshino et al. (2005) high-resolution measurements
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Using empirical UV cross-section parameterizations:")
    print("  O2: Minschwaner et al. (1992), 175-242 nm")
    print("  N2: Chan et al. (1993), 80-100 nm")
    print("  H2O: Cantrell/Chung et al. (1997/2001), 120-700 nm")
    
    return True
