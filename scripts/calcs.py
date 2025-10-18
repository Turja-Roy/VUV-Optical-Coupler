def fused_silica_n(lambda_nm):
    """Compute fused silica refractive index (Malitson/Sellmeier)"""
    lam_um = lambda_nm / 1000.0
    B1 = 0.6961663
    B2 = 0.4079426
    B3 = 0.8974794
    C1 = 0.0684043**2
    C2 = 0.1162414**2
    C3 = 9.896161**2
    lam2 = lam_um*lam_um
    n2 = 1 + B1*lam2/(lam2 - C1) + B2*lam2/(lam2 - C2) + B3*lam2/(lam2 - C3)

    return __import__("math").sqrt(n2)


def medium_refractive_index(wavelength_nm, medium, pressure_atm=1.0, temp_k=293.15):
    """
    Calculate wavelength-dependent refractive index of propagation medium.
    
    Uses empirical formulas for UV wavelengths (180-300 nm).
    Based on Edlen equation and NIST data for standard conditions.
    """
    lam_um = wavelength_nm / 1000.0
    
    if medium == 'air':
        s = 1.0 / lam_um
        n_minus_1 = (8342.54 + 2406147.0 / (130.0 - s**2) + 15998.0 / (38.9 - s**2)) * 1e-8
        n_stp = 1.0 + n_minus_1
        n = 1.0 + (n_stp - 1.0) * (pressure_atm / 1.0) * (293.15 / temp_k)
        return n
    
    elif medium == 'argon':
        s = 1.0 / lam_um
        n_minus_1 = (6432.8 + 2949810.0 / (146.0 - s**2) + 25540.0 / (417.0 - s**2)) * 1e-8
        n_stp = 1.0 + n_minus_1
        n = 1.0 + (n_stp - 1.0) * (pressure_atm / 1.0) * (293.15 / temp_k)
        return n
    
    elif medium == 'helium':
        n_stp = 1.0 + 3.48e-5
        n = 1.0 + (n_stp - 1.0) * (pressure_atm / 1.0) * (293.15 / temp_k)
        return n
    
    else:
        return 1.0


def o2_cross_section_minschwaner(wavelength_nm):
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


def o2_cross_section(wavelength_nm):
    try:
        from scripts.hitran_data import o2_cross_section_interpolated
        return o2_cross_section_interpolated(wavelength_nm)
    except:
        return o2_cross_section_minschwaner(wavelength_nm)


def n2_cross_section_empirical(wavelength_nm):
    try:
        from scripts.hitran_data import n2_cross_section_interpolated
        return n2_cross_section_interpolated(wavelength_nm)
    except:
        if wavelength_nm > 100:
            return 0.0
        return 0.0


def h2o_cross_section(wavelength_nm):
    try:
        from scripts.hitran_data import h2o_cross_section_interpolated
        return h2o_cross_section_interpolated(wavelength_nm)
    except:
        return 0.0


def number_density(pressure_atm, temp_k):
    P_pa = pressure_atm * 101325.0
    k_B = 1.380649e-23
    n_m3 = P_pa / (k_B * temp_k)
    n_cm3 = n_m3 * 1e-6
    return n_cm3


def calculate_attenuation_coefficient(wavelength_nm, medium, pressure_atm=1.0, temp_k=293.15, humidity_fraction=0.01):
    if medium == 'argon' or medium == 'helium':
        return 0.0
    
    if medium == 'air':
        n_total = number_density(pressure_atm, temp_k)
        
        n_h2o = humidity_fraction * n_total
        n_dry = (1.0 - humidity_fraction) * n_total
        n_o2 = 0.21 * n_dry
        n_n2 = 0.78 * n_dry
        
        sigma_o2 = o2_cross_section(wavelength_nm)
        sigma_n2 = n2_cross_section_empirical(wavelength_nm)
        sigma_h2o = h2o_cross_section(wavelength_nm)
        
        alpha_cm = sigma_o2 * n_o2 + sigma_n2 * n_n2 + sigma_h2o * n_h2o
        return alpha_cm / 10.0
    
    return 0.0


def transmission_through_medium(distance_mm, wavelength_nm, medium, pressure_atm=1.0, temp_k=293.15, humidity_fraction=0.01):
    alpha = calculate_attenuation_coefficient(wavelength_nm, medium, pressure_atm, temp_k, humidity_fraction)
    
    if alpha == 0.0:
        return 1.0
    
    import math
    return math.exp(-alpha * distance_mm)
