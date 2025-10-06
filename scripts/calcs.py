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
