import numpy as np
from scripts.PlanoConvex import PlanoConvex
from scripts import consts as C
from scripts.raytrace_helpers import sample_rays

# Import vectorized tracing if available, fallback to standard
try:
    from scripts.raytrace_helpers_vectorized import trace_system_vectorized as trace_system
    VECTORIZED_AVAILABLE = True
except ImportError:
    from scripts.raytrace_helpers import trace_system
    VECTORIZED_AVAILABLE = False


def _get_r_mm(lens_data):
    """Helper to get radius of curvature from lens data (handles both R_mm and R1_mm)."""
    return lens_data.get('R1_mm', lens_data.get('R_mm'))


def analyze_tolerance(lens_data, result, n_rays=2000, n_samples=21, 
                     z_range_mm=0.5, medium='air'):
    """
    Analyze manufacturing tolerance sensitivity for an optimized lens configuration.
    
    This function tests how sensitive the coupling efficiency is to errors in 
    lens positioning along the optical axis (z-direction).
    
    Parameters:
    -----------
    lens_data : dict
        Dictionary containing lens specifications (lenses[name1], lenses[name2])
    result : dict
        Optimized result containing z_l1, z_l2, z_fiber, lens names, orientation
    n_rays : int
        Number of rays to trace for each configuration
    n_samples : int
        Number of samples for each tolerance parameter (should be odd for symmetry)
    z_range_mm : float
        Range of longitudinal displacement to test (±z_range_mm from optimal)
    medium : str
        Medium for ray tracing ('air', 'argon', 'helium')
    
    Returns:
    --------
    dict containing:
        - 'baseline': baseline coupling efficiency
        - 'z_l1_sensitivity': dict with 'displacements', 'couplings', and 'metric'
        - 'z_l2_sensitivity': dict with 'displacements', 'couplings', and 'metric'
        - 'parameters': dict with test configuration
    """
    
    # Extract baseline configuration
    name1 = result['lens1']
    name2 = result['lens2']
    d1 = lens_data[name1]
    d2 = lens_data[name2]
    
    z_l1_opt = result['z_l1']
    z_l2_opt = result['z_l2']
    z_fiber = result['z_fiber']
    
    # Determine lens orientation from result
    orientation = result.get('orientation', 'ScffcF')
    flipped1 = (orientation == 'SfccfF')
    flipped2 = not flipped1
    
    print(f"\nTolerance Analysis for {name1} + {name2} (Orientation: {orientation})")
    print("="*70)
    print(f"Baseline configuration:")
    print(f"  z_l1 = {z_l1_opt:.3f} mm, z_l2 = {z_l2_opt:.3f} mm, z_fiber = {z_fiber:.3f} mm")
    print(f"  Baseline coupling = {result['coupling']:.4f}")
    print(f"\nTest parameters:")
    print(f"  Longitudinal range: ±{z_range_mm} mm ({n_samples} samples)")
    print(f"  Rays per configuration: {n_rays}")
    if VECTORIZED_AVAILABLE:
        print(f"  Using vectorized ray tracing (fast)")
    else:
        print(f"  Using standard ray tracing (may be slow)")
    print("="*70)
    
    # Generate displacement arrays
    z_displacements = np.linspace(-z_range_mm, z_range_mm, n_samples)
    
    # Sample rays once for all tests (for consistency)
    origins, dirs = sample_rays(n_rays)
    
    # Helper function to evaluate coupling
    def evaluate_coupling(z_l1, z_l2, z_fiber_val):
        lens1 = PlanoConvex(z_l1, _get_r_mm(d1), d1['tc_mm'], d1['te_mm'], 
                           d1['dia']/2.0, flipped=flipped1)
        lens2 = PlanoConvex(z_l2, _get_r_mm(d2), d2['tc_mm'], d2['te_mm'], 
                           d2['dia']/2.0, flipped=flipped2)
        
        accepted, transmission = trace_system(
            origins, dirs, lens1, lens2, z_fiber_val,
            C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD,
            medium, C.PRESSURE_ATM, C.TEMPERATURE_K, C.HUMIDITY_FRACTION
        )
        
        avg_transmission = np.mean(transmission[accepted]) if np.any(accepted) else 0.0
        coupling = (np.count_nonzero(accepted) / n_rays) * avg_transmission
        return coupling
    
    # Baseline coupling (should match result['coupling'] closely)
    baseline_coupling = evaluate_coupling(z_l1_opt, z_l2_opt, z_fiber)
    
    print(f"\nCalculated baseline coupling: {baseline_coupling:.4f}")
    print(f"Original result coupling: {result['coupling']:.4f}")
    diff = abs(baseline_coupling - result['coupling'])
    print(f"Difference: {diff:.4f} ", end='')
    if diff > 0.01:
        print("(large difference - may be due to different n_rays)")
    else:
        print("(good agreement)")
    
    # 1. z_l1 sensitivity
    print("\n[1/2] Testing L1 longitudinal position sensitivity...")
    z_l1_couplings = []
    for i, dz in enumerate(z_displacements):
        coupling = evaluate_coupling(z_l1_opt + dz, z_l2_opt, z_fiber)
        z_l1_couplings.append(coupling)
        if (i+1) % 5 == 0 or i == len(z_displacements)-1:
            print(f"  Progress: {i+1}/{n_samples} samples completed")
    
    # 2. z_l2 sensitivity
    print("[2/2] Testing L2 longitudinal position sensitivity...")
    z_l2_couplings = []
    for i, dz in enumerate(z_displacements):
        coupling = evaluate_coupling(z_l1_opt, z_l2_opt + dz, z_fiber)
        z_l2_couplings.append(coupling)
        if (i+1) % 5 == 0 or i == len(z_displacements)-1:
            print(f"  Progress: {i+1}/{n_samples} samples completed")
    
    print("\nTolerance analysis complete!")
    
    # Calculate sensitivity metrics
    def calculate_sensitivity(displacements, couplings, baseline):
        """Calculate sensitivity metrics from displacement scan"""
        coupling_array = np.array(couplings)
        coupling_drops = baseline - coupling_array
        
        # Find worst case (maximum coupling drop)
        max_drop_idx = np.argmax(coupling_drops)
        max_drop = coupling_drops[max_drop_idx]
        worst_displacement = displacements[max_drop_idx]
        
        # Calculate sensitivity as max drop per unit displacement
        sensitivity = max_drop / abs(worst_displacement) if worst_displacement != 0 else 0
        
        # Find tolerance for 1% coupling drop
        tolerance_1pct = None
        threshold = baseline - 0.01
        for i, c in enumerate(coupling_array):
            if c < threshold:
                tolerance_1pct = abs(displacements[i])
                break
        
        # Calculate RMS coupling drop
        rms_drop = np.sqrt(np.mean(coupling_drops**2))
        
        return {
            'sensitivity': sensitivity,
            'max_drop': max_drop,
            'worst_displacement': worst_displacement,
            'tolerance_1pct': tolerance_1pct,
            'rms_drop': rms_drop
        }
    
    # Compile results
    z_l1_metrics = calculate_sensitivity(z_displacements, z_l1_couplings, baseline_coupling)
    z_l2_metrics = calculate_sensitivity(z_displacements, z_l2_couplings, baseline_coupling)
    
    results = {
        'baseline': baseline_coupling,
        'z_l1_sensitivity': {
            'displacements': z_displacements,
            'couplings': np.array(z_l1_couplings),
            'metrics': z_l1_metrics
        },
        'z_l2_sensitivity': {
            'displacements': z_displacements,
            'couplings': np.array(z_l2_couplings),
            'metrics': z_l2_metrics
        },
        'parameters': {
            'n_rays': n_rays,
            'n_samples': n_samples,
            'z_range_mm': z_range_mm,
            'medium': medium,
            'lens1': name1,
            'lens2': name2,
            'z_l1_opt': z_l1_opt,
            'z_l2_opt': z_l2_opt,
            'z_fiber': z_fiber,
            'orientation': orientation
        }
    }
    
    # Print summary
    print("\n" + "="*70)
    print("Tolerance Sensitivity Summary:")
    print("="*70)
    print(f"{'Parameter':<20} {'Max Drop':<12} {'At Δz':<12} {'Tol(1%)':<12} {'RMS Drop'}")
    print("-"*70)
    
    for name, key in [('L1 z-position', 'z_l1_sensitivity'),
                      ('L2 z-position', 'z_l2_sensitivity')]:
        m = results[key]['metrics']
        tol_str = f"{m['tolerance_1pct']:.3f} mm" if m['tolerance_1pct'] else "N/A"
        print(f"{name:<20} {m['max_drop']:<12.4f} {m['worst_displacement']:>7.3f} mm  "
              f"{tol_str:<12} {m['rms_drop']:.4f}")
    
    print("="*70)
    print("\nKey findings:")
    
    # Identify which lens is more sensitive
    if z_l1_metrics['sensitivity'] > z_l2_metrics['sensitivity']:
        print(f"  - L1 position is MORE sensitive (sens={z_l1_metrics['sensitivity']:.4f}/mm)")
        print(f"  - L1 requires tighter positioning tolerance")
    else:
        print(f"  - L2 position is MORE sensitive (sens={z_l2_metrics['sensitivity']:.4f}/mm)")
        print(f"  - L2 requires tighter positioning tolerance")
    
    # Report tolerance for 1% coupling drop
    if z_l1_metrics['tolerance_1pct']:
        print(f"  - L1 tolerance for 1% coupling drop: ±{z_l1_metrics['tolerance_1pct']:.3f} mm")
    if z_l2_metrics['tolerance_1pct']:
        print(f"  - L2 tolerance for 1% coupling drop: ±{z_l2_metrics['tolerance_1pct']:.3f} mm")
    
    print("="*70)
    
    return results


def save_tolerance_results(results, run_id, output_dir='./results'):
    """
    Save tolerance analysis results to CSV file.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_tolerance()
    run_id : str
        Identifier for this run
    output_dir : str
        Base directory for results
    """
    import pandas as pd
    from pathlib import Path
    
    # Create output directory
    output_path = Path(output_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    params = results['parameters']
    lens_pair = f"{params['lens1']}+{params['lens2']}"
    
    # Save L1 sensitivity data
    df_l1 = pd.DataFrame({
        'displacement_mm': results['z_l1_sensitivity']['displacements'],
        'coupling': results['z_l1_sensitivity']['couplings'],
        'coupling_drop': results['baseline'] - results['z_l1_sensitivity']['couplings']
    })
    csv_l1 = output_path / f"tolerance_L1_{lens_pair}.csv"
    df_l1.to_csv(csv_l1, index=False)
    print(f"\nSaved L1 tolerance data: {csv_l1}")
    
    # Save L2 sensitivity data
    df_l2 = pd.DataFrame({
        'displacement_mm': results['z_l2_sensitivity']['displacements'],
        'coupling': results['z_l2_sensitivity']['couplings'],
        'coupling_drop': results['baseline'] - results['z_l2_sensitivity']['couplings']
    })
    csv_l2 = output_path / f"tolerance_L2_{lens_pair}.csv"
    df_l2.to_csv(csv_l2, index=False)
    print(f"Saved L2 tolerance data: {csv_l2}")
    
    # Save summary metrics
    metrics_data = {
        'lens_pair': [lens_pair],
        'orientation': [params['orientation']],
        'baseline_coupling': [results['baseline']],
        'z_l1_opt_mm': [params['z_l1_opt']],
        'z_l2_opt_mm': [params['z_l2_opt']],
        'z_fiber_mm': [params['z_fiber']],
        'L1_max_drop': [results['z_l1_sensitivity']['metrics']['max_drop']],
        'L1_worst_displacement_mm': [results['z_l1_sensitivity']['metrics']['worst_displacement']],
        'L1_tolerance_1pct_mm': [results['z_l1_sensitivity']['metrics']['tolerance_1pct']],
        'L1_sensitivity_per_mm': [results['z_l1_sensitivity']['metrics']['sensitivity']],
        'L2_max_drop': [results['z_l2_sensitivity']['metrics']['max_drop']],
        'L2_worst_displacement_mm': [results['z_l2_sensitivity']['metrics']['worst_displacement']],
        'L2_tolerance_1pct_mm': [results['z_l2_sensitivity']['metrics']['tolerance_1pct']],
        'L2_sensitivity_per_mm': [results['z_l2_sensitivity']['metrics']['sensitivity']],
        'n_rays': [params['n_rays']],
        'n_samples': [params['n_samples']],
        'z_range_mm': [params['z_range_mm']],
        'medium': [params['medium']]
    }
    
    df_summary = pd.DataFrame(metrics_data)
    csv_summary = output_path / f"tolerance_summary_{lens_pair}.csv"
    df_summary.to_csv(csv_summary, index=False)
    print(f"Saved tolerance summary: {csv_summary}")
    
    return csv_summary
