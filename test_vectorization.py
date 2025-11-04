#!/usr/bin/env python3
"""
Test script to verify vectorized ray tracing implementation and measure speedup.

Usage:
    python test_vectorization.py [--rays N] [--trials M]

This script:
1. Loads two test lenses (LA4001 and LA4647)
2. Generates sample rays
3. Traces rays using both serial and vectorized implementations
4. Compares results for correctness
5. Measures and reports performance improvement
"""

import numpy as np
import time
import argparse
from scripts.data_io import particular_combo
from scripts.PlanoConvex import PlanoConvex
from scripts.raytrace_helpers import sample_rays, trace_system
from scripts.raytrace_helpers_vectorized import trace_system_vectorized
from scripts import consts as C


def main():
    parser = argparse.ArgumentParser(description='Test vectorized ray tracing')
    parser.add_argument('--rays', type=int, default=1000, 
                       help='Number of rays to trace (default: 1000)')
    parser.add_argument('--trials', type=int, default=3,
                       help='Number of timing trials (default: 3)')
    args = parser.parse_args()
    
    n_rays = args.rays
    n_trials = args.trials
    
    print("=" * 70)
    print("VECTORIZED RAY TRACING PERFORMANCE TEST")
    print("=" * 70)
    print(f"Number of rays: {n_rays}")
    print(f"Number of trials: {n_trials}")
    print()
    
    # Load test lenses
    print("Loading test lenses (LA4001 + LA4647)...")
    combos, lenses = particular_combo('LA4001', 'LA4647')
    lens1_name, lens2_name = combos[0]
    
    # Test configuration (from default optimization parameters)
    z_l1 = 50.0  # mm
    z_l2 = 100.0  # mm
    z_fiber = 150.0  # mm
    medium = 'air'
    pressure_atm = 1.0
    temp_k = 293.15
    humidity_fraction = 0.01
    
    # Create lens objects
    d1 = lenses[lens1_name]
    d2 = lenses[lens2_name]
    lens1 = PlanoConvex(z_l1, d1['R_mm'], d1['tc_mm'], d1['te_mm'], d1['dia']/2.0, flipped=False)
    lens2 = PlanoConvex(z_l2, d2['R_mm'], d2['tc_mm'], d2['te_mm'], d2['dia']/2.0, flipped=False)
    
    print(f"Lens 1: {lens1_name} (f={lenses[lens1_name]['f_mm']:.1f} mm)")
    print(f"Lens 2: {lens2_name} (f={lenses[lens2_name]['f_mm']:.1f} mm)")
    print()
    
    # Generate sample rays
    print("Generating sample rays...")
    origins, dirs = sample_rays(n_rays)
    
    print(f"Generated {n_rays} rays")
    print()
    
    # =========================================================================
    # Test 1: Correctness check (small sample)
    # =========================================================================
    print("-" * 70)
    print("TEST 1: CORRECTNESS VERIFICATION")
    print("-" * 70)
    
    n_check = min(100, n_rays)
    print(f"Tracing {n_check} rays with both implementations...")
    
    # Serial implementation
    accepted_serial, trans_serial = trace_system(
        origins[:n_check], dirs[:n_check], lens1, lens2, z_fiber,
        C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD,
        medium, pressure_atm, temp_k, humidity_fraction
    )
    
    # Vectorized implementation
    accepted_vec, trans_vec = trace_system_vectorized(
        origins[:n_check], dirs[:n_check],
        lens1, lens2, z_fiber,
        C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD,
        medium, pressure_atm, temp_k, humidity_fraction
    )
    
    # Compare results
    acceptance_match = np.sum(accepted_serial == accepted_vec)
    acceptance_rate = acceptance_match / n_check * 100
    
    trans_diff = np.abs(trans_serial - trans_vec)
    max_trans_diff = np.max(trans_diff)
    mean_trans_diff = np.mean(trans_diff)
    
    print(f"Results comparison:")
    print(f"  Acceptance match: {acceptance_match}/{n_check} ({acceptance_rate:.1f}%)")
    print(f"  Transmission difference:")
    print(f"    Max:  {max_trans_diff:.2e}")
    print(f"    Mean: {mean_trans_diff:.2e}")
    
    if acceptance_rate >= 95.0 and max_trans_diff < 1e-6:
        print(f"  ✓ CORRECTNESS CHECK PASSED")
    else:
        print(f"  ✗ WARNING: Results may differ significantly!")
    
    print()
    
    # =========================================================================
    # Test 2: Performance benchmark
    # =========================================================================
    print("-" * 70)
    print("TEST 2: PERFORMANCE BENCHMARK")
    print("-" * 70)
    
    # Benchmark serial implementation
    print(f"Running serial implementation ({n_trials} trials)...")
    serial_times = []
    for trial in range(n_trials):
        t_start = time.time()
        trace_system(
            origins, dirs, lens1, lens2, z_fiber,
            C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD,
            medium, pressure_atm, temp_k, humidity_fraction
        )
        t_end = time.time()
        elapsed = t_end - t_start
        serial_times.append(elapsed)
        print(f"  Trial {trial+1}/{n_trials}: {elapsed:.3f} s")
    
    serial_mean = np.mean(serial_times)
    serial_std = np.std(serial_times)
    print(f"  Mean: {serial_mean:.3f} ± {serial_std:.3f} s")
    print()
    
    # Benchmark vectorized implementation
    print(f"Running vectorized implementation ({n_trials} trials)...")
    vec_times = []
    for trial in range(n_trials):
        t_start = time.time()
        trace_system_vectorized(
            origins, dirs, lens1, lens2, z_fiber,
            C.FIBER_CORE_DIAM_MM/2.0, C.ACCEPTANCE_HALF_RAD,
            medium, pressure_atm, temp_k, humidity_fraction
        )
        t_end = time.time()
        elapsed = t_end - t_start
        vec_times.append(elapsed)
        print(f"  Trial {trial+1}/{n_trials}: {elapsed:.3f} s")
    
    vec_mean = np.mean(vec_times)
    vec_std = np.std(vec_times)
    print(f"  Mean: {vec_mean:.3f} ± {vec_std:.3f} s")
    print()
    
    # Calculate speedup
    speedup = serial_mean / vec_mean
    
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Serial time:     {serial_mean:.3f} ± {serial_std:.3f} s")
    print(f"Vectorized time: {vec_mean:.3f} ± {vec_std:.3f} s")
    print(f"Speedup:         {speedup:.1f}x")
    print()
    
    if speedup >= 10.0:
        print(f"✓ EXCELLENT: {speedup:.1f}x speedup achieved!")
    elif speedup >= 5.0:
        print(f"✓ GOOD: {speedup:.1f}x speedup achieved")
    elif speedup >= 2.0:
        print(f"⚠ MODERATE: {speedup:.1f}x speedup (expected 10-50x)")
    else:
        print(f"✗ POOR: Only {speedup:.1f}x speedup (expected 10-50x)")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
