---
title: "Optimization of Two-Lens Coupling Systems for VUV Flashlamp to Fiber Applications"
subtitle: "Ray Tracing and Multi-Algorithm Comparison"
author: "Turja Roy"
institute: "Department of Physics, University of Texas at Arlington"
---

# Overview

- **Problem**: Efficiently couple VUV light from xenon flashlamp into optical fiber
- **Wavelength**: 200 nm (Vacuum Ultraviolet)
- **Approach**: Monte Carlo ray tracing + optimization algorithms
- **Goal**: Maximize coupling efficiency while minimizing system length
- **Methods**: Compare 6 optimization algorithms

---

# Motivation

**Challenges in VUV Fiber Coupling:**

- Large divergence from arc lamp sources (33° half-angle)
- Limited fiber acceptance angle (NA = 0.22, θ = 12.4°)
- Compact source (3 mm diameter arc)
- Small fiber core (1 mm diameter)

**Applications:**

- Spectroscopy
- Photochemistry  
- Materials characterization
- Remote VUV illumination

---

# System Configuration

**Optical Components:**

1. Xenon arc lamp (3 mm diameter, 33° divergence)
2. Protective fused silica window (14.3 mm diameter, 8.7 mm from arc)
3. Two plano-convex lenses (UV-grade fused silica)
4. Multi-mode optical fiber (1 mm core, NA = 0.22)

**Key Challenge:** Design a compact two-lens relay system to efficiently collect and focus divergent VUV light into the fiber

---

# Source Characteristics

**Arc Lamp Model:**

- Extended circular source: r_arc = 1.5 mm
- Position-dependent divergence: θ(r) = θ_max × (r/r_arc)
- Maximum half-angle: θ_max = 33° at window edge
- Rays originate uniformly across arc area

**Physical Constraint:**

- Window aperture limits angular distribution
- Creates coherent geometric emission pattern

---

# Optical Properties

**Fused Silica at 200 nm:**

- Refractive index: n = 1.578 (Sellmeier equation)
- UV-grade material for VUV transmission
- Plano-convex geometry minimizes spherical aberration

**Fiber Acceptance:**

- Core diameter: 1.0 mm
- Numerical aperture: NA = 0.22
- Acceptance angle: θ_accept = 12.4°

**Both spatial AND angular criteria must be satisfied**

---

# Monte Carlo Ray Tracing

**Ray Sampling (N = 1000-2000 rays):**

1. Uniform spatial distribution over arc area
2. Position-dependent angular distribution
3. Azimuthal angles uniformly spaced for coverage

**Ray Generation:**

- Radial position: r_i = √(U_i) × r_arc (inverse transform sampling)
- Azimuthal angle: φ_i = 2πi/N
- Direction half-angle: θ_i = θ_max × (r_i/r_arc)

---

# Geometric Ray Tracing: Overview

**Full 3D Vector Ray Tracing (No Paraxial Approximation)**

**Key Steps:**

1. Ray-Sphere Intersection
2. Vector Refraction at Interfaces
3. Aperture Clipping
4. Lens Propagation Through Variable Thickness

**Why No Paraxial Approximation?**

- Large ray angles (up to 33°)
- Need accurate modeling of spherical aberration
- Finite aperture effects are significant

---

# Ray-Sphere Intersection (1/2)

**Problem:** Find where ray intersects spherical lens surface

**Ray Equation:**

$$\mathbf{p}(t) = \mathbf{o} + t\mathbf{d}$$

- **o** = ray origin (3D point)
- **d** = ray direction (unit vector)
- **t** = distance parameter

**Sphere Equation:**

$$\|\mathbf{p} - \mathbf{c}\|^2 = R^2$$

- **c** = sphere center
- **R** = radius of curvature

---

# Ray-Sphere Intersection (2/2)

**Substitute ray into sphere equation:**

$$\|\mathbf{o} + t\mathbf{d} - \mathbf{c}\|^2 = R^2$$

**Expand to quadratic form:** $at^2 + bt + c = 0$

**Coefficients:**

$$a = 1 \quad \text{(since } \mathbf{d} \text{ is unit vector)}$$

$$b = 2(\mathbf{o} - \mathbf{c}) \cdot \mathbf{d}$$

$$c = \|\mathbf{o} - \mathbf{c}\|^2 - R^2$$

**Discriminant:** $\Delta = b^2 - 4c$

**Solution:**
- If $\Delta < 0$: No intersection (ray misses)
- If $\Delta \geq 0$: $t = \frac{-b - \sqrt{\Delta}}{2}$ (nearest positive)

---

# Aperture Clipping

**Physical Constraint:** Each lens has finite clear aperture

**After Each Intersection, Check:**

$$r = \sqrt{p_x^2 + p_y^2}$$

**Aperture Test:**
- If $r > r_{\text{aperture}}$: Ray is blocked ❌
- If $r \leq r_{\text{aperture}}$: Ray continues ✓

**Impact:**
- Reduces effective collection area
- Causes vignetting at large angles
- Critical for accurate efficiency prediction

---

# Vector Refraction (1/3)

**Snell's Law in Vector Form**

**Given:**
- Incident ray: $\mathbf{d}_{\text{in}}$
- Surface normal: $\mathbf{n}$ (pointing into incident medium)
- Refractive indices: $n_1$ (incident), $n_2$ (transmitted)

**Step 1: Calculate index ratio**

$$\eta = \frac{n_1}{n_2}$$

**Step 2: Find incident angle**

$$\cos\theta_i = -\mathbf{n} \cdot \mathbf{d}_{\text{in}}$$

---

# Vector Refraction (2/3)

**Step 3: Check for total internal reflection**

$$k = 1 - \eta^2(1 - \cos^2\theta_i)$$

**Condition:**
- If $k < 0$: Total internal reflection → Ray rejected
- If $k \geq 0$: Refraction occurs → Continue

**Step 4: Calculate refracted direction**

$$\mathbf{d}_{\text{out}} = \eta \mathbf{d}_{\text{in}} + \left(\eta \cos\theta_i - \sqrt{k}\right)\mathbf{n}$$

**This is the complete vector form of Snell's law**

---

# Vector Refraction (3/3)

**Application to Plano-Convex Lens:**

**Front Surface (Spherical):**
- Normal: $\mathbf{n} = \frac{\mathbf{p} - \mathbf{c}}{R}$ (outward from center)
- Air → Glass: $n_1 = 1.0$, $n_2 = 1.578$

**Back Surface (Planar):**
- Normal: $\mathbf{n} = (0, 0, -1)$ (along z-axis)
- Glass → Air: $n_1 = 1.578$, $n_2 = 1.0$

**Note:** Normal direction is critical for correct refraction!

---

# Lens Propagation Through Variable Thickness

**Challenge:** Plano-convex lens has non-uniform thickness

**Thickness at radial position r:**

$$t_{\text{local}}(r) = t_c - (t_c - t_e) \cdot \frac{r}{r_{\text{ap}}}$$

- $t_c$ = center thickness
- $t_e$ = edge thickness
- $r_{\text{ap}}$ = aperture radius

**Exit point on back surface:**

$$\mathbf{p}_{\text{back}} = \mathbf{p}_{\text{front}} + \frac{t_{\text{local}}}{|d_z|} \mathbf{d}_{\text{refracted}}$$

**Key:** Local thickness depends on where ray enters lens

---

# Complete Ray Tracing Sequence

**For Each Ray Through Complete System:**

1. **Window front:** Air → Glass refraction
2. **Window back:** Glass → Air refraction
3. **Lens 1 front:** Air → Glass + aperture check
4. **Lens 1 back:** Glass → Air + variable thickness
5. **Lens 2 front:** Air → Glass + aperture check
6. **Lens 2 back:** Glass → Air + variable thickness
7. **Fiber face:** Check spatial + angular acceptance

**Total per ray: 6 refractions + 4 aperture checks + 2 acceptance criteria**

**For N = 1000 rays: ~12,000+ geometric calculations**

---

# Fiber Coupling Criteria

**A ray successfully couples if BOTH conditions are met:**

**1. Spatial Criterion:**
   - Ray hits within fiber core
   - √(x² + y²) ≤ 0.5 mm

**2. Angular Criterion:**
   - Ray arrives within acceptance cone
   - θ ≤ 12.4°

**Coupling Efficiency:**
η = N_accepted / N_total

---

# Optimization Problem

**Design Variables:**

- z₁: Position of first lens
- z₂: Position of second lens  
- z_fiber: Position of fiber face

**Objective Function (Multi-objective):**

f(z₁, z₂) = α(1 - η_coupling) + (1-α)(z_fiber/L_norm)

- α = 0.7 (prioritize coupling efficiency)
- L_norm = 80 mm (normalization constant)
- Balances efficiency vs. compactness

---

# Optimization Challenges

**Why This Problem is Difficult:**

- **Non-differentiable**: Discrete ray counting, aperture clipping
- **Expensive**: Each evaluation requires full ray tracing (1000+ rays)
- **Non-convex**: Multiple local minima possible
- **High-dimensional**: 3 continuous parameters with complex interactions

**Solution Strategy:**

Compare multiple optimization algorithms to find best approach

---

# Optimization Algorithms (1/2)

**1. Grid Search (Baseline)**
   - Exhaustive 2-stage search (coarse + fine)
   - 130 function evaluations per lens pair
   - Reliable but computationally expensive

**2. Powell's Method**
   - Derivative-free local optimization
   - Conjugate direction search
   - Fast convergence for smooth functions
   - 200 max iterations

**3. Nelder-Mead Simplex**
   - Simplex geometric transformations
   - Robust to function noise
   - No derivative required
   - 200 max iterations

---

# Optimization Algorithms (2/2)

**4. Differential Evolution**
   - Population-based global optimizer
   - Evolutionary strategy with mutation
   - Population size: 10, Max iterations: 50
   - Good exploration of parameter space

**5. Dual Annealing**
   - Combines simulated annealing + local search
   - Probabilistic acceptance of worse solutions
   - Adaptive cooling schedule
   - 300 max iterations

**6. Bayesian Optimization**
   - Gaussian process surrogate model
   - Expected improvement acquisition function
   - Sample-efficient for expensive objectives
   - 100 total evaluations (20 random + 80 guided)

---

# Algorithm Comparison

| Algorithm | Type | Evaluations | Strengths | Weaknesses |
|-----------|------|-------------|-----------|------------|
| Grid Search | Exhaustive | 130 | Reliable, global | Expensive |
| Powell | Local | ~50-100 | Fast convergence | Local minima |
| Nelder-Mead | Local | ~50-100 | Noise robust | Local minima |
| Diff. Evolution | Global | ~500 | Global search | Expensive |
| Dual Annealing | Global | ~300 | Escape local minima | Moderate cost |
| Bayesian | Global | 100 | Sample efficient | Setup complexity |

---

# Model Assumptions

**Simplifications Made:**

1. Geometric optics regime (λ << apertures)
2. Coherent source model (idealized angular distribution)
3. Perfect optical surfaces (no manufacturing errors)
4. No optical losses (Fresnel reflections, absorption neglected)
5. Monochromatic light (200 nm only)
6. Perfect alignment (no tilt or decentration)
7. Uniform fiber acceptance across core

**Impact:** Results represent upper bound on coupling efficiency

---

# Key Results Insights

**Trade-offs Identified:**

- **Local vs. Global**: Fast convergence vs. solution quality
- **Cost vs. Quality**: Function evaluations vs. optimality
- **Exploration vs. Exploitation**: Broad search vs. refinement

**Algorithm Selection Depends On:**

- Available computational budget
- Need for global optimum
- Function evaluation cost
- Initial guess quality

---

# Implementation Details

**Software Stack:**

- Python with NumPy for numerical computations
- SciPy for optimization algorithms
- Scikit-optimize for Bayesian optimization
- Monte Carlo ray tracing from scratch

**Computational Approach:**

- Modular design for easy algorithm swapping
- Parallel batch processing for lens pair combinations
- Automated result logging and visualization

---

# Future Extensions

**Potential Improvements:**

1. Include Fresnel reflection and absorption losses
2. Tolerance analysis via Monte Carlo perturbations
3. Extend to 3+ lens systems
4. Multi-wavelength optimization for broad-spectrum sources
5. Experimental validation with physical prototypes
6. Machine learning surrogate models for faster optimization

---

# Conclusions

**Key Achievements:**

- Developed comprehensive Monte Carlo ray tracing framework
- Full geometric optics (no paraxial approximation)
- Systematic comparison of 6 optimization algorithms
- Modular, extensible computational tool

**Insights:**

- Algorithm choice depends on budget and requirements
- Global methods more expensive but thorough
- Bayesian optimization offers good middle ground
- Framework applicable to other VUV optical design problems

---

# Thank You

**Questions?**

Contact: Turja Roy
Department of Physics
University of Texas at Arlington

---
