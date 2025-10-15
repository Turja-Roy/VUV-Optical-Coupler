import numpy as np
from tqdm import tqdm

from .fetcher import write_temp
from .PlanoConvex import PlanoConvex
from . import consts as C
from .visualizers import plot_system_rays
from .raytrace_helpers import sample_rays, trace_system

import logging
from pathlib import Path


# Evaluate a single configuration
# (given lens vertex positions and a fixed fiber z)
def evaluate_config(z_l1, z_l2, origins, dirs, d1, d2, z_fiber, n_rays):
    # Validate positions before evaluating
    if z_l1 < C.SOURCE_TO_LENS_OFFSET:
        return 0.0, np.zeros(n_rays, dtype=bool)  # Invalid configuration
    if z_l2 <= z_l1 + 0.1:  # Minimum spacing between lenses
        return 0.0, np.zeros(n_rays, dtype=bool)  # Invalid configuration
        
    lens1 = PlanoConvex(vertex_z_front=z_l1,
                        R_front_mm=d1['R_mm'],
                        thickness_mm=d1['t_mm'],
                        ap_rad_mm=d1['dia'])
    lens2 = PlanoConvex(vertex_z_front=z_l2,
                        R_front_mm=d2['R_mm'],
                        thickness_mm=d2['t_mm'],
                        ap_rad_mm=d2['dia'])
    accepted = trace_system(origins, dirs, lens1, lens2,
                            z_fiber, C.FIBER_CORE_DIAM_MM/2.0,
                            C.ACCEPTANCE_HALF_RAD)
    coupling = np.count_nonzero(accepted) / n_rays

    return coupling, accepted

# Coarse + refine grid search per lens pair


def run_grid(run_date, lenses, name1, name2,
             coarse_steps=C.COARSE_STEPS, refine_steps=C.REFINE_STEPS,
             n_coarse=C.N_COARSE, n_refine=C.N_REFINE):
    d1 = lenses[name1]
    d2 = lenses[name2]
    f1 = d1['f_mm']
    f2 = d2['f_mm']
    # Generate ray set once per pair for fair comparison
    origins_coarse, dirs_coarse = sample_rays(n_coarse)
    # coarse search ranges:
    # place lens1 roughly near its focal length, lens2 downstream
    z_l1_min = C.SOURCE_TO_LENS_OFFSET
    # z_l1_max = f1 * 2.0
    z_l1_max = f1 * 1.5

    if z_l1_max <= z_l1_min:
        return None

    best = {'coupling': -1}
    for z_l1 in np.linspace(z_l1_min, z_l1_max, coarse_steps):
        # allow lens2 to vary relative to lens1;
        # keep fiber at z_l2 + f2 (imaging plane assumption)
        z_l2_min = z_l1 + f2 * 0.5
        # z_l2_min = z_l1
        z_l2_max = z_l1 + f2 * 2.5
        for z_l2 in np.linspace(z_l2_min, z_l2_max, coarse_steps):
            z_fiber = z_l2 + f2
            coupling, accepted = evaluate_config(z_l1, z_l2,
                                                 origins_coarse, dirs_coarse,
                                                 d1, d2, z_fiber, n_coarse)
            if coupling > best['coupling']:
                best = {'z_l1': z_l1, 'z_l2': z_l2, 'z_fiber': z_fiber,
                        'coupling': coupling, 'accepted': accepted,
                        'origins': origins_coarse, 'dirs': dirs_coarse}

    # refine around best
    z1c = best['z_l1']
    z2c = best['z_l2']
    dz1 = max(0.05, (z_l1_max - z_l1_min) / (coarse_steps-1))
    dz2 = max(0.05, ((z2c - (z1c + f2*0.5)) +
              ((z1c + f2*2.5) - z2c)) / (coarse_steps-1))
    z1_min = max(C.SOURCE_TO_LENS_OFFSET, z1c - dz1*2)
    z1_max = z1c + dz1*2
    # Ensure z2 is after z1 with enough space for the lens
    z2_min = max(z1_min + 0.1, z2c - dz2*2)
    z2_max = z2c + dz2*2
    origins_ref, dirs_ref = sample_rays(n_refine)
    for z_l1 in np.linspace(z1_min, z1_max, refine_steps):
        for z_l2 in np.linspace(z2_min, z2_max, refine_steps):
            z_fiber = z_l2 + f2
            coupling, accepted = evaluate_config(z_l1, z_l2,
                                                 origins_ref, dirs_ref,
                                                 d1, d2, z_fiber, n_refine)
            if coupling > best['coupling']:
                best = {'z_l1': z_l1, 'z_l2': z_l2, 'z_fiber': z_fiber,
                        'coupling': coupling, 'accepted': accepted,
                        'origins': origins_ref, 'dirs': dirs_ref}

    # attach metadata
    best.update({'lens1': name1, 'lens2': name2, 'f1_mm': f1,
                'f2_mm': f2, 'total_len_mm': best['z_fiber']})

    return best


def _setup_logger(run_date: str):
    """
    Return a module logger that appends to logs/run_<run_date>.log.
    If a FileHandler for that file already exists on the logger, reuse it.
    """
    logger = logging.getLogger("raytrace")

    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"run_{run_date}.log"

    # If a FileHandler for this logfile already exists, reuse logger as-is.
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(logfile):
            return logger

    # Configure logger when adding the first handler
    logger.setLevel(logging.INFO)

    # FileHandler defaults to append mode, so no need to pass mode='a'
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def run_combos(lenses, combos, run_date, batch_num=None):
    logger = _setup_logger(run_date)

    for (a, b) in tqdm(combos):
        logger.info(f"\nEvaluating {a} + {b} ...")

        res = run_grid(run_date, lenses, a, b)  # Coarse + refine search

        if res is None:
            logger.warning("Lens 1 focal length too short for placement.")
            continue
        else:
            plot_system_rays(lenses, res, run_date)  # Visualize this combination
            write_temp(res, run_date, batch_num)
            logger.info(f"best coupling={res['coupling']:.4f} at z_l1={
                      res['z_l1']:.2f}, z_l2={res['z_l2']:.2f}")

    # Read all results from temp file and write to a CSV
    results = []
    
    filename = 'temp.json' if batch_num is None else f'temp_batch_{batch_num}.json'
    filepath = f'./results/{run_date}/{filename}'

    try:
        if Path(filepath).exists():
            import json
            import numpy as np
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Convert lists back to numpy arrays where needed
                for result in data:
                    if 'origins' in result:
                        result['origins'] = np.array(result['origins'])
                    if 'dirs' in result:
                        result['dirs'] = np.array(result['dirs'])
                    if 'accepted' in result:
                        result['accepted'] = np.array(result['accepted'])
                results.extend(data)
            Path(filepath).unlink()  # delete temp file
    except Exception as e:
        logger.error(f"Error reading temporary file {filename}: {str(e)}")

    return results
