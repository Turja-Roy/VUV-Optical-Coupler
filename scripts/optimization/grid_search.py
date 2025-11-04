import numpy as np
from tqdm import tqdm

from scripts.data_io import write_temp
from scripts.PlanoConvex import PlanoConvex
from scripts import consts as C
from scripts.visualizers import plot_system_rays
from scripts.raytrace_helpers import sample_rays
from scripts.raytrace_helpers_vectorized import trace_system_vectorized as trace_system

import logging
from pathlib import Path


DEFAULT_COARSE_STEPS = 7
DEFAULT_REFINE_STEPS = 9
DEFAULT_N_COARSE = 500
DEFAULT_N_REFINE = 1000


def evaluate_config(z_l1, z_l2, origins, dirs, d1, d2, z_fiber, n_rays, medium='air', flipped1=False, flipped2=True):
    if z_l1 < C.SOURCE_TO_LENS_OFFSET:
        return 0.0, np.zeros(n_rays, dtype=bool)
    if z_l2 <= z_l1 + d1['tc_mm'] + 0.5:
        return 0.0, np.zeros(n_rays, dtype=bool)
        
    lens1 = PlanoConvex(vertex_z_front=z_l1,
                        R_front_mm=d1['R_mm'],
                        center_thickness_mm=d1['tc_mm'],
                        edge_thickness_mm=d1['te_mm'],
                        ap_rad_mm=d1['dia']/2.0,
                        flipped=flipped1)
    lens2 = PlanoConvex(vertex_z_front=z_l2,
                        R_front_mm=d2['R_mm'],
                        center_thickness_mm=d2['tc_mm'],
                        edge_thickness_mm=d2['te_mm'],
                        ap_rad_mm=d2['dia']/2.0,
                        flipped=flipped2)
    accepted, transmission = trace_system(origins, dirs, lens1, lens2,
                            z_fiber, C.FIBER_CORE_DIAM_MM/2.0,
                            C.ACCEPTANCE_HALF_RAD,
                            medium, C.PRESSURE_ATM, C.TEMPERATURE_K, C.HUMIDITY_FRACTION)
    avg_transmission = np.mean(transmission[accepted]) if np.any(accepted) else 0.0
    coupling = (np.count_nonzero(accepted) / n_rays) * avg_transmission

    return coupling, accepted


def run_grid(run_id, lenses, name1, name2,
             coarse_steps=DEFAULT_COARSE_STEPS, refine_steps=DEFAULT_REFINE_STEPS,
             n_coarse=DEFAULT_N_COARSE, n_refine=DEFAULT_N_REFINE, medium='air'):
    d1 = lenses[name1]
    d2 = lenses[name2]
    f1 = d1['f_mm']
    f2 = d2['f_mm']
    
    if max(C.SOURCE_TO_LENS_OFFSET + 5.0, f1 * 1.5) <= C.SOURCE_TO_LENS_OFFSET:
        return None
    
    # Test both orientations
    results = []
    orientations = [
        (False, True, 'ScffcF'),   # lens1 curved-first, lens2 flat-first
        (True, False, 'SfccfF')    # lens1 flat-first, lens2 curved-first
    ]
    
    for flipped1, flipped2, orientation_name in orientations:
        origins_coarse, dirs_coarse = sample_rays(n_coarse)
        z_l1_min = C.SOURCE_TO_LENS_OFFSET
        z_l1_max = max(C.SOURCE_TO_LENS_OFFSET + 5.0, f1 * 1.5)

        best = {'coupling': -1}
        for z_l1 in np.linspace(z_l1_min, z_l1_max, coarse_steps):
            z_l2_min = z_l1 + f2 * 0.5
            z_l2_max = z_l1 + f2 * 2.5
            for z_l2 in np.linspace(z_l2_min, z_l2_max, coarse_steps):
                z_fiber = z_l2 + f2
                coupling, accepted = evaluate_config(z_l1, z_l2,
                                                     origins_coarse, dirs_coarse,
                                                     d1, d2, z_fiber, n_coarse, medium,
                                                     flipped1, flipped2)
                if coupling > best['coupling']:
                    best = {'z_l1': z_l1, 'z_l2': z_l2, 'z_fiber': z_fiber,
                            'coupling': coupling, 'accepted': accepted,
                            'origins': origins_coarse, 'dirs': dirs_coarse}

        z1c = best['z_l1']
        z2c = best['z_l2']
        dz1 = max(0.05, (z_l1_max - z_l1_min) / (coarse_steps-1))
        dz2 = max(0.05, ((z2c - (z1c + f2*0.5)) +
                  ((z1c + f2*2.5) - z2c)) / (coarse_steps-1))
        z1_min = max(C.SOURCE_TO_LENS_OFFSET, z1c - dz1*2)
        z1_max = z1c + dz1*2
        z2_min = max(z1_min + 0.1, z2c - dz2*2)
        z2_max = z2c + dz2*2
        origins_ref, dirs_ref = sample_rays(n_refine)
        for z_l1 in np.linspace(z1_min, z1_max, refine_steps):
            for z_l2 in np.linspace(z2_min, z2_max, refine_steps):
                z_fiber = z_l2 + f2
                coupling, accepted = evaluate_config(z_l1, z_l2,
                                                     origins_ref, dirs_ref,
                                                     d1, d2, z_fiber, n_refine, medium,
                                                     flipped1, flipped2)
                if coupling > best['coupling']:
                    best = {'z_l1': z_l1, 'z_l2': z_l2, 'z_fiber': z_fiber,
                            'coupling': coupling, 'accepted': accepted,
                            'origins': origins_ref, 'dirs': dirs_ref}

        best.update({'lens1': name1, 'lens2': name2, 'f1_mm': f1,
                    'f2_mm': f2, 'total_len_mm': best['z_fiber'],
                    'orientation': orientation_name})
        results.append(best)
    
    # Return the best orientation based on coupling
    best_result = max(results, key=lambda x: x['coupling'])
    return best_result


def _setup_logger(run_id: str):
    logger = logging.getLogger("raytrace")

    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"run_{run_id}.log"

    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(logfile):
            return logger

    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def run_combos(lenses, combos, run_id, batch_num=None, medium='air', db=None):
    logger = _setup_logger(run_id)
    
    # Initialize database run if provided and not already exists
    if db is not None:
        import scripts.consts as C
        existing_run = db.get_run(run_id)
        if not existing_run:
            config = {'method': 'grid_search'}
            db.insert_run(
                run_id=run_id,
                method='grid_search',
                medium=medium,
                n_rays=C.N_RAYS,
                wavelength_nm=C.WAVELENGTH_NM,
                pressure_atm=C.PRESSURE_ATM,
                temperature_k=C.TEMPERATURE_K,
                humidity_fraction=C.HUMIDITY_FRACTION,
                config=config
            )

    for (a, b) in tqdm(combos):
        logger.info(f"\nEvaluating {a} + {b} ...")

        res = run_grid(run_id, lenses, a, b, medium=medium)

        if res is None:
            logger.warning("Lens 1 focal length too short for placement.")
            continue
        else:
            plot_system_rays(lenses, res, run_id)
            write_temp(res, run_id, batch_num)
            
            # Save to database immediately if enabled
            if db is not None:
                try:
                    result_with_method = dict(res, method='grid_search')
                    db.insert_result(run_id, result_with_method)
                except Exception as e:
                    logger.error(f"Failed to write to database: {e}")
            
            logger.info(f"best coupling={res['coupling']:.4f} at z_l1={
                      res['z_l1']:.2f}, z_l2={res['z_l2']:.2f}")

    results = []
    
    filename = 'temp.json' if batch_num is None else f'temp_batch_{batch_num}.json'
    filepath = f'./results/{run_id}/{filename}'

    try:
        if Path(filepath).exists():
            import json
            import numpy as np
            with open(filepath, 'r') as f:
                data = json.load(f)
                for result in data:
                    if 'origins' in result:
                        result['origins'] = np.array(result['origins'])
                    if 'dirs' in result:
                        result['dirs'] = np.array(result['dirs'])
                    if 'accepted' in result:
                        result['accepted'] = np.array(result['accepted'])
                results.extend(data)
            Path(filepath).unlink()
    except Exception as e:
        logger.error(f"Error reading temporary file {filename}: {str(e)}")

    return results
