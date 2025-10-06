import sys
import numpy as np
from tqdm import tqdm
from scripts import consts as C
from scripts.fetcher import find_combos, particular_combo
from scripts.runner import run_grid
from scripts.visualizers import plot_spot_diagram


def main():
    if len(sys.argv) == 3:  # Particular lenses
        combos, lenses = particular_combo(sys.argv[1], sys.argv[2])
    elif sys.argv[1] == 'select':  # Select candidates
        combos, lenses = find_combos('select')
    else:  # All lens combos
        combos, lenses = find_combos('combine')

    """
        Run sweep across all combos (coarse+refine)
    """
    results = []
    print(f"Running coarse+refined grid sweep for {
          len(combos)} combos (this may take from a few minutes to hours)...")
    for (a, b) in tqdm(combos):
        print(f"\nEvaluating {a} + {b} ...")
        res = run_grid(lenses, a, b, coarse_steps=7,
                       refine_steps=9, n_coarse=2000, n_refine=6000)

        if res is None:
            print("Lens 1 focal length too short for placement.")
            continue
        else:
            print(f"best coupling={res['coupling']:.4f} at z_l1={
                  res['z_l1']:.2f}, z_l2={res['z_l2']:.2f}")

        results.append(res)

    """
        Build a results DataFrame and print summary
    """
    rows = [{k: v for k, v in r.items()
             if k in ['lens1', 'lens2', 'f1_mm', 'f2_mm',
                      'z_l1', 'z_l2', 'z_fiber', 'total_len_mm', 'coupling']}
            for r in results]
    df = __import__("pandas").DataFrame(rows).sort_values(
        ['coupling', 'total_len_mm'],
        ascending=[False, True]).reset_index(drop=True)
    print('\nSummary (coarse+refined search):')
    print(df.to_string(index=False))

    # pick best overall
    best = results[np.argmax([r['coupling'] for r in results])]
    print('\nBest combo overall:', best['lens1'],
          best['lens2'], 'coupling =', best['coupling'])

    # Generate spot diagram for best combo
    plot_spot_diagram(best, lenses)

    # Save summary table to CSV and latex
    if not __import__("os").path.exists('./results_' + C.DATE_STR):
        __import__("os").makedirs('./results_' + C.DATE_STR)
    df.to_csv('./results_' + C.DATE_STR + '/two_lens_coupling_summary.csv',
              index=False)
    df.to_latex('./results_' + C.DATE_STR + '/two_lens_coupling_summary.tex',
                index=False)


if __name__ == "__main__":
    main()
