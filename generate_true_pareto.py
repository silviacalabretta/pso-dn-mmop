import argparse
import time
import numpy as np
from tqdm import tqdm

from src.problem import LocationProblem
from data.test_map import facilities, x_bounds


def is_non_dominated_vectorized(F):
    n = F.shape[0]
    if n <= 1:
        return np.ones(n, dtype=bool)
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dominated = np.any(le & lt, axis=0)
    return ~dominated

def kung_pareto_filter(F, threshold=2000):
    """Kung's divide-and-conquer for maximal vectors (Pareto front).
    O(n log n) expected for low dimensions.
    """
    n = F.shape[0]
    if n <= 1:
        return np.ones(n, dtype=bool)
    if n <= threshold:
        return is_non_dominated_vectorized(F)

    order = np.argsort(F[:, 0])
    F_sorted = F[order]

    mid = n // 2
    L = F_sorted[:mid]
    R = F_sorted[mid:]

    L_mask = kung_pareto_filter(L, threshold)
    R_mask = kung_pareto_filter(R, threshold)

    L_front = L[L_mask]
    R_candidates = R[R_mask]

    if len(L_front) > 0 and len(R_candidates) > 0:
        keep = np.ones(len(R_candidates), dtype=bool)
        for i in range(0, len(R_candidates), 500):
            batch = R_candidates[i:i + 500]
            dominated = np.zeros(len(batch), dtype=bool)
            for j in range(0, len(L_front), 2000):
                L_chunk = L_front[j:j + 2000]
                le_all = np.all(L_chunk[:, None, :] <= batch[None, :, :], axis=2)
                lt_any = np.any(L_chunk[:, None, :] < batch[None, :, :], axis=2)
                dominated |= np.any(le_all & lt_any, axis=0)
            keep[i:i + 500] = ~dominated

        R_mask_filtered = np.zeros(len(R), dtype=bool)
        R_mask_filtered[R_mask] = keep
        R_mask = R_mask_filtered

    result = np.zeros(n, dtype=bool)
    result[:mid] = L_mask
    result[mid:] = R_mask

    final = np.zeros(n, dtype=bool)
    final[order] = result
    return final

def make_regions_grid(step):
    regions = [
        (10.0, 20.0, 75.0, 85.0),
        (45.0, 60.0, 20.0, 40.0),
        (30.0, 60.0, 55.0, 100.0),
    ]
    parts = []
    total = 0
    for i, (x1, x2, y1, y2) in enumerate(regions):
        xs = np.arange(x1, x2 + 1e-10, step)
        ys = np.arange(y1, y2 + 1e-10, step)
        xx, yy = np.meshgrid(xs, ys)
        part = np.column_stack((xx.ravel(), yy.ravel()))
        parts.append(part)
        total += len(part)
        print(f"  Region {i+1}: [{x1},{x2}]x[{y1},{y2}] -> {len(part):,} pts")
    X_grid = np.concatenate(parts, axis=0)
    print(f"  Total grid points: {total:,}")
    return X_grid


def main():
    parser = argparse.ArgumentParser(
        description="Generate true Pareto set of the map-based location problem by brute-force grid search."
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Grid step size (default: 0.05)",
    )
    args = parser.parse_args()
    step = args.step

    print("=" * 60)
    print("True Pareto Set Generation")
    print("=" * 60)
    print(f"Grid step: {step}")
    print()

    print("Initializing problem...")
    problem = LocationProblem(facilities=facilities, x_bounds=x_bounds)
    print(f"  Facilities: {list(problem.facilities.keys())}")
    print(f"  Objectives (M): {problem.M}")
    print(f"  Decision vars (D): {problem.D}")
    print()

    print("Generating grid in Pareto-relevant regions...")
    X_grid = make_regions_grid(step)
    N = len(X_grid)
    print()

    print("Evaluating objectives (batched)...")
    BATCH_SIZE = 200000
    F_all = np.empty((N, problem.M), dtype=np.float64)
    for start in tqdm(range(0, N, BATCH_SIZE), desc="Evaluating"):
        end = min(start + BATCH_SIZE, N)
        F_all[start:end] = problem.evaluate_population(X_grid[start:end])
    print(f"  F shape: {F_all.shape}")
    print()

    print("Finding Pareto-optimal points (Kung divide-and-conquer)...")
    t0 = time.perf_counter()
    mask = kung_pareto_filter(F_all)
    X_true = X_grid[mask]
    F_true = F_all[mask]
    t_elapsed = time.perf_counter() - t0
    print(f"  Pareto set size: {len(X_true)}")
    print(f"  Filter time: {t_elapsed:.2f}s")
    print()

    outpath = f"data/true_pareto_step{step}.npz"
    print(f"Saving to {outpath}...")
    np.savez(outpath, X_true=X_true, F_true=F_true)
    print(f"  X_true shape: {X_true.shape}")
    print(f"  F_true shape: {F_true.shape}")
    print(f"  X_true dtype: {X_true.dtype}")
    print(f"  F_true dtype: {F_true.dtype}")
    print()

    print("Done! To load in notebook:")
    print(f'  d = np.load("{outpath}")')
    print("  X_true = d['X_true']")
    print("  F_true = d['F_true']")


if __name__ == "__main__":
    main()
