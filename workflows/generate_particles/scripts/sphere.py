from math import ceil
from typing import Literal

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, reference_states


def construct_fcc_sphere(
    symbol: str,
    natoms: int,
    rounding: Literal["closest", "above", "below"] = "closest",
    zaxis: Literal["c4"] = "c4",
    latticeconstant: float | None = None,
) -> Atoms:
    if rounding not in ("closest", "above", "below"):
        raise ValueError(
            f"Invalid rounding option: {rounding}. "
            "Choose from 'closest', 'above', 'below'"
        )
    if zaxis not in ("c4",):
        raise ValueError("'zaxis' should be 'c4'")

    latticeconstant = (
        reference_states[atomic_numbers[symbol]]["a"]
        if latticeconstant is None
        else latticeconstant
    )
    fcc = Atoms(
        symbol + "4",
        positions=np.multiply(
            latticeconstant,
            [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
        ),
        cell=3 * [latticeconstant],
        pbc=False,
    )
    n = ceil((3 * natoms / (2 * np.pi)) ** (1 / 3)) + 1
    cube = fcc.repeat((n, n, n))
    cube.cell = None

    # Place center of cube at origin
    center_idx = np.linalg.norm(
        cube.positions - cube.positions.mean(axis=0), axis=1
    ).argmin()
    cube.positions -= cube.positions[center_idx]

    # Find all unique distances and corresponding cutoffs
    distances = np.linalg.norm(cube.positions, axis=1)
    cutoffs = np.unique(np.round(distances, 8))
    cutoffs = 0.5 * (cutoffs[:-1] + cutoffs[1:])

    # Find appropriate cutoff to get desired number of atoms
    masks = distances[None, :] < cutoffs[:, None]
    counts = masks.sum(axis=1)
    mask_index = np.searchsorted(counts, natoms, side="left")

    if counts[mask_index] != natoms and mask_index > 0:
        if rounding == "below":
            mask_index -= 1
        elif rounding == "closest":
            current_count = counts[mask_index]
            low_count = counts[mask_index - 1]
            if abs(current_count - natoms) > abs(low_count - natoms):
                mask_index -= 1

    mask = masks[mask_index]
    sphere: Atoms = cube[mask]  # type: ignore
    return sphere


def scan_size_fcc_spheres(
    nmin: int,
    nmax: int,
    symbol: str = "Au",
    latticeconstant: float | None = None,
) -> dict[tuple[int], Atoms]:
    if not (0 <= nmin <= nmax):
        raise ValueError("'nmin' should be non-negative and at most 'nmax'.")
    sphere_max = construct_fcc_sphere(
        symbol, nmax, rounding="above", latticeconstant=latticeconstant
    )

    # Find all unique distances and corresponding cutoffs
    distances = np.linalg.norm(sphere_max.positions, axis=1)
    cutoffs = np.unique(np.round(distances, 8))
    cutoffs[:-1] = 0.5 * (cutoffs[:-1] + cutoffs[1:])
    cutoffs[-1] += 1.0  # Ensure the last cutoff includes all atoms

    # Find appropriate cutoff to get desired number of atoms
    masks = distances[None, :] < cutoffs[:, None]
    counts = masks.sum(axis=1)
    low_index, high_index = np.searchsorted(counts, [nmin, nmax], side="left")
    high_index = high_index + 1 if counts[high_index] == nmax else high_index
    return {
        (counts[i],): sphere_max[masks[i]]
        for i in range(low_index, high_index)
    }  # type: ignore
