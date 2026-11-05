"""Generate, deduplicate, and save gold Wulff particles and FCC spheres."""

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.io import write
from scipy.spatial.distance import cdist
from sphere import scan_size_fcc_spheres
from tqdm import tqdm
from wulffpack import Decahedron, Icosahedron, SingleCrystal

# device for distance calculations when checking uniqueness of positions
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_unique_particles(
    nmin: int,
    nmax: int,
    surface_energies: dict[tuple[int, int, int], float],
    twin_energy: float,
    compute_oct: bool,
    latticeconstant: float,
) -> list[Atoms]:
    """Scan the requested Wulff families and attach construction metadata.

    ``compute_oct`` controls whether single-crystal particles are included;
    decahedral and icosahedral particles use the supplied twin energy.
    """
    primitive_structure = bulk("Au", crystalstructure="fcc", a=latticeconstant)
    particles = {
        "oct": SingleCrystal(surface_energies, primitive_structure),
        "dec": Decahedron(surface_energies, twin_energy, primitive_structure),
        "ico": Icosahedron(surface_energies, twin_energy, primitive_structure),
    }
    if not compute_oct:
        particles.pop("oct")
    images = []
    miller_indices, se = zip(*surface_energies.items())
    for fam, wulff in particles.items():
        for atoms in wulff.scan_size(nmin, nmax):
            # Metadata for later analysis
            atoms.info["wulff"] = True
            atoms.info["family"] = fam
            atoms.info["miller_indices"] = miller_indices
            atoms.info["surface_energies"] = se
            atoms.info["twin_energy"] = None if fam == "oct" else twin_energy
            images.append(atoms)
    return images


def compare_positions(pos1: np.ndarray, pos2: np.ndarray) -> bool:
    """Return whether every position in ``pos1`` matches exactly one in ``pos2``.

    Distance calculations are split into chunks to limit temporary memory use.
    """
    for rows in np.array_split(pos1, int(np.ceil(len(pos1) / 25))):
        distances = cdist(rows, pos2)
        pos_match = np.all(np.isclose(distances, 0).sum(axis=1) == 1)
        if not pos_match:
            return False
    return True


def compare_positions_torch(pos1: np.ndarray, pos2: np.ndarray) -> bool:
    """Torch-based position comparison, using the configured device."""
    a = torch.from_numpy(pos1).to(device=DEVICE)
    b = torch.from_numpy(pos2).to(device=DEVICE)
    a_splits = torch.split(a, 256)
    zero = a.new_zeros(1)
    for a_split in a_splits:
        distances = torch.cdist(a_split, b)  # Fast on GPUs
        pos_match = torch.all(
            torch.isclose(distances, zero, rtol=0.0, atol=1e-3).sum(dim=1) == 1
        )
        if not pos_match:
            return False
    return True


def same_atoms(atoms1: Atoms, atoms2: Atoms) -> bool:
    """Check structural equality, allowing atoms to be ordered differently."""
    same = (
        len(atoms1) == len(atoms2)
        and np.array_equal(atoms1.pbc, atoms2.pbc)
        and np.allclose(atoms1.cell.array, atoms2.cell.array)
        and np.array_equal(atoms1.numbers, atoms2.numbers)
    )
    # Check whether each atom has exactly one corresponding atom in
    # the reference at the same position, i.e. that the positions are
    # identical but the labeling may be different.
    if same:
        if DEVICE == torch.device("cuda"):
            same = compare_positions_torch(atoms1.positions, atoms2.positions)
        else:
            same = compare_positions(atoms1.positions, atoms2.positions)
    return same


def find_unique_images(
    images: Iterable[Atoms],
) -> tuple[list[Atoms], dict[int, list[int]]]:
    """Return unique structures and their indices grouped by atom count."""
    unique_images: list[Atoms] = []
    size_indices: dict[int, list[int]] = {}
    for atoms in tqdm(images, desc="Finding unique images"):
        same = False
        for idx_compare in size_indices.get(len(atoms), []):
            same = same_atoms(atoms, unique_images[idx_compare])
            if same:
                break
        if not same:
            size_indices.setdefault(len(atoms), []).append(len(unique_images))
            unique_images.append(atoms)
    return unique_images, size_indices


def main():
    """Generate particle scans, report size duplicates, and write trajectories."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nmin = 55
    nmax = 50_000

    # Lattice constant of bulk FCC gold from the MACE potential
    latticeconstant = 4.155328300317813

    # Curate surface energies from the Materials Project API
    # from mp_api.client import MPRester
    # api_key = "<MATERIALS_PROJECT_API_KEY>"
    # with MPRester(api_key=api_key) as mpr:
    #     surface_properties_doc = mpr.surface_properties.search()
    # surfaces = next(
    #     x for x in surface_properties_doc if x.pretty_formula == "Au"
    # ).surfaces
    # surface_energies = {}
    # for surface in surfaces:
    #     miller_index = tuple(surface.miller_index)
    #     surface_energies[miller_index] = surface.surface_energy

    # Here are the extracted surface energies for FCC gold from the Materials Project API
    surface_energies = {
        (1, 1, 0): 0.9085794742471774,
        (2, 2, 1): 0.7799619121494142,
        (1, 0, 0): 0.8605973842342504,
        (2, 1, 0): 0.9058146799329917,
        (2, 1, 1): 0.8215335200954694,
        (1, 1, 1): 0.7103681479067603,
        (3, 3, 2): 0.7481345416473041,
        (3, 1, 1): 0.8695803749005572,
        (3, 2, 2): 0.747612550095718,
        (3, 1, 0): 0.9057588277620238,
        (3, 2, 0): 0.913040766040574,
        (3, 3, 1): 0.8307824167114329,
        (3, 2, 1): 0.8482990179817306,
    }

    wulff_images = []
    for max_miller in (1, 2, 3):
        surface_energies_ = {
            k: v
            for k, v in surface_energies.items()
            if all(x <= max_miller for x in k)
        }
        twin_energies = [
            min(surface_energies_.values()) * x
            for x in [1e-8, 0.25, 0.5 - 1e-8]
        ]
        compute_oct = True
        print(f"{max_miller=}")
        print(f"{surface_energies_=}")
        print(f"{twin_energies=}")
        for t in twin_energies:
            wulff_images.extend(
                get_unique_particles(
                    nmin=nmin,
                    nmax=nmax,
                    surface_energies=surface_energies_,
                    twin_energy=t,
                    compute_oct=compute_oct,
                    latticeconstant=latticeconstant,
                )
            )
            compute_oct = False

    wulff_images_unique, _ = find_unique_images(wulff_images)

    print("Total number of Wulff-constructed particles:", len(wulff_images))
    print("Total number of unique Wulff particles:", len(wulff_images_unique))

    spheres = scan_size_fcc_spheres(
        nmin=nmin, nmax=nmax, latticeconstant=latticeconstant
    )
    for atoms in spheres.values():
        atoms.info["wulff"] = False
        atoms.info["family"] = "sph"
        atoms.info["miller_indices"] = None
        atoms.info["surface_energies"] = None
        atoms.info["twin_energy"] = None
    print("Total number of FCC spheres:", len(spheres))

    all_structures = wulff_images_unique + list(spheres.values())
    for i, atoms in enumerate(all_structures):
        atoms.info["sim_idx"] = i

    write(args.output_dir / "particles.traj", all_structures)


if __name__ == "__main__":
    main()
