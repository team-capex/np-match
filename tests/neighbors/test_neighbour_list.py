from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from np_match.neighbors import NeighbourList, PrimitiveBuilder
from np_match.neighbors.builder_base import NeighbourListBuilder
from np_match.neighbors.utils import ase_atoms_to_torch, sort_edges
from numpy.typing import NDArray
from torch import Tensor


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
    ids=["float32", "float64"],
)
@pytest.mark.parametrize(
    ("builder_type", "builder_kwargs"),
    [(PrimitiveBuilder, {})],
    ids=["primitive"],
)
def test_neighbour_list_cutoff_cases(
    dtype: torch.dtype,
    device: torch.device,
    builder_type: type[NeighbourListBuilder],
    builder_kwargs: dict[str, object],
) -> None:
    inter_dist = 0.3
    tol = 1e-6 if dtype == torch.float32 else 1e-12

    cell = torch.eye(3, dtype=dtype, device=device)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [inter_dist, 0.0, 0.0]], dtype=dtype, device=device
    )
    pbc = torch.tensor([1, 1, 1], dtype=torch.bool, device=device)

    # Small cutoff should give zero neighbors
    nb_list = NeighbourList(
        builder_type=builder_type,
        cutoff=0.1,
        **builder_kwargs,
    )
    sender, receiver, unit_shifts, distances = nb_list.update(
        positions=positions, cell=cell, pbc=pbc
    )
    assert nb_list.builder.supercell.num_cells == 27
    assert sender.dtype == torch.int64
    assert receiver.dtype == torch.int64
    assert unit_shifts.dtype == dtype
    assert distances.dtype == dtype
    assert sender.device == positions.device
    assert receiver.device == positions.device
    assert unit_shifts.device == positions.device
    assert distances.device == positions.device

    empty = torch.tensor([], dtype=dtype, device=device)
    assert torch.equal(sender, empty.view(0))
    assert torch.equal(receiver, empty.view(0))
    assert torch.equal(unit_shifts, empty.view(0, 3))
    assert torch.equal(distances, empty.view(0))

    # Two edges should be found
    nb_list = NeighbourList(
        builder_type=builder_type,
        cutoff=inter_dist + tol,
        **builder_kwargs,
    )
    sender, receiver, unit_shifts, distances = nb_list.update(
        positions=positions, cell=cell, pbc=pbc
    )
    assert nb_list.builder.supercell.num_cells == 27

    assert sender.dtype == torch.int64
    assert receiver.dtype == torch.int64
    assert unit_shifts.dtype == dtype
    assert distances.dtype == dtype
    assert sender.device == positions.device
    assert receiver.device == positions.device
    assert unit_shifts.device == positions.device
    assert distances.device == positions.device

    sender, receiver, unit_shifts, (distances,), edge_counts = sort_edges(
        sender, receiver, unit_shifts, distances
    )
    assert torch.equal(edge_counts, torch.tensor([1, 1], device=device))
    assert sender.shape == (2,)
    assert receiver.shape == (2,)
    assert torch.equal(sender, torch.tensor([0, 1], device=device))
    assert torch.equal(receiver, torch.tensor([1, 0], device=device))
    assert torch.equal(unit_shifts, torch.zeros(2, 3, device=device))
    assert torch.allclose(
        distances,
        torch.tensor([inter_dist, inter_dist], dtype=dtype, device=device),
        atol=tol,
        rtol=0,
    )

    # No periodic boundary conditions, two edges should be found, even if
    # the cutoff is larger than the cell lengths.
    nb_list = NeighbourList(
        builder_type=builder_type,
        cutoff=10,
        **builder_kwargs,
    )
    sender, receiver, unit_shifts, distances = nb_list.update(
        positions=positions,
        cell=cell,
        pbc=torch.zeros(3, dtype=torch.bool, device=device),
    )
    assert nb_list.builder.supercell.num_cells == 1

    assert sender.dtype == torch.int64
    assert receiver.dtype == torch.int64
    assert unit_shifts.dtype == dtype
    assert distances.dtype == dtype
    assert sender.device == positions.device
    assert receiver.device == positions.device
    assert unit_shifts.device == positions.device
    assert distances.device == positions.device

    sender, receiver, unit_shifts, (distances,), edge_counts = sort_edges(
        sender, receiver, unit_shifts, distances
    )
    assert torch.equal(edge_counts, torch.tensor([1, 1], device=device))
    assert sender.shape == (2,)
    assert receiver.shape == (2,)
    assert torch.equal(sender, torch.tensor([0, 1], device=device))
    assert torch.equal(receiver, torch.tensor([1, 0], device=device))
    assert torch.equal(unit_shifts, torch.zeros(2, 3, device=device))
    assert torch.allclose(
        distances,
        torch.tensor([inter_dist, inter_dist], dtype=dtype, device=device),
        atol=tol,
        rtol=0,
    )

    # Four edges should be found, two within the cell and two across cell
    # boundaries
    nb_list = NeighbourList(
        builder_type=builder_type,
        cutoff=1 - inter_dist + tol,
        **builder_kwargs,
    )
    sender, receiver, unit_shifts, distances = nb_list.update(
        positions=positions, cell=cell, pbc=pbc
    )
    assert nb_list.builder.supercell.num_cells == 27

    assert sender.dtype == torch.int64
    assert receiver.dtype == torch.int64
    assert unit_shifts.dtype == dtype
    assert distances.dtype == dtype
    assert sender.device == positions.device
    assert receiver.device == positions.device
    assert unit_shifts.device == positions.device
    assert distances.device == positions.device

    sender, receiver, unit_shifts, (distances,), edge_counts = sort_edges(
        sender, receiver, unit_shifts, distances
    )
    assert torch.equal(edge_counts, torch.ones(4, device=device))
    assert sender.shape == (4,)
    assert receiver.shape == (4,)
    assert torch.equal(
        sender,
        torch.tensor([0, 0, 1, 1], device=device),
    )
    assert torch.equal(
        receiver,
        torch.tensor([1, 1, 0, 0], device=device),
    )
    assert torch.equal(
        unit_shifts,
        torch.tensor(
            [[-1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]], device=device
        ),
    )
    assert torch.allclose(
        distances,
        torch.tensor(
            [1 - inter_dist, inter_dist, inter_dist, 1 - inter_dist],
            dtype=dtype,
            device=device,
        ),
        atol=tol,
        rtol=0,
    )

    # 16 edges should be found: 2 in the cell, 8 in neighboring y and z cells,
    # and 6 in the neighboring x cells
    nb_list = NeighbourList(
        builder_type=builder_type,
        cutoff=1 + tol,
        **builder_kwargs,
    )
    sender, receiver, unit_shifts, distances = nb_list.update(
        positions=positions, cell=cell, pbc=pbc
    )
    assert nb_list.builder.supercell.num_cells == 81

    assert sender.dtype == torch.int64
    assert receiver.dtype == torch.int64
    assert unit_shifts.dtype == dtype
    assert distances.dtype == dtype
    assert sender.device == positions.device
    assert receiver.device == positions.device
    assert unit_shifts.device == positions.device
    assert distances.device == positions.device

    sender, receiver, unit_shifts, (distances,), edge_counts = sort_edges(
        sender, receiver, unit_shifts, distances
    )
    assert torch.equal(edge_counts, torch.ones(16, device=device))
    assert sender.shape == (16,)
    assert receiver.shape == (16,)
    assert torch.equal(
        sender,
        torch.tensor(8 * [0] + 8 * [1], device=device),
    )
    assert torch.equal(
        receiver,
        torch.tensor(6 * [0] + 2 * [1] + 2 * [0] + 6 * [1], device=device),
    )
    shifts = torch.tensor(
        [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        device=device,
    )
    assert torch.equal(
        unit_shifts,
        shifts,
    )
    assert torch.allclose(
        distances,
        torch.tensor(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1 - inter_dist,
                inter_dist,
                inter_dist,
                1 - inter_dist,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            dtype=dtype,
            device=device,
        ),
        atol=tol,
        rtol=0,
    )


######################## MATSCIPY NEIGHBOUR LIST TESTS ########################
# from https://github.com/libAtoms/matscipy/blob/master/tests/test_neighbours.py
def _convert_matscipy_data(
    i: NDArray[np.int32],
    j: NDArray[np.int32],
    shift: NDArray[np.int32],
    abs_dr: NDArray[np.float64],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    sender = torch.from_numpy(i)
    receiver = torch.from_numpy(j)
    unit_shifts = torch.from_numpy(shift)
    distances = torch.from_numpy(abs_dr)
    return sender, receiver, unit_shifts, distances


def _compute_matscipy_neighbour_list(
    atoms: Atoms, cutoff: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    neighbour_list = pytest.importorskip("matscipy.neighbours").neighbour_list

    i, j, abs_dr, shift = neighbour_list("ijdS", atoms, cutoff)
    sender, receiver, unit_shifts, distances = _convert_matscipy_data(
        i, j, shift, abs_dr
    )
    sender, receiver, unit_shifts, (distances,), counts = sort_edges(
        sender, receiver, unit_shifts, distances
    )
    assert torch.equal(counts, torch.ones_like(counts))
    return sender, receiver, unit_shifts, distances


def _compute_own_neighbour_list(
    positions: Tensor,
    cell: Tensor,
    pbc: Tensor,
    cutoff: float,
    builder_type: type[NeighbourListBuilder],
    **builder_kwargs: object,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    nb_list = NeighbourList(
        builder_type=builder_type,
        cutoff=cutoff,
        **builder_kwargs,
    )
    sender, receiver, unit_shifts, distances = nb_list.update(
        positions=positions, cell=cell, pbc=pbc
    )
    sender, receiver, unit_shifts, (distances,), edge_counts = sort_edges(
        sender, receiver, unit_shifts, distances
    )
    assert torch.equal(edge_counts, torch.ones_like(edge_counts))
    return sender, receiver, unit_shifts, distances


def _check_distances(
    positions: Tensor,
    cell: Tensor,
    sender: Tensor,
    receiver: Tensor,
    unit_shifts: Tensor,
    distances: Tensor,
    tol: float,
) -> None:
    positions = positions.to(device=distances.device, dtype=torch.float64)
    cell = cell.to(device=distances.device, dtype=torch.float64)
    unit_shifts = unit_shifts.to(device=distances.device, dtype=torch.float64)
    D = positions[receiver] - positions[sender] + unit_shifts @ cell
    distances_compare = torch.linalg.norm(D, dim=1)
    assert torch.allclose(
        distances,
        distances_compare.to(distances.dtype),
        rtol=0,
        atol=tol,
    )


def _compare_neighbour_lists(
    sender: Tensor,
    receiver: Tensor,
    unit_shifts: Tensor,
    distances: Tensor,
    sender_ref: Tensor,
    receiver_ref: Tensor,
    unit_shifts_ref: Tensor,
    distances_ref: Tensor,
    tol: float,
) -> None:
    sender_ref = sender_ref.to(device=sender.device)
    receiver_ref = receiver_ref.to(device=receiver.device)
    unit_shifts_ref = unit_shifts_ref.to(device=unit_shifts.device)
    distances_ref = distances_ref.to(
        device=distances.device, dtype=distances.dtype
    )
    assert torch.equal(sender, sender_ref)
    assert torch.equal(receiver, receiver_ref)
    assert torch.equal(unit_shifts, unit_shifts_ref)
    assert torch.allclose(
        distances,
        distances_ref,
        rtol=0,
        atol=tol,
    )


@pytest.mark.parametrize(
    "pbc",
    [True, False, [True, False, True]],
    ids=["fully-periodic", "non-periodic", "partial-periodic"],
)
@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(torch.float32, 5e-4), (torch.float64, 1e-12)],
    ids=["float32", "float64"],
)
@pytest.mark.parametrize(
    ("builder_type", "builder_kwargs"),
    [(PrimitiveBuilder, {})],
    ids=["primitive"],
)
def test_neighbour_list_matches_matscipy(
    pbc: bool | list[bool],
    dtype: torch.dtype,
    tol: float,
    builder_type: type[NeighbourListBuilder],
    builder_kwargs: dict[str, object],
    device: torch.device,
) -> None:
    from ase.io import read

    atoms = read(Path(__file__).resolve().parents[1] / "aC.cfg")
    atoms.set_pbc(pbc)
    positions, cell, torch_pbc = ase_atoms_to_torch(atoms, dtype="float64")
    positions = positions.to(device=device, dtype=dtype)
    cell = cell.to(device=device, dtype=dtype)
    torch_pbc = torch_pbc.to(device=device)
    cutoff = 1.85

    sender_ref, receiver_ref, shifts_ref, distances_ref = (
        _compute_matscipy_neighbour_list(atoms, cutoff)
    )
    sender, receiver, shifts, distances = _compute_own_neighbour_list(
        positions,
        cell,
        torch_pbc,
        cutoff,
        builder_type,
        **builder_kwargs,
    )
    _check_distances(positions, cell, sender, receiver, shifts, distances, tol)
    _compare_neighbour_lists(
        sender,
        receiver,
        shifts,
        distances,
        sender_ref,
        receiver_ref,
        shifts_ref,
        distances_ref,
        tol,
    )


@pytest.mark.parametrize(
    ("pbc", "cutoff"),
    [
        (True, 1.1),
        (True, 1.5),
        (False, 1.1),
        ([True, False, False], 1.1),
        ([True, False, True], 1.1),
    ],
    ids=[
        "periodic-cutoff-1.1",
        "periodic-cutoff-1.5",
        "non-periodic",
        "x-periodic",
        "xz-periodic",
    ],
)
@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(torch.float32, 5e-4), (torch.float64, 1e-12)],
    ids=["float32", "float64"],
)
@pytest.mark.parametrize(
    ("builder_type", "builder_kwargs"),
    [(PrimitiveBuilder, {})],
    ids=["primitive"],
)
def test_small_cell_matches_matscipy(
    pbc: bool | list[bool],
    cutoff: float,
    dtype: torch.dtype,
    tol: float,
    builder_type: type[NeighbourListBuilder],
    builder_kwargs: dict[str, object],
    device: torch.device,
) -> None:
    from ase import Atoms

    atoms = Atoms("C", positions=[[0.5, 0.5, 0.5]], cell=[1, 1, 1], pbc=True)
    atoms.set_pbc(pbc)
    positions, cell, torch_pbc = ase_atoms_to_torch(atoms, dtype="float64")
    positions = positions.to(device=device, dtype=dtype)
    cell = cell.to(device=device, dtype=dtype)
    torch_pbc = torch_pbc.to(device=device)
    sender_ref, receiver_ref, shifts_ref, distances_ref = (
        _compute_matscipy_neighbour_list(atoms, cutoff)
    )
    sender, receiver, shifts, distances = _compute_own_neighbour_list(
        positions,
        cell,
        torch_pbc,
        cutoff,
        builder_type,
        **builder_kwargs,
    )
    _check_distances(positions, cell, sender, receiver, shifts, distances, tol)
    _compare_neighbour_lists(
        sender,
        receiver,
        shifts,
        distances,
        sender_ref,
        receiver_ref,
        shifts_ref,
        distances_ref,
        tol,
    )


@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(torch.float32, 5e-4), (torch.float64, 1e-8)],
    ids=["float32", "float64"],
)
@pytest.mark.parametrize(
    ("builder_type", "builder_kwargs"),
    [(PrimitiveBuilder, {})],
    ids=["primitive"],
)
def test_noncubic_cell(
    dtype: torch.dtype,
    tol: float,
    builder_type: type[NeighbourListBuilder],
    builder_kwargs: dict[str, object],
    device: torch.device,
) -> None:
    from ase.build import bulk

    atoms = bulk("Al", cubic=False)
    positions, cell, pbc = ase_atoms_to_torch(atoms, dtype="float64")
    positions = positions.to(device=device, dtype=dtype)
    cell = cell.to(device=device, dtype=dtype)
    pbc = pbc.to(device=device)
    sender, receiver, shifts, distances = _compute_own_neighbour_list(
        positions,
        cell,
        pbc,
        3.1,
        builder_type,
        **builder_kwargs,
    )
    _check_distances(positions, cell, sender, receiver, shifts, distances, tol)
    assert torch.equal(
        sender, torch.zeros(12, dtype=sender.dtype, device=device)
    )
    assert torch.equal(
        receiver, torch.zeros(12, dtype=receiver.dtype, device=device)
    )
    assert torch.allclose(
        distances,
        torch.full_like(distances, 2.86378246),
        rtol=0,
        atol=tol,
    )


@pytest.mark.parametrize("size", [1, 2, 3], ids=["unit", "2x2", "3x3"])
@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(torch.float32, 5e-4), (torch.float64, 1e-12)],
    ids=["float32", "float64"],
)
@pytest.mark.parametrize(
    ("builder_type", "builder_kwargs"),
    [(PrimitiveBuilder, {})],
    ids=["primitive"],
)
def test_hexagonal_cell(
    size: int,
    dtype: torch.dtype,
    tol: float,
    builder_type: type[NeighbourListBuilder],
    builder_kwargs: dict[str, object],
    device: torch.device,
) -> None:
    from ase.lattice.hexagonal import Graphite

    atoms = Graphite("C", latticeconstant=(2.5, 10.0), size=[size, size, 1])
    positions, cell, pbc = ase_atoms_to_torch(atoms, dtype="float64")
    positions = positions.to(device=device, dtype=dtype)
    cell = cell.to(device=device, dtype=dtype)
    pbc = pbc.to(device=device)
    sender, receiver, shifts, distances = _compute_own_neighbour_list(
        positions,
        cell,
        pbc,
        1.5,
        builder_type,
        **builder_kwargs,
    )
    _check_distances(positions, cell, sender, receiver, shifts, distances, tol)
    assert (sender.bincount() == 3).all()
    assert torch.allclose(
        distances,
        torch.full_like(distances, 2.5 / 3**0.5),
        rtol=0,
        atol=tol,
    )
