# TODO: Test with non-orthogonal cell
import pytest
import torch
from np_match.neighbors.supercell import (
    Supercell,
    get_cell_corners,
    get_cell_normal_vectors,
)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
    ids=["float32", "float64"],
)
def test_cell_corners(dtype: torch.dtype, device: torch.device) -> None:
    cell = 2 * torch.eye(3, dtype=dtype, device=device)
    cell_corners = get_cell_corners(cell)
    assert cell_corners.dtype == cell.dtype
    assert torch.allclose(
        cell_corners.unique(dim=0, sorted=True),
        torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 2],
                [0, 2, 0],
                [0, 2, 2],
                [2, 0, 0],
                [2, 0, 2],
                [2, 2, 0],
                [2, 2, 2],
            ],
            dtype=cell.dtype,
            device=device,
        ),
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
    ids=["float32", "float64"],
)
def test_normal_vectors(dtype: torch.dtype, device: torch.device) -> None:
    cell = 2 * torch.eye(3, dtype=dtype, device=device)
    cell_normals = get_cell_normal_vectors(cell)
    assert cell_normals.dtype == cell.dtype
    assert torch.allclose(
        cell_normals,
        torch.eye(3, dtype=cell.dtype, device=device),
    )

    angle = torch.tensor(30 * torch.pi / 180, dtype=dtype, device=device)
    sin = torch.sin(angle).item()
    cos = torch.cos(angle).item()
    cell = torch.tensor(
        [[1, 0, 0], [cos, sin, 0], [0, 0, 1]], dtype=dtype, device=device
    )
    cell_normals = get_cell_normal_vectors(cell)
    assert cell_normals.dtype == cell.dtype
    assert torch.allclose(
        cell_normals,
        torch.tensor(
            [[sin, -cos, 0], [0, 1, 0], [0, 0, 1]],
            dtype=dtype,
            device=device,
        ),
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
    ids=["float32", "float64"],
)
def test_orthogonal_cell_within_cutoff(
    dtype: torch.dtype, device: torch.device
) -> None:
    cell = 2 * torch.eye(3, dtype=dtype, device=device)
    pbc = torch.tensor([1, 1, 1], dtype=torch.bool, device=device)
    cutoff = 2.5
    supercell = Supercell(cell, pbc, cutoff)

    cell_repetitions = supercell.get_cell_repetitions()
    assert cell_repetitions == [2, 2, 2]
    unit_shifts = supercell.get_unit_shifts_from_cell_repetitions(
        cell_repetitions, supercell.device, supercell.dtype
    )

    unit_shifts_ref = torch.cartesian_prod(
        torch.arange(-2, 3, device=device),
        torch.arange(-2, 3, device=device),
        torch.arange(-2, 3, device=device),
    ).to(dtype=dtype)

    assert unit_shifts.dtype == dtype
    assert torch.equal(
        unit_shifts.unique(dim=0, sorted=True),
        unit_shifts_ref.unique(dim=0, sorted=True),
    )

    bbox_min, bbox_max = supercell.get_bounding_box()
    assert torch.allclose(
        bbox_min, -cutoff * torch.ones(3, dtype=dtype, device=device)
    )
    assert torch.allclose(
        bbox_max,
        cutoff * torch.ones(3, dtype=dtype, device=device) + cell.diagonal(),
    )

    min_distances = supercell.get_minimum_distances_to_cells(unit_shifts_ref)
    r2 = 2**0.5
    r3 = 3**0.5
    row2 = 2 * torch.tensor(
        [
            r3,  # -2
            r2,
            r2,
            r2,
            r3,
            r2,  # -1
            1,
            1,
            1,
            r2,
            r2,  # 0
            1,
            1,
            1,
            r2,
            r2,  # 1
            1,
            1,
            1,
            r2,
            r3,  # 2
            r2,
            r2,
            r2,
            r3,
        ],
        dtype=dtype,
        device=device,
    )
    row1 = 2 * torch.tensor(
        [
            r2,  # -2
            1,
            1,
            1,
            r2,
            1,  # -1
            0,
            0,
            0,
            1,
            1,  # 0
            0,
            0,
            0,
            1,
            1,  # 1
            0,
            0,
            0,
            1,
            r2,  # 2
            1,
            1,
            1,
            r2,
        ],
        dtype=dtype,
        device=device,
    )
    assert min_distances.shape == (125,)
    assert torch.equal(min_distances[:25], row2)
    assert torch.equal(min_distances[25:50], row1)
    assert torch.equal(min_distances[50:75], row1)
    assert torch.equal(min_distances[75:100], row1)
    assert torch.equal(min_distances[100:125], row2)

    mask = min_distances <= cutoff
    assert mask.sum() == 81
    unit_shifts_within_cutoff = unit_shifts_ref[mask]

    shift_ref = torch.zeros((81, 3), dtype=torch.int32, device=device)
    # 27 cells touching the center cell
    shift_ref[:27] = torch.cartesian_prod(
        torch.tensor([-1, 0, 1], device=device),
        torch.tensor([-1, 0, 1], device=device),
        torch.tensor([-1, 0, 1], device=device),
    )
    # 6 3x3 blocks on the faces of the 27 cell block
    c_min_3x3 = torch.tensor(
        [[-2, i, j] for i in range(-1, 2) for j in range(-1, 2)],
        device=device,
    )
    c_plus_3x3 = torch.tensor(
        [[2, i, j] for i in range(-1, 2) for j in range(-1, 2)],
        device=device,
    )
    shift_ref[27:36] = c_min_3x3
    shift_ref[36:45] = torch.roll(c_min_3x3, shifts=1, dims=1)
    shift_ref[45:54] = torch.roll(c_min_3x3, shifts=2, dims=1)
    shift_ref[54:63] = c_plus_3x3
    shift_ref[63:72] = torch.roll(c_plus_3x3, shifts=1, dims=1)
    shift_ref[72:81] = torch.roll(c_plus_3x3, shifts=2, dims=1)

    assert torch.equal(
        unit_shifts_within_cutoff.unique(dim=0, sorted=True),
        shift_ref.unique(dim=0, sorted=True),
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
    ids=["float32", "float64"],
)
def test_nonperiodic_cell_within_cutoff(
    dtype: torch.dtype, device: torch.device
) -> None:
    cell = 2 * torch.eye(3, dtype=dtype, device=device)
    pbc = torch.tensor([0, 0, 0], dtype=torch.bool, device=device)
    cutoff = 2.5
    supercell = Supercell(cell, pbc, cutoff)
    bbox_min, bbox_max = supercell.get_bounding_box()
    cell_repetitions = supercell.get_cell_repetitions()
    assert cell_repetitions == [0, 0, 0]
    unit_shifts = supercell.get_unit_shifts_from_cell_repetitions(
        cell_repetitions, supercell.device, supercell.dtype
    )

    assert torch.equal(
        unit_shifts, torch.zeros(1, 3, dtype=dtype, device=device)
    )
    assert torch.allclose(bbox_min, torch.zeros(3, dtype=dtype, device=device))
    assert torch.allclose(
        bbox_max,
        torch.tensor([2.0, 2.0, 2.0], dtype=dtype, device=device),
    )
