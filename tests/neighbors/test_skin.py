import pytest
import torch
from np_match.neighbors import NeighbourList, PrimitiveBuilder


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
    ids=["float32", "float64"],
)
def test_skin(dtype: torch.dtype, device: torch.device) -> None:
    tol = 1e-6 if dtype == torch.float32 else 1e-12
    nb_list = NeighbourList(
        builder_type=PrimitiveBuilder,
        cutoff=1.0,
        skin=0.5,
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    cell = torch.eye(3, dtype=dtype, device=device) * 3.0
    pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)

    assert nb_list.nupdates == 0
    nb_list.update(
        positions=positions,
        cell=cell,
        pbc=pbc,
    )
    assert nb_list.nupdates == 1

    # Make sure no new update is triggered if the same inputs are provided
    nb_list.update()
    nb_list.update(positions=positions)
    nb_list.update(cell=cell)
    nb_list.update(pbc=pbc)
    nb_list.update(positions=positions, cell=cell, pbc=pbc)
    assert nb_list.nupdates == 1

    # New positions that changed less than half the skin depth should not
    # trigger an update
    positions2 = torch.tensor(
        [
            [0.25 - tol, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    nb_list.update(positions=positions2)
    assert nb_list.nupdates == 1
    assert torch.allclose(positions, nb_list.builder.positions, atol=tol)

    # New cell inside the skin depth should not trigger an update
    cell2 = torch.tensor(
        [
            [2.5 + tol, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=dtype,
        device=device,
    )
    nb_list.update(cell=cell2)
    assert nb_list.nupdates == 1
    assert torch.allclose(cell, nb_list.builder.supercell.cell, atol=tol)

    positions3 = torch.tensor(
        [
            [0.25 + tol, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    nb_list.update(positions=positions3)
    assert nb_list.nupdates == 2
    assert torch.allclose(positions3, nb_list.builder.positions, atol=tol)

    # New cell inside the skin depth should not trigger an update
    cell3 = torch.tensor(
        [
            [2.5 - tol, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=dtype,
        device=device,
    )
    nb_list.update(cell=cell3)
    assert nb_list.nupdates == 3
    assert torch.allclose(cell3, nb_list.builder.supercell.cell, atol=tol)

    positions4 = torch.tensor(
        [
            [0.1 + 2 * tol, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    cell4 = torch.tensor(
        [
            [2.7 - 2 * tol, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=dtype,
        device=device,
    )
    nb_list.update(positions=positions4, cell=cell4)
    assert nb_list.nupdates == 3
    assert torch.allclose(positions3, nb_list.builder.positions, atol=tol)
    assert torch.allclose(cell3, nb_list.builder.supercell.cell, atol=tol)

    positions5 = torch.tensor(
        [
            [0.1, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    cell5 = torch.tensor(
        [
            [2.7, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=dtype,
        device=device,
    )
    nb_list.update(positions=positions5, cell=cell5)
    assert nb_list.nupdates == 4
    assert torch.allclose(positions5, nb_list.builder.positions, atol=tol)
    assert torch.allclose(cell5, nb_list.builder.supercell.cell, atol=tol)
