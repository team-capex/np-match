from pathlib import Path

import numpy as np
import pytest
import torch
from ase.cluster import Icosahedron
from mace.calculators import MACECalculator
from mace.data import neighborhood
from np_match.mace import MACECalculatorNB
from np_match.neighbors.utils import sort_edges

_DTYPES = (torch.float32, torch.float64)
_DEVICES = ("cpu",) + (("cuda",) if torch.cuda.is_available() else ())
_CALCULATOR_CONFIGS = [
    (dtype, device) for dtype in _DTYPES for device in _DEVICES
]


@pytest.fixture(
    scope="module",
    params=_CALCULATOR_CONFIGS,
    ids=lambda config: f"{config[0]}-{config[1]}",
)
def mace_calculators(
    request: pytest.FixtureRequest,
) -> tuple[MACECalculator, MACECalculatorNB, torch.dtype, str]:
    dtype, device = request.param
    model_path = (
        Path(__file__).resolve().parents[1] / "model" / "mace_stagetwo.model"
    )
    calculator_options = {
        "model_paths": str(model_path),
        "device": device,
        "default_dtype": str(dtype).removeprefix("torch."),
        "enable_cueq": False,
        "enable_oeq": False,
        "compile_mode": None,
    }
    return (
        MACECalculator(**calculator_options),
        MACECalculatorNB(**calculator_options),
        dtype,
        device,
    )


def _sort_neighbor_list(
    neighbors: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    senders, receivers, shifts = neighbors
    sorted_senders, sorted_receivers, sorted_shifts, _, edge_counts = (
        sort_edges(
            torch.as_tensor(senders),
            torch.as_tensor(receivers),
            torch.as_tensor(shifts).to(torch.int64),
        )
    )
    return sorted_senders, sorted_receivers, sorted_shifts, edge_counts


@pytest.mark.parametrize(
    ("pbc", "vacuum"),
    [
        (False, 5.0),
        (True, 1.0),
        ([True, False, True], 1.0),
    ],
    ids=["non-periodic", "fully-periodic", "partially-periodic"],
)
def test_mace_neighbor_list_matches_regular_calculator(
    mace_calculators: tuple[
        MACECalculator, MACECalculatorNB, torch.dtype, str
    ],
    pbc: bool | list[bool],
    vacuum: float,
) -> None:
    (
        regular_calculator,
        neighbor_list_calculator,
        _dtype,
        _device,
    ) = mace_calculators
    atoms = Icosahedron("Au", 3)
    atoms.center(vacuum=vacuum)
    atoms.pbc = pbc

    actual_neighbors = neighbor_list_calculator._neighbor_list(
        "ijS", atoms=atoms, cutoff=neighbor_list_calculator.r_max
    )
    expected_neighbors = neighborhood.neighbour_list(
        "ijS", atoms=atoms, cutoff=neighbor_list_calculator.r_max
    )
    actual_sorted = _sort_neighbor_list(actual_neighbors)
    expected_sorted = _sort_neighbor_list(expected_neighbors)
    for actual, expected in zip(actual_sorted, expected_sorted, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    atoms.calc = regular_calculator
    regular_energy = atoms.get_potential_energy()
    regular_forces = atoms.get_forces()

    atoms.calc = neighbor_list_calculator
    neighbor_list_energy = atoms.get_potential_energy()
    neighbor_list_forces = atoms.get_forces()

    assert neighbor_list_energy == pytest.approx(
        regular_energy, rel=1e-5, abs=1e-5
    )
    force_atol = 1e-6 if _dtype == torch.float64 else 3e-3
    torch.testing.assert_close(
        torch.from_numpy(neighbor_list_forces),
        torch.from_numpy(regular_forces),
        rtol=1e-5,
        atol=force_atol,
    )
