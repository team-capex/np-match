"""MACE calculator setup and the optional primitive CUDA neighbor list."""

from dataclasses import dataclass
from threading import RLock
from typing import Any

import torch
from ase.calculators.mixing import SumCalculator
from mace.calculators import MACECalculator
from mace.data import neighborhood

from np_match.neighbors import NeighbourList, PrimitiveBuilder

_NEIGHBORHOOD_PATCH_LOCK = RLock()


@dataclass
class MACESettings:
    model_paths: str
    head: str = "Default"
    device: str = "cuda"
    default_dtype: str = "float32"
    enable_cueq: bool = True
    enable_oeq: bool = False
    compile_mode: str | None = "default"
    cuda_nb_list: bool = False
    nb_list_skin: float = 2.0


@dataclass
class DispersionSettings:
    dispersion_damping: str = "zero"
    dispersion_xc: str = "pbe"
    dispersion_cutoff: float = 15.0


class MACECalculatorNB(MACECalculator):
    """MACE calculator using the bundled GPU-enabled neighbor-list builder."""

    def __init__(
        self,
        *args: Any,
        nb_list_skin: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.nb_list = NeighbourList(
            builder_type=PrimitiveBuilder,
            cutoff=self.r_max,
            skin=nb_list_skin,
        )

    def _atoms_to_batch(self, atoms: Any) -> Any:
        # MACE's graph conversion resolves get_neighborhood's module-level
        # neighbor_list symbol, so patch it only while this calculator converts.
        with _NEIGHBORHOOD_PATCH_LOCK:
            original_neighbor_list = neighborhood.neighbour_list
            neighborhood.neighbour_list = self._neighbor_list
            try:
                return super()._atoms_to_batch(atoms)
            finally:
                neighborhood.neighbour_list = original_neighbor_list

    def _neighbor_list(
        self,
        quantities: str,
        atoms: Any = None,
        cutoff: float | None = None,
        positions: Any = None,
        cell: Any = None,
        pbc: Any = None,
    ) -> tuple[Any, ...]:
        if atoms is not None:
            positions = atoms.positions
            cell = atoms.cell.array
            pbc = atoms.pbc
        if positions is None or cell is None or pbc is None:
            raise ValueError(
                "The MACE neighbor list requires positions, cell, and pbc."
            )
        if cutoff is not None and abs(float(cutoff) - self.r_max) > 1e-8:
            raise ValueError(
                f"Neighbor-list cutoff {cutoff} does not match model cutoff "
                f"{self.r_max}."
            )
        if any(quantity not in "ijS" for quantity in quantities):
            raise NotImplementedError(
                "The bundled MACE neighbor list supports only i, j, and S."
            )

        # Move inputs to device, perform neighbor list update, and return
        # results as numpy arrays
        dtype = {
            "float32": torch.float32,
            "float64": torch.float64,
        }[self.default_dtype]
        positions_tensor = torch.as_tensor(
            positions, dtype=dtype, device=self.device
        )
        cell_tensor = torch.as_tensor(cell, dtype=dtype, device=self.device)
        pbc_tensor = torch.as_tensor(pbc, dtype=torch.bool, device=self.device)
        sender, receiver, unit_shifts, _ = self.nb_list.update(
            positions=positions_tensor,
            cell=cell_tensor,
            pbc=pbc_tensor,
        )
        values = {
            "i": sender.cpu().numpy(),
            "j": receiver.cpu().numpy(),
            "S": unit_shifts.cpu().numpy(),
        }
        return tuple(values[quantity] for quantity in quantities)


def get_calculator(
    mace_settings: MACESettings,
    dispersion_settings: DispersionSettings | None = None,
) -> MACECalculator | SumCalculator:
    """Create a MACE calculator, optionally combined with D3 dispersion."""
    calculator_class = (
        MACECalculatorNB if mace_settings.cuda_nb_list else MACECalculator
    )
    calculator_kwargs = {
        "model_paths": mace_settings.model_paths,
        "device": mace_settings.device,
        "enable_cueq": mace_settings.enable_cueq,
        "enable_oeq": mace_settings.enable_oeq,
        "compile_mode": mace_settings.compile_mode,
        "default_dtype": mace_settings.default_dtype,
        "head": mace_settings.head,
    }
    if mace_settings.cuda_nb_list:
        mace_calculator = calculator_class(
            **calculator_kwargs,
            nb_list_skin=mace_settings.nb_list_skin,
        )
    else:
        mace_calculator = calculator_class(**calculator_kwargs)

    if dispersion_settings is None:
        return mace_calculator

    try:
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    except ImportError as exc:
        raise ImportError(
            "D3 dispersion requires torch-dftd. Install np-match with its "
            "d3 extra or install torch-dftd in the simulation environment."
        ) from exc

    d3_calculator = TorchDFTD3Calculator(
        device=mace_settings.device,
        damping=dispersion_settings.dispersion_damping,
        dtype=torch.float32
        if mace_settings.default_dtype == "float32"
        else torch.float64,
        xc=dispersion_settings.dispersion_xc,
        cutoff=dispersion_settings.dispersion_cutoff,
    )
    return SumCalculator([mace_calculator, d3_calculator])
