"""ASE NVT molecular-dynamics settings and helpers for MACE calculators."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pprint import pprint
from timeit import default_timer
from typing import Any

import torch
from ase import Atoms, units
from ase.cell import Cell
from ase.constraints import FixCom
from ase.md import Langevin
from ase.md.logger import MDLogger
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize.optimize import Dynamics
from ase.parallel import MPI, MPI4PY, DummyMPI, world

from np_match.mace.calculator import (
    DispersionSettings,
    MACESettings,
    get_calculator,
)


@dataclass
class MDSettings:
    nsteps: int
    temperature_K: float
    timestep_fs: float
    friction_fs: float = 0.01
    trajectory: str | None = None
    logfile: str | None = "-"
    loginterval: int = 10
    trajinterval: int = 100
    overwrite_trajectory: bool = False
    random_momenta: bool = False
    rescale_cell_steps: int | None = None
    comm: MPI | DummyMPI | MPI4PY = field(default_factory=DummyMPI)


def run_md(
    atoms: Atoms,
    md_settings: MDSettings,
    verbose: bool = True,
    fixcm: bool | None = None,
    observers_before_save: Iterable[tuple[Callable[..., Any], dict[str, Any]]]
    | None = None,
    observers_after_save: Iterable[tuple[Callable[..., Any], dict[str, Any]]]
    | None = None,
    converged_func: Callable[[], bool] | None = None,
    prev_steps: int = 0,
) -> Langevin:
    """Run Langevin NVT molecular dynamics for the configured step count."""
    torch.set_float32_matmul_precision("high")

    fixcm = not atoms.constraints if fixcm is None else fixcm
    if fixcm:
        atoms.constraints.append(FixCom())

    dynamics = Langevin(
        atoms,
        timestep=md_settings.timestep_fs * units.fs,
        temperature_K=md_settings.temperature_K,
        friction=md_settings.friction_fs / units.fs,
        logfile=None,
        trajectory=None,
        append_trajectory=not md_settings.overwrite_trajectory,
        fixcm=False,
        comm=md_settings.comm,
    )
    dynamics.nsteps = prev_steps
    if converged_func is not None:
        dynamics.converged = converged_func

    if md_settings.logfile is not None:
        logger = dynamics.closelater(
            MDLogger(
                dyn=dynamics,
                atoms=atoms,
                logfile=md_settings.logfile,
                comm=md_settings.comm,
            )
        )
        dynamics.attach(logger, md_settings.loginterval)

    if md_settings.trajectory is not None:
        from ase.io.trajectory import Trajectory

        mode = "a" if not md_settings.overwrite_trajectory else "w"
        trajectory = dynamics.closelater(
            Trajectory(
                md_settings.trajectory,
                mode=mode,
                comm=md_settings.comm,
            )
        )
        dynamics.attach(
            trajectory,
            interval=md_settings.trajinterval,
            atoms=atoms,
        )

    before_save = list(observers_before_save or ())
    after_save = list(observers_after_save or ())
    if md_settings.rescale_cell_steps is not None:
        if md_settings.rescale_cell_steps < 0:
            raise ValueError("rescale_cell_steps must be non-negative.")
        if md_settings.rescale_cell_steps > 0:
            after_save.append(
                get_rescale_cell_observer(
                    atoms,
                    atoms.cell.copy(),
                    md_settings.rescale_cell_steps,
                )
            )
    for observer, kwargs in before_save:
        dynamics.insert_observer(observer, **kwargs, dynamics=dynamics)
    for observer, kwargs in after_save:
        dynamics.attach(observer, **kwargs, dynamics=dynamics)

    if verbose:
        print("Observers for Langevin object:")
        pprint(dynamics.observers)
        print("Constraints on atoms:")
        pprint(atoms.constraints)
        print(
            f"Running MD for "
            f"{md_settings.nsteps * md_settings.timestep_fs / 1000:.2f} ps "
            f"({md_settings.nsteps} steps)",
            flush=True,
        )

    started = default_timer()
    dynamics.run(md_settings.nsteps)
    elapsed = default_timer() - started

    if verbose:
        print(f"Time used for MD: {elapsed:.2f} s")
        print(f"Steps per second: {md_settings.nsteps / elapsed:.4g} steps/s")
        print(
            f"Atom-steps per second: "
            f"{len(atoms) * md_settings.nsteps / elapsed:.4g} atom-steps/s",
            flush=True,
        )
    return dynamics


def initialize_momenta(
    atoms: Atoms,
    temperature_K: float,
    comm: MPI | MPI4PY | DummyMPI = world,
) -> None:
    """Initialize Maxwell-Boltzmann momenta and remove translation/rotation."""
    try:
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=temperature_K, comm=comm
        )
    except TypeError:
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=temperature_K, communicator=comm
        )
    Stationary(atoms)
    ZeroRotation(atoms)


def prepare_atoms(
    atoms: Atoms,
    md_settings: MDSettings,
    mace_settings: MACESettings,
    dispersion_settings: DispersionSettings | None = None,
) -> None:
    """Initialize momenta as needed and attach the requested calculator."""
    if not atoms.has("momenta") or md_settings.random_momenta:
        initialize_momenta(
            atoms,
            md_settings.temperature_K,
            md_settings.comm,
        )
    atoms.calc = get_calculator(mace_settings, dispersion_settings)


def rescale_cell(
    atoms: Atoms,
    dynamics: Dynamics,
    initial_cell: Cell,
    steps: int,
) -> None:
    """Linearly change cell volume to ``atoms.info['target_volume']``."""
    if steps <= 0:
        raise ValueError("Cell-rescaling steps must be positive.")
    if dynamics.nsteps > steps:
        return
    if "target_volume" not in atoms.info:
        raise KeyError("Cell rescaling requires atoms.info['target_volume'].")
    initial_volume = initial_cell.volume
    target_volume = float(atoms.info["target_volume"])
    if initial_volume <= 0 or target_volume <= 0:
        raise ValueError("Initial and target cell volumes must be positive.")
    new_volume = (
        initial_volume
        + (target_volume - initial_volume) * dynamics.nsteps / steps
    )
    atoms.set_cell((new_volume / initial_volume) ** (1 / 3) * initial_cell)


def get_rescale_cell_observer(
    atoms: Atoms, initial_cell: Cell, steps: int
) -> tuple[Callable[..., None], dict[str, Any]]:
    """Return an ASE observer that linearly rescales the cell volume."""
    if steps <= 0:
        raise ValueError("Cell-rescaling steps must be positive.")
    return (
        rescale_cell,
        {
            "atoms": atoms,
            "initial_cell": initial_cell,
            "steps": steps,
        },
    )
