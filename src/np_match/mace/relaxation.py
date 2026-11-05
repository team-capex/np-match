"""ASE geometry-relaxation settings and helpers for MACE calculators."""

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pprint import pprint
from timeit import default_timer
from typing import Any, Literal

import numpy as np
from ase import Atoms
from ase.optimize import (
    BFGS,
    FIRE,
    FIRE2,
    LBFGS,
    BFGSLineSearch,
    GoodOldQuasiNewton,
    GPMin,
    LBFGSLineSearch,
    MDMin,
    QuasiNewton,
)
from ase.optimize.optimize import Optimizer
from ase.parallel import MPI, MPI4PY, DummyMPI


@dataclass
class RelaxSettings:
    fmax: float = 0.01
    max_steps: int = 1000
    trajectory: str | None = None
    logfile: str | None = "-"
    trajinterval: int = 1
    overwrite_trajectory: bool = False
    restart_every: int = 400
    comm: MPI | DummyMPI | MPI4PY = field(default_factory=DummyMPI)
    optimizer: Literal[
        "bfgs",
        "fire",
        "fire2",
        "lbfgs",
        "bfgslinesearch",
        "goodoldquasinewton",
        "gpmin",
        "lbfgslinesearch",
        "mdmin",
        "quasinewton",
    ] = "lbfgs"
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)


def get_optimizer(name: str) -> type[Optimizer]:
    optimizers: dict[str, type[Optimizer]] = {
        "bfgs": BFGS,
        "lbfgs": LBFGS,
        "mdmin": MDMin,
        "fire": FIRE,
        "fire2": FIRE2,
        "lbfgslinesearch": LBFGSLineSearch,
        "bfgslinesearch": BFGSLineSearch,
        "goodoldquasinewton": GoodOldQuasiNewton,
        "quasinewton": QuasiNewton,
        "gpmin": GPMin,
    }
    try:
        return optimizers[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown ASE optimizer: {name}") from exc


def run_relax(
    atoms: Atoms,
    relax_settings: RelaxSettings,
    verbose: bool = True,
    observers: Iterable[tuple[Callable[..., Any], dict[str, Any]]]
    | None = None,
) -> Optimizer:
    """Optimize an ASE structure until its maximum force is below ``fmax``."""
    opt_class = get_optimizer(relax_settings.optimizer)
    dynamics = opt_class(
        atoms,
        logfile=relax_settings.logfile,
        trajectory=relax_settings.trajectory,
        loginterval=relax_settings.trajinterval,
        append_trajectory=not relax_settings.overwrite_trajectory,
        comm=relax_settings.comm,
        **relax_settings.optimizer_kwargs,
    )

    if observers is not None:
        for observer, kwargs in observers:
            dynamics.insert_observer(observer, **kwargs, dynamics=dynamics)

    if opt_class is BFGS and relax_settings.restart_every > 0:

        def restart(optimizer: Optimizer) -> None:
            optimizer.initialize()

        dynamics.attach(
            restart,
            interval=relax_settings.restart_every,
            optimizer=dynamics,
        )

    if verbose:
        print("Observers for Optimizer object:")
        pprint(dynamics.observers)
        print(
            f"Running relaxation for {relax_settings.max_steps} steps with "
            f"fmax={relax_settings.fmax} eV/Å",
            flush=True,
        )

    started = default_timer()
    dynamics.run(fmax=relax_settings.fmax, steps=relax_settings.max_steps)
    elapsed = default_timer() - started

    if verbose:
        print(f"Time used for relaxation: {elapsed:.2f} s")
        print(f"Number of steps taken: {dynamics.nsteps}")
        print(
            f"Steps per second: {dynamics.nsteps / elapsed:.4g} steps/s",
            flush=True,
        )

    if not dynamics.converged():
        max_force = np.linalg.norm(atoms.get_forces(), axis=1).max()
        warnings.warn(
            f"Relaxation did not converge after {dynamics.nsteps} steps. "
            f"Final max force: {max_force:.4f} eV/Å > "
            f"{relax_settings.fmax} eV/Å",
            stacklevel=2,
        )

    return dynamics
