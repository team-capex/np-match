import ast
import logging
from argparse import ArgumentParser
from contextlib import ExitStack
from pathlib import Path

from ase import Atoms
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.parallel import DummyMPI, world
from mpi4py import MPI
from np_match.mace import (
    MACESettings,
    RelaxSettings,
    get_calculator,
    run_relax,
)
from np_match.workflow.config import load_workflow_config
from np_match.workflow.utils import monitor_gpu
from tqdm import tqdm


def get_images(trajectory: str) -> list[Atoms]:
    if not Path(trajectory).exists() or Path(trajectory).stat().st_size == 0:
        return []
    images: list[Atoms] = read(trajectory, index=":", parallel=False)  # type: ignore
    out = []
    for atoms in images:
        if atoms.calc is None:
            return out
        out.append(atoms)
    return out


def get_relax_settings(relax_dir: Path, fmax: float):
    relax_settings = RelaxSettings(
        fmax=fmax,
        max_steps=5000,
        trajectory=None,
        logfile=str(relax_dir / f"{world.rank}_{fmax}.log"),
    )
    return relax_settings


def count_completed_batches(images: list):
    data = len(images)
    comm = MPI.COMM_WORLD
    data = comm.gather(data, root=0)
    num_prev_images = None
    if world.rank == 0:
        num_prev_images = min(data)
    num_prev_images = comm.bcast(num_prev_images, root=0)
    return num_prev_images


def main():
    parser = ArgumentParser()
    parser.add_argument("job_idx", type=int)
    parser.add_argument("batches", type=ast.literal_eval)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_workflow_config(args.config, __file__)
    hardware = config.section("hardware")
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    mace_settings = MACESettings(
        model_paths=str(config.path("model")),
        head="default",
        cuda_nb_list=hardware["cuda_nb_list"],
        device=hardware["device"],
        compile_mode=None
        if hardware["compile_mode"] == "None"
        else hardware["compile_mode"],
    )
    calc = get_calculator(mace_settings, dispersion_settings=None)

    relax_dir = config.path("relax_dir") / str(args.job_idx)
    relax_dir.mkdir(exist_ok=True, parents=True)

    fmaxs: list[float] = sorted(config.value("relax", "fmaxs"), reverse=True)
    trajectories = [relax_dir / f"{world.rank}_{fmax}.traj" for fmax in fmaxs]
    relaxed_images_fmaxs = [get_images(traj) for traj in trajectories]
    writers = [
        TrajectoryWriter(traj, comm=DummyMPI(), mode="w")
        for traj in trajectories
    ]
    relax_settings_fmaxs = [
        get_relax_settings(relax_dir, fmax) for fmax in fmaxs
    ]

    # Align number of images between ranks (relevant for restarts)
    num_prev_images = count_completed_batches(relaxed_images_fmaxs[-1])
    relaxed_images_fmaxs = [
        images[:num_prev_images] for images in relaxed_images_fmaxs
    ]

    pbar = tqdm(args.batches[num_prev_images:], disable=world.rank != 0)
    monitor_gpu(
        interval=hardware["gpu_monitor_interval"], device=hardware["device"]
    )
    with ExitStack() as stack:
        for writer in writers:
            stack.enter_context(writer)

        # Save trajectories from previous runs
        for writer, relaxed_images in zip(writers, relaxed_images_fmaxs):
            for atoms in relaxed_images:
                writer.write(atoms)

        for batch in pbar:
            sim_idx = batch[world.rank]
            atoms: Atoms = read(
                config.path("structures"),
                index=sim_idx,
                parallel=False,
            )  # type: ignore
            atoms.calc = calc
            for fmax, writer, relax_settings in zip(
                fmaxs, writers, relax_settings_fmaxs
            ):
                atoms.info["fmax"] = fmax
                run_relax(atoms, relax_settings)
                writer.write(atoms)
            world.barrier()


if __name__ == "__main__":
    main()
