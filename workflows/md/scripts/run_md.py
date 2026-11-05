import ast
import logging
from argparse import ArgumentParser
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read
from ase.parallel import world
from np_match.mace import (
    MACESettings,
    MDSettings,
    get_calculator,
    initialize_momenta,
    run_md,
)
from np_match.workflow.config import load_workflow_config
from np_match.workflow.utils import monitor_gpu
from tqdm import tqdm

ImageRef = tuple[str, int, int]


def check_md_status(trajectory: str, num_target_images: int):
    try:
        loaded_images = read(trajectory, index=":", parallel=False)
        if len(loaded_images) != num_target_images:
            print(f"Number of images did not match for {trajectory}")
            return False
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        if Path(trajectory).stat().st_size == 0:
            print(f"Found empty file {trajectory}")
            return False
        raise RuntimeError(f"MD trajectory is corrupt. {trajectory=}") from e


def run_md_batch(atoms: Atoms, md_settings: MDSettings, calc: Calculator):
    num_target_images = md_settings.nsteps // md_settings.trajinterval + 1
    assert md_settings.trajectory is not None
    if check_md_status(md_settings.trajectory, num_target_images):
        print(f"MD already done for {md_settings.trajectory=}, skipping")
        return
    initialize_momenta(atoms, md_settings.temperature_K, md_settings.comm)
    atoms.calc = calc
    run_md(atoms, md_settings, fixcm=False)


def get_settings(md_dir: Path, config):
    md_parameters = config.section("molecular_dynamics")
    hardware = config.section("hardware")
    mace_settings = MACESettings(
        model_paths=str(config.path("model")),
        head="default",
        device=hardware["device"],
        cuda_nb_list=hardware["cuda_nb_list"],
        compile_mode=None
        if hardware["compile_mode"] == "None"
        else hardware["compile_mode"],
    )
    md_settings = MDSettings(
        **md_parameters,
        trajectory=str(md_dir / "md.traj"),
        logfile=str(md_dir / "md.log"),
        overwrite_trajectory=True,
    )
    return mace_settings, md_settings


def main():
    parser = ArgumentParser()
    parser.add_argument("job_idx", type=int)
    parser.add_argument("batches", type=ast.literal_eval)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_workflow_config(args.config, __file__)
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    hardware = config.section("hardware")
    monitor_gpu(
        interval=hardware["gpu_monitor_interval"],
        device=hardware["device"],
    )

    mace_settings = get_settings(Path(), config)[0]
    calc = get_calculator(mace_settings, dispersion_settings=None)

    batches: list[list[ImageRef]] = args.batches
    if not batches:
        raise ValueError(f"No MD batches provided for job {args.job_idx}")
    if any(len(batch) != len(batches[0]) for batch in batches):
        raise ValueError("All MD batches in a job must have the same size")

    pbar = tqdm(batches, disable=world.rank != 0)
    for batch in pbar:
        trajectory, image_idx, expected_sim_idx = batch[world.rank]
        atoms = read(trajectory, index=image_idx, parallel=False)
        if not isinstance(atoms, Atoms):
            raise TypeError(
                f"Expected ASE Atoms in {trajectory}, image {image_idx}"
            )
        sim_idx = atoms.info.get("sim_idx")
        if sim_idx != expected_sim_idx:
            raise ValueError(
                f"Expected sim_idx {expected_sim_idx} in {trajectory}, "
                f"image {image_idx}; got {sim_idx}"
            )
        pbar.set_description(f"Job {args.job_idx}, running MD for {sim_idx=}")
        md_dir = config.path("md_dir") / str(sim_idx)
        md_dir.mkdir(parents=True, exist_ok=True)
        md_settings = get_settings(md_dir, config)[1]
        run_md_batch(atoms, md_settings, calc)
        world.barrier()
    print(f"Rank {world.rank} done")


if __name__ == "__main__":
    main()
