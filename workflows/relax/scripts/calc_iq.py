from pathlib import Path
from typing import Any

import torch
from ase import Atoms
from ase.io import read
from np_match.debye import BinnedSingleDebyeCalculator, compute_iq
from np_match.workflow.config import load_workflow_config
from np_match.workflow.utils import monitor_gpu
from perqueue.constants import DYNAMICWIDTHGROUP_KEY
from tqdm import tqdm


def load_pdf(pdf_file: Path) -> list:
    if not pdf_file.exists():
        return []
    try:
        return torch.load(pdf_file)
    except Exception as e:
        print(e)
        print(f"Error occurred while loading {pdf_file=}, re-computing...")
        return []


def read_relaxed_images(job_idx: int, relax_dir: Path) -> list[Atoms]:
    images: list[Atoms] = []
    for p in (relax_dir / str(job_idx)).glob("*.traj"):
        loaded_images: list[Atoms] = read(p, index=":")  # type: ignore
        fmax = float(p.stem.split("_")[-1])
        for atoms in loaded_images:
            atoms.info["fmax"] = fmax
        images.extend(loaded_images)
    images.sort(key=lambda atoms: (atoms.info["sim_idx"], atoms.info["fmax"]))
    return images


def main(
    batch_indices: list[int],
    job_idx: int,
    *args,
    config_path: str | None = None,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
    config = load_workflow_config(config_path, __file__)
    hardware = config.section("hardware")
    calc = BinnedSingleDebyeCalculator(
        **config.section("scattering"),
        device=hardware["device"],
    )
    monitor_gpu(
        interval=hardware["gpu_monitor_interval"],
        device=hardware["device"],
    )
    images = read_relaxed_images(job_idx, config.path("relax_dir"))
    num_fmaxs = 4
    assert len(images) == num_fmaxs * len(batch_indices), (
        len(images),
        len(batch_indices),
    )
    iq_path = config.path("iq_dir") / f"iq_{job_idx}.pt"
    iq_list = load_pdf(iq_path)
    iq_path.parent.mkdir(parents=True, exist_ok=True)
    for counter, atoms in enumerate(tqdm(images[len(iq_list) :])):
        iq_dict = compute_iq(atoms, calc)
        iq_dict["sim_idx"] = atoms.info["sim_idx"]
        iq_dict["fmax"] = atoms.info["fmax"]
        iq_list.append(iq_dict)
        if counter % 100 == 0:
            torch.save(iq_list, iq_path)
    torch.save(iq_list, iq_path)
    assert len(iq_list) == len(images), (len(iq_list), len(images))

    fit_batch_size = config.value("scheduler", "fit_batch_size")
    dwg_width = (len(iq_list) + fit_batch_size - 1) // fit_batch_size
    return True, {
        DYNAMICWIDTHGROUP_KEY: dwg_width,
        "job_idx": job_idx,
        "fit_batch_size": fit_batch_size,
        "batch_indices": batch_indices,
        "iq_path": str(iq_path),
    }
