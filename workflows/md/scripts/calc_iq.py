from pathlib import Path
from typing import Any

import torch
from ase.io import read
from np_match.debye import BinnedSingleDebyeCalculator, compute_batch_iq
from np_match.workflow.config import load_workflow_config
from np_match.workflow.utils import monitor_gpu
from perqueue.constants import DYNAMICWIDTHGROUP_KEY
from tqdm import tqdm


def load_pdf(pdf_file: str | Path) -> list:
    try:
        return torch.load(pdf_file)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(e)
        print(f"Error occurred while loading {pdf_file=}, re-computing...")
        return []


def main(
    job_idx: int,
    sim_indices: list[int],
    *args,
    config_path: str | None = None,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
    config = load_workflow_config(config_path, __file__)
    scattering = config.section("scattering")
    hardware = config.section("hardware")
    calc = BinnedSingleDebyeCalculator(
        **scattering,
        device=hardware["device"],
    )
    monitor_gpu(
        interval=hardware["gpu_monitor_interval"],
        device=hardware["device"],
    )
    iq_path = config.path("iq_dir") / f"iq_{job_idx}.pt"
    iq_list = load_pdf(iq_path)
    iq_path.parent.mkdir(parents=True, exist_ok=True)
    previous_sim_indices = [iq_dict["sim_idx"] for iq_dict in iq_list]
    assert previous_sim_indices == sim_indices[: len(iq_list)], (
        previous_sim_indices,
        sim_indices,
    )
    remaining_sim_indices = sim_indices[len(iq_list) :]
    if previous_sim_indices:
        print(f"Skipping {previous_sim_indices}.")
        print(f"Remaining: {remaining_sim_indices}")
    for sim_idx in tqdm(remaining_sim_indices):
        images = read(
            config.path("md_dir") / str(sim_idx) / "md.traj",
            index=":",
        )
        iq_dict = compute_batch_iq(images, calc)
        iq_dict["size"] = iq_dict["size"][0]
        iq_dict["form_avg_sq"] = iq_dict["form_avg_sq"][0]
        iq_dict["sim_idx"] = images[0].info["sim_idx"]
        iq_list.append(iq_dict)
        torch.save(iq_list, iq_path)

    fit_batch_size = config.value("scheduler", "fit_batch_size")
    dwg_width = (len(iq_list) + fit_batch_size - 1) // fit_batch_size
    return True, {
        DYNAMICWIDTHGROUP_KEY: dwg_width,
        "job_idx": job_idx,
        "fit_batch_size": fit_batch_size,
        "iq_path": str(iq_path),
    }
