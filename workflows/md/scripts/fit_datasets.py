import numpy as np
import torch
from ase.io import read
from np_match.workflow.config import load_workflow_config
from np_match.workflow.fitting import (
    fit_dataset,
    get_particle_info,
    load_experimental_data,
)
from tqdm import tqdm

DEVICE = torch.device("cpu")


def main(
    fit_batch_size: int,
    iq_path: str,
    pq_index: list[int],
    job_idx: int,
    *args,
    config_path: str | None = None,
    **kwargs,
) -> tuple[bool, None]:
    config = load_workflow_config(config_path, __file__)
    qdamp = config.value("scattering", "qdamp")
    experimental_pdfs = load_experimental_data(
        config.path("experimental_data")
    )
    structures_path = config.path("structures")
    fit_list: dict[str, list[np.ndarray | str | float | int]] = {}
    q = None
    iq_list = torch.load(iq_path, map_location=DEVICE)
    iq_list = iq_list[
        pq_index[0] * fit_batch_size : (pq_index[0] + 1) * fit_batch_size
    ]
    pbar = tqdm(iq_list)
    bisos = [0.0]
    for tensor_dict in pbar:
        sim_idx = tensor_dict["sim_idx"]
        pbar.set_description(f"Processing particle {sim_idx}")
        if q is not None:
            assert torch.allclose(q, tensor_dict["q"])
        q = tensor_dict["q"]
        iq_int = torch.mean(tensor_dict["iq_int"][-10:], dim=0)  # time average
        fit_list.setdefault("iq_int", []).append(iq_int.numpy(force=True))
        size = tensor_dict["size"]
        form_avg_sq = tensor_dict["form_avg_sq"]

        np_info = get_particle_info(read(structures_path, index=sim_idx))

        assert size == np_info["size"], f"Size mismatch for particle {sim_idx}"
        assert q.ndim == iq_int.ndim == form_avg_sq.ndim == 1
        assert q.size(0) == iq_int.size(0) == form_avg_sq.size(0)
        assert isinstance(size, int)

        for exp_scat in experimental_pdfs:
            for biso in bisos:
                data_fit: dict[str, np.ndarray | str | float | int] = {}
                data_fit.update(
                    fit_dataset(
                        q,
                        iq_int,
                        form_avg_sq,
                        size,
                        exp_scat,
                        biso,
                        qdamp=qdamp,
                    )
                )
                data_fit.pop("fq")
                data_fit.update(np_info)
                data_fit["sim_idx"] = sim_idx
                data_fit["exp_sample"] = exp_scat.sample
                for k, v in data_fit.items():
                    fit_list.setdefault(k, []).append(v)

    num_exp = len(experimental_pdfs)
    num_particles = len(iq_list)

    # Convert to numpy arrays
    out = {k: np.asarray(v) for k, v in fit_list.items()}
    for k, arr in out.items():
        if k == "iq_int":
            continue
        arr.shape = (num_particles, num_exp) + arr.shape[1:]
    out["q"] = q.numpy(force=True)
    out["form_avg_sq"] = form_avg_sq.numpy(force=True)

    if "output" in kwargs:
        output = kwargs["output"]
    else:
        out_dir = config.path("fits_dir")
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"fit_{job_idx}-{pq_index[0]}.npz"
    np.savez(output, **out)
    return True, None
