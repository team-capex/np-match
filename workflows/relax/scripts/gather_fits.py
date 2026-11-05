import numpy as np
from ase.io import iread
from np_match.workflow.config import load_workflow_config
from tqdm import tqdm


def main(*args, config_path: str | None = None, **kwargs):
    config = load_workflow_config(config_path, __file__)
    fit_dir = config.path("fits_dir")
    reference = config.section("reference")
    files = list(fit_dir.glob("fit_*.npz"))
    out = {}
    q = None
    form_avg_sq = None
    for file in tqdm(files):
        data = np.load(file)
        for k, v in data.items():
            if k == "q":
                if q is not None:
                    assert np.allclose(q, v)
                q = v
            elif k == "form_avg_sq":
                if form_avg_sq is not None:
                    assert np.allclose(form_avg_sq, v)
                form_avg_sq = v
            else:
                out.setdefault(k, []).append(v)
    for k in out:
        out[k] = np.concatenate(out[k], axis=0)

    expected_particle_count = sum(1 for _ in iread(config.path("structures")))
    assert np.all(np.isin(np.arange(expected_particle_count), out["sim_idx"]))

    # sort by sim_idx along axis=0 - already sorted by exp_sample along axis=1.
    sim_indices = out["sim_idx"][:, 0, 0]
    fmaxs = out["fmax"][:, 0, 0]
    indices = np.lexsort((fmaxs, sim_indices))  # sort by sim_idx then by fmax
    num_particles = len(np.unique(sim_indices))
    assert num_particles == expected_particle_count
    fmaxs_unq = np.unique(fmaxs)
    num_fmaxs = len(fmaxs_unq)
    _, num_exp, num_biso = out["sim_idx"].shape
    for k in out:
        out[k] = out[k][indices].reshape(
            (num_particles, num_fmaxs) + out[k].shape[1:]
        )

    assert np.all(
        out["sim_idx"] == np.arange(num_particles).reshape(-1, 1, 1, 1)
    )
    assert np.all(out["fmax"] == fmaxs_unq.reshape(1, -1, 1, 1))
    assert np.all(
        out["exp_sample"] == (np.arange(num_exp) + 1).reshape(1, 1, -1, 1)
    )
    assert np.allclose(
        out["biso"][..., :-1], np.array(0.69).reshape(1, 1, 1, 1)
    )

    out["q"] = q
    out["q_ref"] = np.arange(
        reference["q_ref_min"],
        reference["q_ref_max"] + reference["q_ref_step"] / 2,
        reference["q_ref_step"],
    )
    out["form_avg_sq"] = form_avg_sq
    np.savez(fit_dir / "fit.npz", **out)
    return True, None


if __name__ == "__main__":
    main()
