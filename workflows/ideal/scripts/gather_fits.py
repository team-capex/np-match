import numpy as np
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

    # sort by sim_idx along axis=0 - already sorted by exp_sample along axis=1.
    indices = np.argsort(out["sim_idx"][:, 0, 0])
    for k in out:
        out[k] = out[k][indices]

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
