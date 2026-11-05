"""Shared experimental-data loading and scattering-fit calculations."""

import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import TextIO

import h5py
import numpy as np
import torch
from ase import Atoms
from scipy.optimize import minimize, minimize_scalar
from torch import Tensor

from np_match.debye import compute_fq


@dataclass
class ScatteringData:
    q: Tensor = field(repr=False)
    fq: Tensor = field(repr=False)
    path: str = ""
    sample: int = 0


def load_experimental_data(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> list[ScatteringData]:
    path = Path(path)
    scattering_data: list[ScatteringData] = []
    with h5py.File(path, mode="r") as file:
        for key, group in file.items():
            if not key.startswith("sample_"):
                print("Disregarding key:", key)
                continue
            sample_id = int(key.split("_")[1])
            q, fq = torch.from_numpy(group["Fq"][:]).to(device=device).T
            fq /= fq.max()
            q = q.contiguous()
            fq = fq.contiguous()
            scattering_data.append(
                ScatteringData(q=q, fq=fq, path=str(path), sample=sample_id)
            )
    return scattering_data


def get_norm(fq: Tensor, fq_exp: Tensor) -> Tensor:
    """Return the least-squares scale factor for a simulated F(Q)."""
    ff = (fq_exp * fq).sum()
    f2 = (fq**2).sum()
    return ff / f2


def get_fq_loss(fq: Tensor, fq_ref: Tensor) -> tuple[float, float]:
    norm = get_norm(fq, fq_ref)
    return torch.mean(torch.square(norm * fq - fq_ref)).item(), norm.item()


def get_rwp(pred: Tensor, ref: Tensor) -> Tensor:
    return torch.sqrt(
        torch.mean(torch.square(pred - ref)) / torch.mean(torch.square(ref))
    )


def _fit_output(verbose: bool) -> TextIO:
    return sys.stdout if verbose else StringIO()


def fit_dataset(
    q: Tensor,
    iq: Tensor,
    form_avg_sq: Tensor,
    size: int,
    exp_scat: ScatteringData,
    biso: float | None,
    qdamp: float,
    verbose: bool = True,
) -> dict[str, np.ndarray | float]:
    output = _fit_output(verbose)
    with redirect_stdout(output):
        if biso is None:
            xscale, biso = fit_xscale_biso(
                q, iq, form_avg_sq, size, exp_scat, qdamp
            )
        else:
            xscale = fit_xscale(
                q, iq, form_avg_sq, size, exp_scat, biso, qdamp
            )
        fq = compute_fq(
            form_avg_sq=form_avg_sq,
            size=size,
            q=q,
            iq_int=iq,
            biso=biso,
            qdamp=qdamp,
            xscale=xscale,
            q_ref=exp_scat.q,
        )[0]
        loss, yscale = get_fq_loss(fq, exp_scat.fq)
        rwp_fq = get_rwp(yscale * fq, exp_scat.fq).item()
    return {
        "xscale": xscale,
        "yscale": yscale,
        "biso": biso,
        "mse_fq": loss,
        "rwp_fq": rwp_fq,
        "fq": yscale * fq.numpy(force=True),
    }


def fit_xscale(
    q: Tensor,
    iq: Tensor,
    form_avg_sq: Tensor,
    size: int,
    exp_scat: ScatteringData,
    biso: float,
    qdamp: float,
) -> float:
    def get_loss(xscale: float) -> float:
        fq = compute_fq(
            form_avg_sq=form_avg_sq,
            size=size,
            q=q,
            iq_int=iq,
            biso=biso,
            qdamp=qdamp,
            xscale=xscale,
            q_ref=exp_scat.q,
        )[0]
        return get_fq_loss(fq, exp_scat.fq)[0]

    xscale_scan = np.linspace(0.9, 1.1, 21)
    losses_scan = [get_loss(x) for x in xscale_scan]
    min_idx = np.argmin(losses_scan)
    xscale_min = xscale_scan[min_idx]
    print(f"xscale scan: {losses_scan[min_idx]=:.5f}, xscale={xscale_min:.3f}")

    result = minimize_scalar(
        get_loss,
        bounds=(
            xscale_scan[max(0, min_idx - 1)],
            xscale_scan[min(len(xscale_scan) - 1, min_idx + 1)],
        ),
    )
    if not result.success:
        print(result)
    if result.fun < losses_scan[min_idx]:
        print(f"fit xcsale: {result.fun=:.5f}, xscale={result.x:.5f}")
        return float(result.x)
    print(f"fit xcsale: {losses_scan[min_idx]=:.5f}, xscale={xscale_min:.5f}")
    return float(xscale_min)


def fit_xscale_biso(
    q: Tensor,
    iq: Tensor,
    form_avg_sq: Tensor,
    size: int,
    exp_scat: ScatteringData,
    qdamp: float,
) -> tuple[float, float]:
    def get_loss(parameters: tuple[float, float] | list[float]) -> float:
        xscale, biso = parameters
        fq = compute_fq(
            form_avg_sq=form_avg_sq,
            size=size,
            q=q,
            iq_int=iq,
            biso=biso,
            qdamp=qdamp,
            xscale=xscale,
            q_ref=exp_scat.q,
        )[0]
        return get_fq_loss(fq, exp_scat.fq)[0]

    xscale_scan = np.linspace(0.9, 1.1, 21)
    losses_scan = [get_loss([x, 1.0]) for x in xscale_scan]
    min_idx = np.argmin(losses_scan)
    xscale_min = xscale_scan[min_idx]
    print(f"xscale scan: {losses_scan[min_idx]=:.5f}, xscale={xscale_min:.3f}")

    bounds = [
        (
            xscale_scan[max(0, min_idx - 1)],
            xscale_scan[min(len(xscale_scan) - 1, min_idx + 1)],
        ),
        (0, 2),
    ]
    result = minimize(
        get_loss,
        bounds=bounds,
        method="Powell",
        x0=(xscale_min, 1.0),
    )
    if not result.success:
        print(result)
    print(
        f"fit xcsale+biso: {result.fun=:.5f}, "
        f"xscale={result.x[0]:.5f}, biso={result.x[1]:.5f}"
    )
    return float(result.x[0]), float(result.x[1])


def get_particle_info(
    atoms: Atoms,
) -> dict[str, str | bool | int | float]:
    info = atoms.info
    wulff = info["wulff"]
    out: dict[str, str | bool | int | float] = {
        "family": info["family"],
        "wulff": wulff,
        "size": len(atoms),
    }
    if wulff:
        out["max_miller"] = max(y for x in info["miller_indices"] for y in x)
        out["twin_111_ratio"] = (
            round(
                info["twin_energy"]
                / info["surface_energies"][
                    info["miller_indices"].index([1, 1, 1])
                ],
                2,
            )
            if info["twin_energy"] is not None
            else torch.nan
        )
    else:
        out["max_miller"] = torch.nan
        out["twin_111_ratio"] = torch.nan
    return out
